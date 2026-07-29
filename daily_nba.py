"""
MBM NBA daily runner (rebuild v1.1, frozen spec).
- Downloads hoopR parquet history fresh each run; current-season file refreshes nightly
- Slate from hoopR schedule with ESPN scoreboard fallback (reg/postseason only)
- Availability: ESPN NBA injury feed (players listed Out removed from projected actives)
- Boards upcoming games (2-day horizon): p(home), model margin, model total
- CALIBRATION GATE: value lines suppressed until EVERY team has 8+ GP
- Grades logged predictions, last-300 tripwire, Telegram alerts
Usage: python daily_nba.py [--date YYYY-MM-DD]
"""
import argparse, json, math, os, sys, datetime as dt
import numpy as np, pandas as pd, requests
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tg_delta
import spread
import engine_nba as E

CITY = {"ATL":(33.75,-84.39),"BOS":(42.37,-71.06),"BKN":(40.68,-73.97),"CHA":(35.23,-80.84),
 "CHI":(41.88,-87.67),"CLE":(41.50,-81.69),"DAL":(32.79,-96.81),"DEN":(39.75,-105.01),
 "DET":(42.34,-83.06),"GS":(37.77,-122.39),"HOU":(29.76,-95.36),"IND":(39.76,-86.16),
 "LAC":(34.04,-118.27),"LAL":(34.04,-118.27),"MEM":(35.14,-90.05),"MIA":(25.78,-80.19),
 "MIL":(43.04,-87.92),"MIN":(44.98,-93.28),"NO":(29.95,-90.08),"NY":(40.75,-73.99),
 "OKC":(35.46,-97.51),"ORL":(28.54,-81.38),"PHI":(39.90,-75.17),"PHX":(33.45,-112.07),
 "POR":(45.53,-122.67),"SA":(29.43,-98.44),"SAC":(38.58,-121.50),"TOR":(43.64,-79.38),
 "UTAH":(40.77,-111.90),"WSH":(38.90,-77.02)}
E.CITY = CITY
E.norm_abbr = lambda x: x

H = {"User-Agent": "Mozilla/5.0"}
HOOPR = "https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main/nba"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
DATA = os.path.join(HERE, "data")
LOGS = os.path.join(HERE, "logs")
SPEC = json.load(open(os.path.join(HERE, "spec.json")))
CFG, C, A0 = SPEC["cfg"], SPEC["C"], SPEC["a0"]
TRAIN_LO, TRAIN_HI = 2019, 2026
CAL_MIN_GP = 8
MIN_EDGE = 0.05

SP_LINES = [-7.5, -3.5, -0.5, 3.5, 7.5]     # margin thresholds logged for grading
SP_COLS = ["p_sp_m75", "p_sp_m35", "p_sp_m05", "p_sp_p35", "p_sp_p75"]


def spread_ladder(i, ph, Ph, Pa, exp, home, away, span=4):
    """Quoted lines with both sides priced. A book quotes the handicap added to a team's
    score, which is the negative of the margin threshold for the home side."""
    out = [f"    {'line':>7}  {home + ' covers':>14} {'needs':>7}   {away + ' covers':>14} {'needs':>7}"]
    for q, p, o1, p2, o2 in spread.ladder(ph[i], Ph[i], Pa[i], exp[i], min_odds, span=span):
        out.append(f"    {q:>+7.1f}  {p:>14.3f} {o1:>7.2f}   {p2:>14.3f} {o2:>7.2f}")
    return out


def min_odds(p):
    p = min(max(float(p), 0.01), 0.99)
    return math.ceil((1 + MIN_EDGE) / p * 100) / 100.0

def season_end_year(d):
    return d.year + 1 if d.month >= 8 else d.year

def download_data(cur):
    os.makedirs(DATA, exist_ok=True)
    for y in list(range(2018, 2027)) + ([cur] if cur > 2026 else []):
        for kind, path in (("sched", f"schedules/parquet/nba_schedule_{y}.parquet"),
                           ("team", f"team_box/parquet/team_box_{y}.parquet"),
                           ("player", f"player_box/parquet/player_box_{y}.parquet")):
            fn = os.path.join(DATA, f"{kind}_{y}.parquet")
            fresh = (y >= cur - 0) or not os.path.exists(fn)
            if y >= cur or not os.path.exists(fn):
                try:
                    r = requests.get(f"{HOOPR}/{path}", headers=H, timeout=300)
                    if r.status_code == 200:
                        open(fn, "wb").write(r.content)
                except Exception:
                    pass

def _years():
    ys = []
    for y in range(2018, 2035):
        if os.path.exists(os.path.join(DATA, f"sched_{y}.parquet")):
            ys.append(y)
    return ys

def load_games(years):
    rows = []
    for y in years:
        s = pd.read_parquet(os.path.join(DATA, f"sched_{y}.parquet"))
        s = s[s.season_type.isin([2, 3])].copy()
        s["notes"] = s.get("notes_headline", pd.Series("", index=s.index)).fillna("")
        s = s[~s.notes.str.contains("All-Star", case=False)]
        s = s[~s.notes.str.contains("Championship", case=False)]
        s = s[s.home_abbreviation.isin(CITY) & s.away_abbreviation.isin(CITY)]
        s["season"] = y
        rows.append(s[["game_id","game_date","game_date_time","season","season_type",
                       "home_abbreviation","away_abbreviation","home_score","away_score","neutral_site"]])
    g = pd.concat(rows, ignore_index=True)
    g["game_id"] = g.game_id.astype(str)
    g["game_date"] = pd.to_datetime(g.game_date)
    g["home_score"] = pd.to_numeric(g.home_score, errors="coerce")
    g["away_score"] = pd.to_numeric(g.away_score, errors="coerce")
    played = g.home_score.notna() & (g.home_score > 0)
    g.loc[~played, ["home_score", "away_score"]] = np.nan
    g["home_win"] = np.where(g.home_score.notna(), (g.home_score > g.away_score).astype(float), np.nan)
    g["neutral"] = g.neutral_site.fillna(False).astype(bool)
    bubble = (g.season == 2020) & (g.game_date >= "2020-07-01")
    g.loc[bubble, "neutral"] = True
    return g.sort_values(["game_date","game_id"]).reset_index(drop=True)

def load_team_box(years):
    t = pd.concat([pd.read_parquet(os.path.join(DATA, f"team_{y}.parquet")) for y in years],
                  ignore_index=True)
    t["game_id"] = t.game_id.astype(str)
    for c in ["field_goals_made","field_goals_attempted","three_point_field_goals_made",
              "free_throws_attempted","offensive_rebounds","defensive_rebounds",
              "total_turnovers","team_score","opponent_team_score"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t["poss"] = t.field_goals_attempted - t.offensive_rebounds + t.total_turnovers + 0.44*t.free_throws_attempted
    t["efg"] = (t.field_goals_made + 0.5*t.three_point_field_goals_made) / t.field_goals_attempted
    t["tovr"] = t.total_turnovers / t.poss
    return {(r.game_id, r.team_abbreviation): r for r in t.itertuples()}

def load_player_box(years):
    from collections import defaultdict
    p = pd.concat([pd.read_parquet(os.path.join(DATA, f"player_{y}.parquet")) for y in years],
                  ignore_index=True)
    p["game_id"] = p.game_id.astype(str)
    p["min"] = p.minutes.map(E.parse_min)
    for c in ["points","field_goals_made","field_goals_attempted","free_throws_attempted",
              "free_throws_made","offensive_rebounds","defensive_rebounds","steals","assists",
              "blocks","fouls","turnovers","plus_minus"]:
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0.0)
    p["gmsc"] = (p.points + 0.4*p.field_goals_made - 0.7*p.field_goals_attempted
                 - 0.4*(p.free_throws_attempted - p.free_throws_made) + 0.7*p.offensive_rebounds
                 + 0.3*p.defensive_rebounds + p.steals + 0.7*p.assists + 0.7*p.blocks
                 - 0.4*p.fouls - p.turnovers)
    by_game = defaultdict(dict)
    for (gid, tm), grp in p.groupby(["game_id","team_abbreviation"]):
        by_game[gid][tm] = [(str(a), m, gs, pm) for a, m, gs, pm in
                            grp[["athlete_id","min","gmsc","plus_minus"]].values]
    return by_game

def fetch_slate_espn(dates):
    out = []
    for d in dates:
        try:
            r = requests.get(f"{ESPN}/scoreboard", params={"dates": d.strftime("%Y%m%d")},
                             headers=H, timeout=30).json()
        except Exception:
            continue
        for e in r.get("events", []):
            if int((e.get("season") or {}).get("type", 0)) not in (2, 3):
                continue
            c = e["competitions"][0]
            t = {x["homeAway"]: x for x in c["competitors"]}
            hb, ab = t["home"]["team"]["abbreviation"], t["away"]["team"]["abbreviation"]
            if hb not in CITY or ab not in CITY:
                continue
            if bool(e["status"]["type"].get("completed")):
                continue
            out.append(dict(game_id=str(e["id"]), date=pd.Timestamp(d.date()),
                            home=hb, away=ab, neutral=bool(c.get("neutralSite"))))
    return out

def fetch_out_ids():
    out = {}
    try:
        r = requests.get(f"{ESPN}/injuries", headers=H, timeout=30).json()
        teams = requests.get(f"{ESPN}/teams", headers=H, timeout=30).json()
        id2ab = {t["team"]["id"]: t["team"]["abbreviation"]
                 for t in teams["sports"][0]["leagues"][0]["teams"]}
        for tm in r.get("injuries", []):
            ab = id2ab.get(str(tm.get("id")))
            outs = set()
            for inj in tm.get("injuries", []):
                if str(inj.get("status", "")).lower() == "out":
                    aid = str(inj.get("athlete", {}).get("id", ""))
                    if aid:
                        outs.add(aid)
            if ab:
                out[ab] = outs
    except Exception as ex:
        print("injury feed unavailable, using last-rotation fallback:", ex)
    ov = os.path.join(HERE, "overrides.json")
    if os.path.exists(ov):
        for ab, ids in json.load(open(ov)).get("out", {}).items():
            out.setdefault(ab, set()).update(str(x) for x in ids)
    return out

def expected_actives(pbox, g, team, out_ids):
    games = g[((g.home_abbreviation == team) | (g.away_abbreviation == team)) & g.home_score.notna()]
    recent = games.sort_values("game_date").game_id.tail(3).tolist()
    seen = {}
    for gid in recent:
        for aid, m, gs, pm in pbox.get(gid, {}).get(team, []):
            if m and m > 0:
                seen[str(aid)] = max(seen.get(str(aid), 0.0), float(m))
    return [(aid, m, 0.0, 0.0) for aid, m in seen.items() if aid not in out_ids]

def run(today_str):
    today = pd.Timestamp(today_str)
    cur = season_end_year(today)
    download_data(cur)
    years = _years()
    g = load_games(years)
    tbox = load_team_box(years)
    pbox = load_player_box(years)

    horizon = [today + pd.Timedelta(days=i) for i in range(0, 3)]
    sched_up = g[g.home_win.isna() & g.game_date.isin([h.normalize() for h in horizon])]
    slate = [dict(game_id=r.game_id, date=r.game_date, home=r.home_abbreviation,
                  away=r.away_abbreviation, neutral=bool(r.neutral)) for r in sched_up.itertuples()]
    have = {s["game_id"] for s in slate}
    for s in fetch_slate_espn(horizon):
        if s["game_id"] not in have:
            slate.append(s)
    out_ids = fetch_out_ids()

    pseudo = []
    for s in slate:
        pbox[s["game_id"]] = {tm: expected_actives(pbox, g, tm, out_ids.get(tm, set()))
                              for tm in (s["home"], s["away"])}
        pseudo.append({"game_id": s["game_id"], "game_date": s["date"], "game_date_time": s["date"],
                       "season": cur, "season_type": 2, "home_abbreviation": s["home"],
                       "away_abbreviation": s["away"], "home_score": np.nan, "away_score": np.nan,
                       "neutral_site": s["neutral"], "home_win": np.nan, "neutral": s["neutral"]})
    g_hist_only = g[~g.game_id.isin({s["game_id"] for s in slate})]
    g_all = pd.concat([g_hist_only, pd.DataFrame(pseudo)], ignore_index=True) if pseudo else g
    g_all = g_all.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    df = E.add_diffs(E.build_features(g_all, tbox, pbox, CFG))
    scores = g_all.set_index("game_id")[["home_score", "away_score"]]
    df = df.merge(scores, left_on="game_id", right_index=True, how="left")
    df["margin"] = df.home_score - df.away_score
    df["tot"] = df.home_score + df.away_score
    tr = (df.season >= TRAIN_LO) & (df.season <= TRAIN_HI) & df.home_win.notna()
    sc = StandardScaler().fit(df.loc[tr, E.NEW_FEATS].values)
    mdl = LogisticRegression(C=C, max_iter=2000).fit(
        sc.transform(df.loc[tr, E.NEW_FEATS].values), df.loc[tr, "home_win"].astype(int).values)
    trm = df.loc[tr]
    MGF = E.NEW_FEATS
    TTF = ["h_net_s","a_net_s","h_availstr","a_availstr","h_b2b","a_b2b","h_gp","a_gp"]
    scm = StandardScaler().fit(trm[MGF].values)
    mm = Ridge(alpha=5.0).fit(scm.transform(trm[MGF].values), trm.margin.values)
    sct = StandardScaler().fit(trm[TTF].values)
    mt = Ridge(alpha=5.0).fit(sct.transform(trm[TTF].values), trm.tot.values)
    base = float(df.loc[tr, "home_win"].mean())
    # Spread head. Conditional on the winner, so P(margin>0) equals the moneyline number
    # exactly. Magnitude weights are fitted, not hardcoded.
    SH = spread.fit(df.assign(margin=df.margin), E.NEW_FEATS, tr & df.margin.notna(), C)

    done_cur = g[(g.season == cur) & g.home_win.notna()]
    counts = pd.concat([done_cur.home_abbreviation, done_cur.away_abbreviation]).value_counts()
    gate_open = len(counts) >= 30 and counts.min() >= CAL_MIN_GP

    alerts = []
    os.makedirs(LOGS, exist_ok=True)
    pred_path = os.path.join(LOGS, "predictions.csv")
    log = pd.read_csv(pred_path, dtype={"game_id": str}) if os.path.exists(pred_path) else \
        pd.DataFrame(columns=["date","game_id","home","away","p_home","p_elo","model"])


    # One-time backfill: fill the spread columns for games logged before the spread head
    # existed, so the dashboard has history to grade instead of waiting a season. Self
    # healing, runs once, then the condition is false forever after.
    need_bf = [c for c in SP_COLS + ["sp_margin"] if c not in log.columns
               or log[c].isna().all()]
    if need_bf and len(log):
        try:
            have = df[df.game_id.astype(str).isin(log.game_id.astype(str))]
            if len(have):
                bph, bPh, bPa = spread.price(SH, have)
                bf = pd.DataFrame({"game_id": have.game_id.astype(str).values,
                                   "sp_margin": spread.expected_margin(bph, bPh, bPa)})
                for nm, k in zip(SP_COLS, SP_LINES):
                    bf[nm] = spread.cover(bph, bPh, bPa, k)
                log = log.drop(columns=[c for c in bf.columns if c != "game_id"
                                        and c in log.columns], errors="ignore")
                log["game_id"] = log.game_id.astype(str)
                log = log.merge(bf, on="game_id", how="left")
                print(f"spread backfill: filled {bf.sp_margin.notna().sum()} historical rows")
        except Exception as ex:
            print(f"spread backfill skipped: {type(ex).__name__}: {ex}")

    board = None
    if slate:
        rows = df[df.game_id.isin([s["game_id"] for s in slate])].copy()
        p = mdl.predict_proba(sc.transform(rows[E.NEW_FEATS].values))[:, 1]
        mg = rows[["h_gp","a_gp"]].min(axis=1).values
        alpha = A0 + (1 - A0)*np.minimum(mg, 10)/10.0
        lo = np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
        p = 1/(1+np.exp(-alpha*lo))
        margins = mm.predict(scm.transform(rows[MGF].values))
        totals = mt.predict(sct.transform(rows[TTF].values))
        pe = 1/(1+10**(-((rows.h_elo + np.where(rows.neutral==1, 0, CFG["elo_ha"])) - rows.a_elo)/400))
        sp_ph, sp_Ph, sp_Pa = spread.price(SH, rows)
        sp_exp = spread.expected_margin(sp_ph, sp_Ph, sp_Pa)
        board = pd.DataFrame({"date": pd.to_datetime(rows.date).dt.strftime("%Y-%m-%d"),
                              "game_id": rows.game_id.astype(str), "home": rows.home,
                              "away": rows.away, "p_home": p, "p_elo": pe.values,
                              "sp_margin": sp_exp, "model": "v1"})
        for nm, k in zip(SP_COLS, SP_LINES):
            board[nm] = spread.cover(sp_ph, sp_Ph, sp_Pa, k)
        log = pd.concat([log[~log.game_id.isin(board.game_id)], board], ignore_index=True)
        gate_txt = "" if gate_open else f"  [CALIBRATION: no betting until every team has {CAL_MIN_GP} GP]"
        print(f"\n=== MBM NBA board ==={gate_txt}")
        for bi, ((_, r), tot) in enumerate(zip(board.iterrows(), totals)):
            mgn = sp_exp[bi]
            side = f"{r.home} by {mgn:.1f}" if mgn > 0 else f"{r.away} by {-mgn:.1f}"
            vals = (f"value: {r.home} >= {min_odds(r.p_home):.2f} / {r.away} >= {min_odds(1-r.p_home):.2f}"
                    if gate_open else "CALIBRATION: predictions only")
            print(f"{r.away:>4} @ {r.home:<4}  p(home)={r.p_home:.3f}  model: {side}, total {tot:.0f}  | {vals}")
            if gate_open:
                for ln in spread_ladder(bi, sp_ph, sp_Ph, sp_Pa, sp_exp, r.home, r.away):
                    print(ln)
    else:
        print("no games in the next 2 days")

    fin = g.set_index("game_id")["home_win"]
    log["result"] = log.game_id.map(fin)
    # The spread head is graded against the actual margin. `result` here is a win/loss
    # flag, which cannot answer "did the home side win by four or more".
    mrg = df.dropna(subset=["margin"]).drop_duplicates("game_id").set_index(
        df.dropna(subset=["margin"]).drop_duplicates("game_id").game_id.astype(str))["margin"]
    log["final_margin"] = log.game_id.astype(str).map(mrg)
    log.to_csv(pred_path, index=False)
    dec = log[log.result.notna()]
    if len(dec) >= 30:
        y = dec.result.astype(int).values
        print(f"\nrolling grade (n={len(y)}):")
        for lbl, pp in (("model v1.1", dec.p_home), ("elo floor ", dec.p_elo),
                        ("base rate ", pd.Series(np.full(len(y), base)))):
            pc = np.clip(pp.values.astype(float), 1e-6, 1-1e-6)
            print(f"{lbl}: LL={log_loss(y, pc):.4f}  Brier={brier_score_loss(y, pc):.4f}  "
                  f"hit={np.mean((pc > .5) == (y == 1)):.3f}")
        w = dec.tail(300)
        if len(w) >= 300:
            yy = w.result.astype(int).values
            edge = log_loss(yy, np.full(len(yy), base)) - \
                log_loss(yy, np.clip(w.p_home.values.astype(float), 1e-6, 1-1e-6))
            print(f"last-300 edge vs base: {edge:+.4f}  [{'OK' if edge > 0 else 'TRIPWIRE: bench review'}]")
            if edge <= 0:
                alerts.append("NBA winner")
        dec.to_csv(os.path.join(LOGS, "graded.csv"), index=False)

    tg, ch = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT")
    if tg and ch and (board is not None or alerts):
        push = {}
        if board is not None:
            for _, r in board.iterrows():
                line = (f"{r.away} @ {r.home}: {r.p_home:.3f} | "
                        f"{r.home} >= {min_odds(r.p_home):.2f} / {r.away} >= {min_odds(1-r.p_home):.2f}"
                        if gate_open else
                        f"{r.away} @ {r.home}: {r.p_home:.3f} | CALIBRATION, no betting")
                push[r.game_id] = {"line": line, "min_odds": min_odds(r.p_home)}
        mode, lines = tg_delta.gate(LOGS, dt.date.today().strftime("%Y%m%d"), push)
        if not (lines or alerts):
            return
        txt = "\n".join(lines) if lines else "no price changes since the last push"
        if mode == "delta" and lines:
            txt = "changes since the last push:\n" + txt
        if alerts:
            txt += "\n\u26a0 TRIPWIRE: " + ", ".join(alerts) + " below base rate, stop betting this board"
        requests.post(f"https://api.telegram.org/bot{tg}/sendMessage",
                      data={"chat_id": ch, "text": f"MBM NBA\n{txt}"}, timeout=20)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    run(ap.parse_args().date)
