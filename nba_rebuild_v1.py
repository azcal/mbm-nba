"""
MBM NBA rebuild v1 -- standalone, no market inputs.
Architecture: the validated WNBA engine (player-impact minutes model, opponent-adjusted
decayed efficiency, Elo, fatigue/travel) pointed at hoopR NBA mirrors (2017-18 burn-in,
2018-19..2025-26 modeled). Tune on 2023-24 only; frozen holdouts 2024-25 and 2025-26.
Early-season shrink fitted on pre-holdout walk-forward OOS. Margin/total reference heads
validated on MAE floors.
"""
import sys, json
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss
sys.path.insert(0, "nba")
import nba_engine as E

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
YEARS = list(range(2018, 2027))

def load_games():
    rows = []
    for y in YEARS:
        s = pd.read_parquet(f"nba/sched_{y}.parquet")
        s = s[s.season_type.isin([2, 3])].copy()
        s["notes"] = s.get("notes_headline", pd.Series("", index=s.index)).fillna("")
        s = s[~s.notes.str.contains("All-Star", case=False)]
        s = s[~s.notes.str.contains("Championship", case=False)]   # NBA Cup / IST final
        s = s[s.home_score.notna() & s.away_score.notna()]
        s = s[(s.home_score.astype(float) > 0)]
        s = s[s.home_abbreviation.isin(CITY) & s.away_abbreviation.isin(CITY)]
        s["season"] = y
        rows.append(s[["game_id","game_date","game_date_time","season","season_type",
                       "home_abbreviation","away_abbreviation","home_score","away_score","neutral_site"]])
    g = pd.concat(rows, ignore_index=True)
    g["game_id"] = g.game_id.astype(str)
    g["game_date"] = pd.to_datetime(g.game_date)
    g = g.sort_values(["game_date","game_id"]).reset_index(drop=True)
    g["home_win"] = (g.home_score.astype(float) > g.away_score.astype(float)).astype(int)
    g["neutral"] = g.neutral_site.fillna(False).astype(bool)
    bubble = (g.season == 2020) & (g.game_date >= "2020-07-01")
    g.loc[bubble, "neutral"] = True
    return g

def load_team_box():
    t = pd.concat([pd.read_parquet(f"nba/team_{y}.parquet") for y in YEARS], ignore_index=True)
    t["game_id"] = t.game_id.astype(str)
    for c in ["field_goals_made","field_goals_attempted","three_point_field_goals_made",
              "free_throws_attempted","offensive_rebounds","defensive_rebounds",
              "total_turnovers","team_score","opponent_team_score"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t["poss"] = t.field_goals_attempted - t.offensive_rebounds + t.total_turnovers + 0.44*t.free_throws_attempted
    t["efg"] = (t.field_goals_made + 0.5*t.three_point_field_goals_made) / t.field_goals_attempted
    t["tovr"] = t.total_turnovers / t.poss
    return {(r.game_id, r.team_abbreviation): r for r in t.itertuples()}

def load_player_box():
    p = pd.concat([pd.read_parquet(f"nba/player_{y}.parquet") for y in YEARS], ignore_index=True)
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
        by_game[gid][tm] = grp[["athlete_id","min","gmsc","plus_minus"]].values
    return by_game

def shrink(p, min_gp, a0, cap=10):
    alpha = a0 + (1-a0)*np.minimum(min_gp, cap)/cap
    lo = np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
    return 1/(1+np.exp(-alpha*lo))

def run():
    g = load_games()
    print("games:", len(g), "| by season:", g.groupby("season").size().to_dict())
    tbox = load_team_box(); pbox = load_player_box()
    best = None
    for hl_s in (10, 15):
        for carry in (0.5, 0.7):
            for k in (16, 24):
                cfg = {"hl_fast": 5, "hl_slow": hl_s, "carry": carry, "elo_k": k, "elo_ha": 60, "tau": 150}
                df = E.add_diffs(E.build_features(g, tbox, pbox, cfg))
                tr = (df.season >= 2019) & (df.season <= 2023); va = df.season == 2024
                for C in (0.05, 0.2, 1.0):
                    p, y, _ = E.fit_eval(df, E.NEW_FEATS, tr, va, C)
                    ll = log_loss(y, np.clip(p, 1e-6, 1-1e-6))
                    if best is None or ll < best[0]: best = (ll, cfg, C, df)
    ll24, cfg, C, df = best
    print(f"tuned on 2023-24: LL={ll24:.4f}  hl_slow={cfg['hl_slow']} carry={cfg['carry']} elo_k={cfg['elo_k']} C={C}")
    scores = g.set_index("game_id")[["home_score","away_score"]].astype(float)
    df = df.merge(scores, left_on="game_id", right_index=True, how="left")
    df["margin"] = df.home_score - df.away_score
    df["tot"] = df.home_score + df.away_score
    df["min_gp"] = df[["h_gp","a_gp"]].min(axis=1)
    # shrink fit on walk-forward OOS 2021-2024
    oos = []
    for t in (2021, 2022, 2023, 2024):
        tr = (df.season >= 2019) & (df.season < t); te = df.season == t
        p, y, _ = E.fit_eval(df, E.NEW_FEATS, tr, te, C)
        rows = df.loc[te]
        oos.append(pd.DataFrame({"p": p, "y": y, "min_gp": rows.min_gp.values}))
    oos = pd.concat(oos); early = oos[oos.min_gp < 10]
    grid = np.round(np.arange(0.3, 1.01, 0.05), 2)
    a0 = float(min(grid, key=lambda a: log_loss(early.y, np.clip(shrink(early.p.values, early.min_gp.values, a),1e-6,1-1e-6))))
    print(f"shrink a0={a0} on {len(early)} OOS early games")
    json.dump({"cfg": cfg, "C": C, "a0": a0}, open("nba/spec.json", "w"))
    df.to_parquet("nba/features_best.parquet")
    MGF = E.NEW_FEATS; TTF = ["h_net_s","a_net_s","h_availstr","a_availstr","h_b2b","a_b2b","h_gp","a_gp"]
    out, preds = [], []
    for test, hi in ((2025, 2024), (2026, 2025)):
        tr = (df.season >= 2019) & (df.season <= hi); te = df.season == test
        p, y, _ = E.fit_eval(df, E.NEW_FEATS, tr, te, C)
        rows = df.loc[te]
        p = shrink(p, rows.min_gp.values, a0)
        preds.append(rows.assign(p_home=p))
        base = df.loc[tr, "home_win"].mean()
        e = 1/(1+10**(-((rows.h_elo + np.where(rows.neutral==1, 0, cfg["elo_ha"])) - rows.a_elo)/400)).values
        lbl = f"{test-1}-{str(test)[2:]}"
        res = [dict(model="coin flip", n=len(y), log_loss=0.6931, brier=0.25, hit=0.5),
               E.metrics(y, np.full_like(y, base, float), f"base rate ({base:.3f})"),
               E.metrics(y, e, "Elo (floor)"),
               {"model": "always Elo favorite", "n": len(y), "log_loss": None, "brier": None,
                "hit": round(float(np.mean((e > .5).astype(int) == y)), 4)},
               E.metrics(y, p, "REBUILD v1.1")]
        R = pd.DataFrame(res); R.insert(0, "holdout", lbl); out.append(R)
        trm = df.loc[tr]
        scm = StandardScaler().fit(trm[MGF].values)
        mm = Ridge(alpha=5.0).fit(scm.transform(trm[MGF].values), trm.margin.values)
        pm = mm.predict(scm.transform(rows[MGF].values))
        elo_m = ((rows.h_elo + np.where(rows.neutral==1,0,cfg["elo_ha"])) - rows.a_elo).values/28.0
        sct = StandardScaler().fit(trm[TTF].values)
        mt = Ridge(alpha=5.0).fit(sct.transform(trm[TTF].values), trm.tot.values)
        pt = mt.predict(sct.transform(rows[TTF].values))
        print(f"{lbl} margin MAE: model {np.mean(np.abs(pm-rows.margin)):.2f} | Elo {np.mean(np.abs(elo_m-rows.margin)):.2f} | constant {np.mean(np.abs(trm.margin.mean()-rows.margin)):.2f}")
        print(f"{lbl} total  MAE: model {np.mean(np.abs(pt-rows.tot)):.2f} | constant {np.mean(np.abs(trm.tot.mean()-rows.tot)):.2f}")
    allr = pd.concat(out, ignore_index=True)
    allr.to_csv("nba/comparison.csv", index=False)
    pd.concat(preds).to_csv("nba/holdout_predictions.csv", index=False)
    pd.set_option("display.width", 160)
    print("\n" + pd.DataFrame(allr).round(4).to_string(index=False))

if __name__ == "__main__":
    run()
