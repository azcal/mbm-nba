# NBA rebuild v1.1

Architecture: the WNBA-validated engine (player-impact minutes model, opponent-adjusted
decayed efficiency, Elo, fatigue/travel/b2b, availability from actives) pointed at
hoopR NBA mirrors. 11,503 games, 2017-18 burn-in, 2018-19..2025-26 modeled.
All-Star and NBA Cup Championship games excluded; 2020 bubble games neutral-site.
Tuned on 2023-24 only (hl_slow=15, carry=0.5, K=16, C=1.0); shrink a0=0.5 fitted on
626 walk-forward OOS early-season games. Frozen holdouts:

  2024-25 n=1,314: LL 0.5930 / Brier 0.2040 / hit 67.7%  (Elo 0.6186 / 65.1%, base 0.6894)
  2025-26 n=1,315: LL 0.5862 / Brier 0.2013 / hit 67.7%  (Elo 0.6125 / 67.0%, base 0.6872)

Largest Elo edge in the stack (+0.026 both years). Reference heads, all clear:
  margin MAE 10.89 / 11.18 vs Elo 11.40 / 11.86 vs constant 12.65 / 13.16
  total  MAE 15.63 / 16.29 vs constant 16.03 / 16.39

Board ships all three heads. Runner: hoopR mirrors refresh nightly, availability from
the ESPN NBA injury feed (same convention as WNBA), calibration gate = every team 8 GP.
