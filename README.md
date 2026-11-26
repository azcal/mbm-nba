# MBM NBA daily engine (rebuild v1.1)

Standalone predictive model, no market inputs. Three heads, all floor-validated:
winner P(home), model margin, model total. Frozen through 2025-26; 2026-27 is the
forward test. Holdouts: 2024-25 LL 0.6051 (66.9%), 2025-26 LL 0.5991 (67.7%),
+0.013 over Elo both years. Those are the 2026-09-24 pregame-availability engine, the one
that runs live, regenerated 2026-11-26. The 0.5930 / 0.5862 printed here until then came
from the earlier build that read the game's own box score for availability, a lookahead
worth 0.013 log loss; holdout_predictions.csv, comparison.csv and the backfill rows in
logs/ were still that build's numbers and have been replaced.

Availability layer is live: ESPN NBA injury feed, players listed Out are removed from
projected actives (last-3-games rotation) before pricing. overrides.json accepts
manual out-lists per team. hoopR data mirrors refresh nightly in-season.
CALIBRATION GATE: predictions print from opening night; value lines unlock only when
EVERY team has played 8+ games. Runs twice daily. Tripwire: last-300 vs base rate.
