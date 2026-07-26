# MBM NBA daily engine (rebuild v1.1)

Standalone predictive model, no market inputs. Three heads, all floor-validated:
winner P(home), model margin, model total. Frozen through 2025-26; 2026-27 is the
forward test. Holdouts: 2024-25 LL 0.5930 (67.7%), 2025-26 LL 0.5862 (67.7%),
the largest Elo edge in the MBM stack (+0.026 both years).

Availability layer is live: ESPN NBA injury feed, players listed Out are removed from
projected actives (last-3-games rotation) before pricing. overrides.json accepts
manual out-lists per team. hoopR data mirrors refresh nightly in-season.
CALIBRATION GATE: predictions print from opening night; value lines unlock only when
EVERY team has played 8+ games. Runs twice daily. Tripwire: last-300 vs base rate.
