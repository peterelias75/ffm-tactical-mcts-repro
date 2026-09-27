# Phase 2 v1 archive (2026-07-19)

These ~44 candidate-selector/scorer/calibrator scripts (and their companion
artifacts in `runs/archive_phase2_v1/`) were built between 2026-06-28 and
2026-07-19 to search for profitable in-game interventions on top of
`rich_actions_3400matches_finetune_3epochs_20260712_maxlen2048/best_forward_simulator.pt`.

They are archived, not deleted, because:

- The simulator's free-running rollout MAE (32-51pp on a decode diagnostic,
  see `runs/archive_phase2_v1/phase2_rollout_realism_status_20260719.md`) was
  never close to the project's own 5pp gate for growing planner complexity.
- Three "confirmed corridor" milestones from 2026-07-18 were all
  self-retracted on 2026-07-19 (`runs/archive_phase2_v1/phase2_fresh_seed_audit_status_20260719.md`)
  when re-checked with fresh seeds — the effects being chased were within
  the noise band of the token-by-token generation process itself.
- `calibrate_tail_selector_thresholds.py`'s leave-match-out calibration
  approach is methodologically sound and worth reusing once Phase 1 is
  trustworthy again.

Do not resume iterating on this Phase 2 apparatus until the rebuilt
simulator (see `../README.md` and the project plan) actually clears its
accuracy gate on `runs/pinned_diagnostic_holdout_25.json`. The new Phase 2
plan is to score every legal action across that full pinned holdout using
the (retrained, counterfactual-aware) deterministic value head, with
pre-registered bootstrap-CI acceptance criteria, instead of hand-picking
matches and chasing single-seed deltas.
