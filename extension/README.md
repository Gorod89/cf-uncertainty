# Extension of the cf-uncertainty study (work towards the journal article)

Everything of the six extension items (preview sweep, cost table, protocol framing, shortening, reimplementation
check, second training domain) lives in this directory. `paper/v13` and its supplement are the frozen first
version of the article and are never edited here; the extended manuscript is a new numbered version under `extension/paper/`
(`v14` and up), copied from `paper/v13` when the text work starts.

| Path | Content |
|---|---|
| `docs/PLAN.md` | the plan: questions, designs, reading rules fixed before the runs, cost estimates, order of work |
| `docs/decisions.md` | decisions and deviations recorded while running (appended, dated) |
| `configs/experiment/` | configurations of the new roots and E3 cells (`base:` points at `configs/experiment/` of the repository) |
| `scripts/` | new scripts (preview sweep, cost benchmark, FollowNet-protocol check; the strengthening items of `docs/PLAN.md` section S: `calibrate_idm_modes.py`, `idm_ref_rerate.py`, `state_reweight.py`, `conditional_coverage.py`, `string_stability.py`, `replan_tradeoff.py`, `closed_loop_compare.py`, the queues `queue2.sh` and `queue3.sh`); the runners of `scripts/` of the repository are reused |
| `artifacts/` | roots of the new runs (`ext_*`); the roots of `artifacts/` of the repository are read, never rewritten |
| `paper/` | `v14/`: the extended manuscript (created from `paper/v13` at the start of the text work) |

Conventions of the repository apply (`docs/design.md` section 1): `dv = v - v_lead`, the one shared environment,
every claim of the text traced to a table, nothing of the existing artefacts invalidated (the new options of the
engine and of the diffusion sampler default to the stored behaviour, checked by the tests).
