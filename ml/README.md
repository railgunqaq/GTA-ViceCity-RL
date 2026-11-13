# reVC RC helicopter learning

Windows tools for controlling and studying the RC helicopter in mission 19
(Demolition Man). Install dependencies using the [repository README](../README.md)
and build/deploy the matching native bridge using [BUILD.md](../docs/BUILD.md).
Game data and a compiled game executable are not included.

The verified task result is **one damage-free placement by a route planner and a
separate PD controller**. It is not a SAC policy result or a completed four-bomb
mission. The standard SAC/DQN trainer does not consume navigation waypoints.
Current horizon, stall and credit-assignment limits are documented in the
[next-stage assessment](../audit/2026-11-13/NEXT_STAGE_ASSESSMENT.md); longer
training alone does not resolve them.

## Game directory

Run commands from the repository root. Set the game directory before starting
Python; all entry points share this default:

```powershell
$env:REVC_GAME_DIR = 'C:/VC'
```

If unset, the default remains `C:/VC`. `calibrate_controls.py`,
`smoke_contract.py` and `smoke_training.py` also accept `--game-dir` for an
explicit override. Use the environment variable consistently for a run so
training metadata fingerprints refer to the same installation. Changing it
after importing the modules does not change existing Python default arguments.

## Version and control contract

| Contract | Value |
|---|---|
| Native bridge protocol | v7 |
| Environment | rc-heli-v5 |
| SAC checkpoint format | v4 |
| Obstacle semantics | collision-rays-v2 |
| Observation | 76 scalars, defined by `rl_train.OBS_SCHEMA` |
| Action | 5 values: roll, pitch, yaw, collective, release |
| Simulation step | 10 fixed 10 ms frames per action |

Deploy the matching bridge before running an integration command. Old
checkpoints are rejected when their environment, schemas, release threshold,
obstacle or reward contracts differ. Equal tensor sizes do not imply
compatibility. No trained model is distributed in this repository.

Actions are normalized to [-1, 1]. The bridge clamps sticks to +/-128 and buttons
to 0..255. Positive collective maps to Cross, negative collective to Square;
neutral collective is not a hover controller. The release component sends
Circle only when `release > 0.5` and the preceding native snapshot reports a
carried bomb. A pickup during the requested step cannot enable release
retroactively. Takeoff actions always use release=0.

The release output is a continuous Gaussian/tanh value with a threshold and
state gate. It is not a mixed discrete/continuous SAC algorithm; the entropy
term does not separately model a discrete release decision.

Observations contain target geometry, velocity, rays, health, native attitude,
mission/bomb/timer state, semantic objective identity and reward-event ledgers.
Bomb states are 0=available, 1=carried, 2=falling and 3=placed. The timer is
normalized by 420,000 ms. World Z is altitude, not ground clearance.

The 13 rays start at the helicopter center and extend 20 m. They check loaded
building/object/dummy collision geometry, including physical glass, mesh and
scaffolding. Vehicles, pedestrians and the verified carried bomb are excluded.
Unknown coverage reports zero range. Raw hit validity, identity, material and
normal are available as diagnostics but do not add inputs to the 76-scalar
policy observation. Sparse static rays are not a complete collision guarantee.

After each `step()` the simulator pauses, so learning and serialization latency
do not extend the action. `release()` restores normal time. Diagnostic `hold()`
runs freely and must not supply replay transitions. A lost controller heartbeat
releases control after five seconds; unobserved motion invalidates continuity
and requires reset.

## Events, rewards and termination

The reward contract is `verified-events-damage-no-carry-shaping-v3`, with
`progress_scope=pickup_only`:

- Native available-to-carried transition: +5 once per bomb and episode.
- Native destination completion bit 0-to-1: +25 once per destination and episode.
- Native mission-passed counter increment: +100. A valid same-epoch final
  snapshot can also pay the fourth placement's +25.
- Every step: -0.15. Same-target distance progress applies only when both
  snapshots are in the pickup phase. Carrying/falling receive no distance
  shaping; target/phase switches establish a fresh baseline.
- Health loss above 1e-6: -150 times the loss. Smaller float jitter is ignored;
  health recovery earns nothing. Actual nonnegative loss is still reported.
- Unsuccessful mission termination: -10.

`info.task_events`, `pickup_events`, `drop_events`, `stage` and `reward_terms`
record native events and their reward components. Radar-marker disappearance
is never proof of pickup, placement or success. Objective identities use
`(episodeId, objectiveKind, objectiveIndex)`; radar handles are diagnostic only.
Reward history survives radar rebuilding.

External stall/step/time limits are truncations and retain bootstrapping;
actual mission termination masks it. Missing acknowledgements, game loss,
non-finite transitions and process/episode/mission-epoch discontinuities must
not enter replay. Invalid nonterminal mission/objective telemetry is rejected.
Invalid terminal cleanup telemetry cannot fabricate pickup or placement;
the independent native success counter can still establish mission success.

Active missions and busy cleanup states reset through an owned process restart.
Reset verifies fresh mission 19 telemetry, all bombs available, zero completed
bits and a timer that has not started, then pauses at the ready barrier. Tutorial
input is cleared before readiness is accepted. A timeout is not a ready state.
Reset startup time is part of operational cost.

## Offline tests

```powershell
python -B -m unittest discover -s ml/tests -v
```

These tests use in-memory stubs, synthetic geometry and temporary checkpoints.
They do not launch or attach to the game and need no game assets. Passing them
does not establish real-game task performance.

## Opt-in integration checks

The following commands launch and clean up their own game processes. Use a new
output path for each invocation:

```powershell
python -B ml/smoke_contract.py --output ml/runs/NEW_CONTRACT.json
python -B ml/smoke_training.py --output ml/runs/NEW_SMOKE
python -B ml/calibrate_controls.py --output ml/runs/NEW_CALIBRATION --axes
```

The training smoke check runs 70 training steps, 30 resumed steps and 20
evaluation steps. Its reduced warmup/batch settings exercise gradient updates;
its checkpoints are plumbing test artifacts. Calibration records bounded
collective/axis probes and native telemetry, not task success.

## Geometry diagnostics

`navigation.py` searches a bounded route for a frozen snapshot.
`probe_navigation.py` performs a native-geometry survey, or with `--fly` follows
a verified route using feedback control. Neither is integrated into SAC/DQN.

```powershell
python -u -B ml/probe_navigation.py --output ml/runs/NEW_SURVEY --queries 15000 --nodes 10000 --steps 2400
python -u -B ml/probe_navigation.py --output ml/runs/NEW_NAVIGATION --fly --goal 1 --margin 0.15 --resolutions 2 1 --queries 7000 --steps 3800 --via-file audit/2026-11-06/first-floor-route-hints.json
```

The [building geometry report](../audit/2026-11-06/BUILDING_ENTRANCE_GEOMETRY.md)
explains why the verified first-floor route enters below the continuous upper
walls and climbs through the internal stair opening. Topology hints never waive
a native clearance failure.

Every returned segment passes a native sampled swept-sphere query. The radius
covers the helicopter, rotor and carried bomb; sample inflation covers the gaps
between checks. Queries require controller-owned lockstep and unchanged frame,
episode and game time. Unknown loading or invalid/changed snapshots cannot be
used as free space. A bounded search failure does not prove no route exists.

Flight rechecks short segments and stops on damage or other bounds. Dynamic
people/vehicles are excluded and objects may move after a query, so frozen
clearance does not guarantee future flight safety. The diagnostic uses normal
controls without teleporting or setting completion bits.

New runs write `summary.json`, `queries.jsonl` and `trace.jsonl`. Geometric route
existence, physical traversal, native placement and four-target success are
separate outcomes. The repository retains selected historical summaries; full
raw run archives are not included.

`plot_navigation.py RUN_DIRECTORY` generates an evidence plot when that run's
required files are available and Matplotlib is installed.

## Bounded learning experiments

The following is a short fresh-run example, not a recommended full-task training
configuration. It does not enable the diagnostic planner:

```powershell
python -u -B ml/rl_sac.py --total-steps 100 --minutes 5 --trace --seed 1
```

Each run creates `ml/runs/RUN_NAME/` with `config.json`, `episodes.csv`,
`checkpoint.pt` and a final `summary.json`; `--trace` adds per-step JSONL.
Runtime versions and source/game/script fingerprints are recorded. Generated
metadata can contain absolute local paths and is excluded from version control.

Use a checkpoint from your own compatible run for these separate commands:

```powershell
python -B ml/rl_sac.py --resume ml/runs/RUN_NAME/checkpoint.pt --total-steps 100 --minutes 5
python -B ml/rl_sac.py --eval --checkpoint ml/runs/RUN_NAME/checkpoint.pt --episodes 5 --eval-steps 300 --stall-limit 0 --trace
python -B ml/rl_sac.py --eval --baseline random --episodes 5 --eval-steps 300
python -B ml/rl_sac.py --eval --baseline scripted --episodes 5 --eval-steps 300
python -B ml/sac_stats.py ml/runs/RUN_NAME/episodes.csv
```

`--total-steps` counts this invocation's valid transitions, including after
resume; checkpoint cumulative counts are separate. Resume restores SAC model,
target, optimizer, temperature, replay, RNG and counters into a new output run.
It starts a fresh game episode, not a simulator snapshot. Resume requires
matching stall/progress/launch settings. DQN provides explicit warm start only
and rejects resume. Load only project-owned/trusted checkpoint files.

`--stall-limit 0` disables external stall. Increasing `--eval-steps` alone does
not. Current training still has a 1500-step episode cap; the verified placement
needed 2775 steps. The default Euclidean-distance stall can stop necessary
detours. See the assessment before scheduling long experiments.

Evaluation defaults to 16 conservative takeoff steps; `--launch-steps 0` removes
them for a separate comparison. The scripted baseline continues the vertical
command and is not a mission solver. Python/NumPy/PyTorch seeds are controlled;
game-world RNG and exact reset initial state are not.

For an explicit comparison of stall settings, `run_stall_screen.py` runs two
fresh models sequentially, each with its own step/time budget:

```powershell
python -u -B ml/run_stall_screen.py --output ml/runs/NEW_SCREEN --steps 3000 --minutes 30 --seed 1
python -u -B ml/run_policy_eval.py --source-dir ml/runs/NEW_SCREEN --output ml/runs/NEW_EVALUATION --episodes-per-block 5 --steps 300 --seed 1
```

The frozen evaluation uses ABBAABBA blocks, verifies checkpoint hashes and owns
its game cleanup. Five episodes per block yield 20 episodes per model, each at
300 steps with no stall cutoff. These are early-behavior comparisons, not full
mission evaluations. Environment errors and incomplete budgets must be reported;
only equal collected budgets support an equal-budget comparison.

`analyze_runs.py` and the `curve*.py` aliases require explicit CSV paths and split
run IDs or episode-counter restarts. Distance/proximity statistics are distinct
from native task success. `summarize_policy_eval.py` analyzes detailed evaluation
runs produced locally; no historical pretrained checkpoints are bundled.
