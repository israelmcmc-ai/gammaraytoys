# cosimita implementation — coordination state

Live status of the work described in `docs/dev/inertial_sim_plan.md`. Updated by the
orchestrating session as each PR moves. If you are a fresh session picking this up,
read the plan first, then this file.

## Cadence

**Strictly one PR at a time**, in the order below. Each PR is reviewed and merged by
the maintainer before the next begins. The maintainer reviews every PR; agents never
merge and never open PRs — the orchestrator opens them after its own verification.

Per PR, three agents with separate roles:

| Role | Model | Does |
|---|---|---|
| Implementer | Sonnet (Opus for PR 3, PR 5) | Code + docstrings. Keeps the existing suite green. Writes no new physics tests. |
| Test author | Sonnet | First-principles tests. **Must derive expected values from the plan's formulas, never from running the code.** |
| Reviewer | Opus | Reviews implementation + tests against the plan and its §8 traps. |

From PR 3 onward the implementer and test author run **concurrently**, both working
only from the plan's API spec, so the tests are genuinely independent of the
implementation rather than anchored on it.

The orchestrator independently re-verifies every claim before opening a PR, including
mutation-testing the new tests by injecting deliberate bugs.

## Status

| PR | Scope | Branch | State |
|---|---|---|---|
| 1 | Source hierarchy + `simulated_rate()` | `claude/cosimita-pr1-source-hierarchy` | In review |
| 2 | `Earth`, `SpacecraftHistory`, orbits | `claude/cosimita-pr2-spacecraft-history` | In review |
| 3 | `InertialSimulator`, transforms, occultation | — | Not started |
| 4 | `NearPointSource`, `ExtendedSource` | — | Not started |
| 5 | `EarthAlbedoSource` | — | Not started |
| 6 | Time-dependent scaling + event CSV I/O | — | Not started |
| 7 | YAML configuration | — | Not started |

Branch naming: `claude/cosimita-prN-<topic>`, always cut from `main`.

## Decisions already taken

- The doppler-broadening inconsistency in `tracker_2d.py` is **deliberate**. See trap 11.
- Layers stay infinitesimal planes. The near-horizontal blind wedge is expected.
- `SimpleTraditionalReconstructor`'s asymmetric up-going acceptance is correct as-is.
- Eccentricity defaults to 0 (circular). `e = 1` is parabolic and has no period.
- Occultation is a per-photon rejection drawn from the *unocculted* mean.
- Occultation must **not** be applied to `EarthAlbedoSource` (trap 1).

## Open questions for the maintainer

Carried forward until answered; they shape later PRs.

1. **Periapsis epoch** (PR 2). The plan gives `M = n(t - t_periapsis)` but never defines
   `t_periapsis`. Implemented as `0` on the absolute clock, so periapsis passage is
   always at global `t = 0` regardless of `initial_time`. Confirm, especially for
   `initial_time != 0`.
2. **`time_step` is a target, not exact** (PR 2). Rounded to an integer number of equal
   intervals tiling `duration`, rather than fixed steps with a short final one.
3. **`SpacecraftHistory.open()` gained an `earth = None` kwarg** (PR 2), not in the
   plan's literal signature. Forced by the plan's own validation rule
   `orbit_radius > earth.radius`. Either the reader knows about the Earth, or that
   validation moves elsewhere.
4. **`plot_spectrum()` will break for near-field sources** (blocks PR 4). `flux` lives
   on the base `Source` and `NearFieldSource.flux` is hardcoded `None`, but
   `plot_spectrum()` raises when `flux is None`. PR 4's `NearPointSource` therefore
   cannot plot its spectrum. Needs a `normalization` property resolving to flux or
   rate, with an adapting axis label. The plan (§5.1) intends these helpers to serve
   all sources.
5. **Blanket docstring constraint** (constraint 7). PR 1 left pre-existing untouched
   `Simulator` methods undocumented. `run_events`/`run_binned` are the main public
   teaching surface and arguably should be covered; the trivial axis accessors
   arguably not.

## Known environment traps for agents

- The system Python lacks pandas, and `pip install -e .` fails against Debian's
  managed `packaging`. Use
  `python -m venv --system-site-packages .venv && .venv/bin/pip install -e ".[test,lint]"`,
  created **inside** the agent's own worktree so the editable install points at its copy.
- Add `.venv/` to the worktree's `.git/info/exclude`.
- A branch already checked out in another worktree cannot be checked out again. Work on
  a local branch and push to the target name.
- Test budget: whole suite under 120 s. Currently ~24 s.
