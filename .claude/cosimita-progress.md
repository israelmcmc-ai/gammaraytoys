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
| 1 | Source hierarchy + `simulated_rate()` | `claude/cosimita-pr1-source-hierarchy` | **PR #13 open** — awaiting maintainer |
| 2 | `Earth`, `SpacecraftHistory`, orbits | `claude/cosimita-pr2-spacecraft-history` | **PR #14 open** — awaiting maintainer |
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

## Queued for the PR 1 follow-up pass

Decided by the maintainer; to be applied together with the PR 1 reviewer's findings
in a single follow-up commit on `claude/cosimita-pr1-source-hierarchy`.

- **`plot_spectrum()` must serve both source families.** Use the flux for a
  `FarFieldSource` and the rate for a `NearFieldSource`, and adjust the y-axis units
  to match: `1/(erg cm s)` and `erg/(cm s)` for far field, `1/(erg s)`
  and `erg/s` for near field. `diff_flux`, `integrate_flux` and
  `discretize_spectrum` feed `plot_spectrum` and must stay consistent with it.
  This requires `NearFieldSource` to expose a `rate` property, which PR 1 did not
  add; PR 4's `NearPointSource` will implement it. Without this, `NearPointSource`
  cannot plot its spectrum at all.

  **Implemented polymorphically**, via an abstract `Source.normalization` that
  `FarFieldSource` resolves to `flux` and `NearFieldSource` to `rate`, rather than
  by `isinstance` branching inside `plot_spectrum`. Same dispatch-by-base-class
  behaviour, and it additionally removes a latent bug: `diff_flux` and
  `integrate_flux` reached into a private `self._flux` that the base class never
  defines, which breaks any subclass that does not happen to set it. Far-field
  y-units are unchanged; near-field gets `1/(erg s)` and `erg/s`.

- **`Simulator.__init__` no longer accepts `duration`/`nsim`/`ntrig`.** They were
  silently discarded (`Simulator(..., duration=1000*u.s).duration` returned `0 s`)
  while PR 1's new docstring claimed they worked. Nothing in the repo, tests or
  tutorials passed them — they are only ever given to `run_events`/`run_binned`,
  which do consume them. This is a deliberate API removal, visible in the diff.
- **Docstrings on `Simulator.run_events` and `run_binned`.** They are the main
  public teaching surface. The trivial axis property accessors stay as they are.

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

## Carried into PR 3

- **`is_occulted` costs 140 us/call** vs 0.5 us for the plain-float equivalent (277x),
  because it does Quantity arithmetic per photon and recomputes `arcsin(R_E/r)` every
  call although it is constant per interval. That is 22% of `random_photon` -- inside
  the plan's 50% budget, but exactly the pattern SS3.5 warns about. Hoist `nadir` and
  `rho` to floats once per interval and unit-test the fast path against `is_occulted`
  so the two cannot drift.
- **`attitude` and `orbit_angle` are unwrapped past 360 deg** (matching the plan's own
  SS4.1 example). PR 3's `Nu = A - lambda` owns the wrapping; PR 2 does none.
- **`FarFieldSource` has no `occultable` property yet** (trap 1 requires it, `False` on
  the albedo). Deliberately left out of PR 1; PR 3 or PR 5 adds it.
- **`PointSource.__init__` takes `offaxis_angle` as a required first positional.** PR 3
  must add `sky_angle` and make them mutually exclusive. Every call site in `docs/` and
  `tests/` uses the keyword form, so positional compatibility is not load-bearing.
- **SS5.3 commits occultation to the *source*** (`random_photon` returns `Photon | None`),
  not to the simulator recovering lambda from `photon.direction`. PR 3 must not drift to
  the latter.
- **`simulated_rate()` returns `None` for an unnormalized source**, which would surface
  in PR 3 as a `TypeError` deep inside `mu = rate * livetime * scaling`.
  `InertialSimulator.__init__` should validate and raise naming the offender.
- **`Earth` is not carried on `SpacecraftHistory`'s public API for PR 3's use** beyond
  the new read-only property -- PR 3 and PR 7's YAML loader must use that property
  rather than constructing their own, or the validated and simulated Earths can differ.
- **The per-photon body of `Simulator.run_events` is inlined**, so `InertialSimulator`
  will duplicate `random_photon -> simulate_event -> reconstruct`. Extracting a shared
  `_simulate_one(source, pose)` helper would avoid that.

## Lessons for later PRs

- **Mutation-test the tests, not just the code.** Both the first test round and the
  orchestrator's own mutation check passed PR 1 while the two behaviours it actually
  changed -- rate-weighted source selection and the `duration` accumulator -- could be
  deleted outright with the whole suite still green. Passing a mutation check only
  proves the suite catches the mutations you happened to pick. Test authors must inject
  the specific bug their test targets and show it fails.
- **The reviewer slot earns its cost.** Review found real defects in both wave-1 PRs
  that implementation, testing and orchestrator verification had all missed.
- **Agents must commit and push as soon as work is done**, not after polishing: a rate
  limit killed one agent mid-task and lost the entire round.
- **`git push` can fail transiently with `could not read Username`** while `git fetch`
  keeps working -- the proxy's injected write credential drops and comes back. Six
  consecutive attempts failed and the seventh succeeded with nothing changed. Commits
  are not lost when this happens; re-push rather than redoing the work.

## Known environment traps for agents

- The system Python lacks pandas, and `pip install -e .` fails against Debian's
  managed `packaging`. Use
  `python -m venv --system-site-packages .venv && .venv/bin/pip install -e ".[test,lint]"`,
  created **inside** the agent's own worktree so the editable install points at its copy.
- Add `.venv/` to the worktree's `.git/info/exclude`.
- A branch already checked out in another worktree cannot be checked out again. Work on
  a local branch and push to the target name.
- Test budget: whole suite under 120 s. Currently ~24 s.
