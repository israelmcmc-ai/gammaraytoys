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
from one written API contract given to both, on **separate branches**
(`...-pr3-inertial-simulator` and `...-pr3-tests`) so their independence is auditable
and there is no push race. The test author reconciles at the end by merging the
implementer's branch and running; a disagreement is a finding to report, never
something to resolve by weakening the test.

The orchestrator independently re-verifies every claim before opening a PR, including
mutation-testing the new tests by injecting deliberate bugs.

## Status

| PR | Scope | Branch | State |
|---|---|---|---|
| 1 | Source hierarchy + `simulated_rate()` | merged | **Merged** (PR #13) |
| 2 | `Earth`, `SpacecraftHistory`, orbits | merged | **Merged** (PR #14) |
| 3 | `InertialSimulator`, transforms, occultation | `claude/cosimita-pr3-inertial-simulator` | Implemented, tested, reviewed, fixed — 312 tests; notebook in progress |
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

## Landed in PR 1 beyond the original plan

- **`flux` moved off the base `Source`** and `sky_integrated_flux(pose)` became
  `FarFieldSource.flux(pose)`, a method. `NearFieldSource.flux` is gone entirely.
- **`Simulator.total_flux` removed**; `total_rate` renamed `total_simulated_rate`.
- **`chirality_degree` defaults to 0** everywhere.
- **Source plotting**: `Source.plot(ax, detector)`, polymorphic. Near-field sources
  draw a red star at their `position` (a new abstract property on `NearFieldSource`);
  far-field point sources draw a sky circle at `2 x surrounding_circle_radius` with a
  star just outside it at `1.08 x` that radius, along `(sin Nu, cos Nu)`;
  `IsotropicSource` draws a full-circle arc. **PRs 4 and 5 add `ExtendedSource` and
  `EarthAlbedoSource` by calling the existing `plot_sky_arc` with their own extent** --
  the primitive is already there, do not rebuild it.
- The cosimita notebooks use **COMPTELito**, not cosita: `SimpleTraditionalReconstructor`
  needs the first hit in layer 0 with a calorimeter below, which cosita's layer ordering
  does not provide. Measured on cosita: 0.38% trigger, psi std 142.9 deg (no cone).
  On COMPTELito: 10.35% trigger, psi std 30.9 deg, properly centred.

## Traps for anyone editing sources

- **An abstract `@property` cannot be satisfied by `self.position = value` in
  `__init__`** -- a property with no setter is a data descriptor and blocks the
  assignment. Back it with `self._position` and override the property, the pattern
  already used for `PointSource.spectrum`.
- **`ToyTracker2D.plot()` hardcodes centimetres** for its data coordinates and sets
  axes limits to +-1.5x the surrounding radius. Anything drawn on top must use cm and
  must expand those limits, or it lands in the wrong place or off-screen -- silently,
  in both cases.

## Landed in PR 3, beyond the plan

- **`random_photon(detector, pose = None, earth = None)`** — see the plan's §5.3 note.
  PRs 4-7 must write their sources against this three-parameter signature.
- **`SimulatorBase`** — shared base for `Simulator` and `InertialSimulator`, holding
  `_simulate_one`, the six CDS axis properties and `_run_binned`. Verified
  bit-identical output for `Simulator` against `main` by seeded comparison.
- **`PointSource` takes either `offaxis_angle` or `sky_angle`**, mutually exclusive.
  A `sky_angle` source **mutates `self.offaxis_angle` on every draw** — do not share
  one instance between two runs, and PR 7's config loader must not hand the same
  source object to two simulators.
- **`InertialSimulator` validates the Earth at construction.** A mismatched Earth
  used to silently disable occultation entirely (1137 occulted -> 0), because
  `_is_occulted` does no validation and `arcsin(R/r)` goes `nan`.

## Known issues, each awaiting its own PR

- **`run_binned` drops ~8.5% of an inertial run.** It fills `Nu = 270 deg - direction`
  while `direction` is wrapped to `[0,360)`, giving `Nu` in `(-90, 270]` against an
  axis of `[-180, 180]`; everything with `direction < 90 deg` falls off and histpy
  drops it with a warning. Pre-existing and byte-identical on `main`, but the inertial
  simulator sweeps the whole sky, so it is now routine. Fixing it changes `Simulator`
  output, hence its own PR.
- **`SimpleTraditionalReconstructor` returns `triggered = True` with `psi = nan`**
  when every hit is in layer 0. Measured at 0.19% of triggers in the uniform-stack
  fixture and 0% in COMPTELito. **Verified mechanism**: the photon Comptons in layer 0
  (recorded), deposits below `energy_threshold` in a nearby layer (so no hit is
  recorded), backscatters, and interacts in layer 0 again -- giving recorded hits
  `[0, 0]` with an invisible step between them. Sub-threshold deposits of 6.9-10.0 keV
  against a 20 keV threshold were seen in all three sampled events. The cached
  responses use the COMPTELito geometry, where the rate is zero, so a fix does **not**
  require regenerating them.
- **A stale comment**: `tests/test_inertial_simulator.py`'s module docstring says
  `rho = arcsin(6371/6771) = 70.2513 deg`; the true value is `70.2074 deg`.

## Carried into PR 3

- **`is_occulted` costs 140 us/call** vs 0.5 us for the plain-float equivalent (277x),
  because it does Quantity arithmetic per photon and recomputes `arcsin(R_E/r)` every
  call although it is constant per interval. That is 22% of `random_photon` -- inside
  the plan's 50% budget, but exactly the pattern SS3.5 warns about. Hoist `nadir` and
  `rho` to floats once per interval and unit-test the fast path against `is_occulted`
  so the two cannot drift.
- **`attitude` and `orbit_angle` are unwrapped past 360 deg** (matching the plan's own
  SS4.1 example). PR 3's `Nu = A - lambda` owns the wrapping; PR 2 does none.
- **`occultable`** is being added in PR 3, default `True`; PR 5's `EarthAlbedoSource`
  overrides it `False` (trap 1).
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
