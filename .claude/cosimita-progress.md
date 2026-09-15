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

PR 3 ran the implementer and test author **concurrently**, on separate branches, both
working from one written API contract, so their independence was auditable. It paid off
(303/307 passed on reconciliation, and the 4 failures were all test-fixture bugs).

**PR 4 reverts to the simpler sequential flow** -- implementer, then test author, then
reviewer -- agreed with the maintainer on the grounds that PR 4's two sources are far
more mechanical than PR 3's frame algebra. The test author still works from the written
contract and from independently derived numbers, never from the implementation, and a
disagreement is still a finding to report rather than something to resolve by weakening
the test.

The orchestrator independently re-verifies every claim before opening a PR, including
mutation-testing the new tests by injecting deliberate bugs.

## Status

**The plan is complete.** All seven PRs are merged as of 2026-09-15 (PR #21, merge
commit `9a4f20a`). Nothing in the cosimita plan is outstanding. What remains below is
kept as the record of how it was built and what was learned; a fresh session picking
this up should read the "Lessons" sections before changing the simulator or its
configuration schema, and the "Decisions already taken" section before re-litigating
anything there.

| PR | Scope | Branch | State |
|---|---|---|---|
| 1 | Source hierarchy + `simulated_rate()` | merged | **Merged** (PR #13) |
| 2 | `Earth`, `SpacecraftHistory`, orbits | merged | **Merged** (PR #14) |
| 3 | `InertialSimulator`, transforms, occultation | merged | **Merged** (PR #15) |
| 4 | `NearPointSource`, `ExtendedSource` | merged | **Merged** (PR #17) |
| 5 | `EarthAlbedoSource` | merged | **Merged** (PR #19) |
| 6 | Time-dependent scaling + event CSV I/O | merged | **Merged** (PR #20) |
| 7 | YAML configuration | merged | **Merged** (PR #21) — three review rounds: expression evaluator removed entirely, `*_from_config` moved onto the classes, Python floor raised to 3.12 with `astropy>=8.0` pinned. 570 tests green. |

Side PRs, outside the seven:

| PR | Scope | State |
|---|---|---|
| #16 | `SimpleTraditionalReconstructor` trigger guard (`and np.any(hits.layer > 0)`) | **Merged** |
| #18 | `PointSource` chirality coverage (tests only) | **Merged** |

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
- **`transform.py` lives in `gammaraytoys/coordinates/`, not `sims/`** -- it is frame
  algebra, not simulation. The `sims` re-export was dropped, so import it from
  `gammaraytoys.coordinates.transform`.
- **`SpacecraftHistory.intervals(tstart, tstop)` owns the window clipping** and the
  livetime rescale `live * (hi-lo)/(stop-start)`, moved out of the simulator. PRs 4-7
  get clipped intervals for free and must not re-implement it.
- **CI executes `docs/examples/cosimita/*.ipynb`** in a `notebooks` job after `test`.
  Every future PR's notebooks must run clean from a fresh kernel. It is a smoke check,
  to be replaced by lightweight unit tests eventually.
- **`progress = True` on `run_events` and `run_binned`** for both simulators, forwarded
  to tqdm's `disable`. **Every cosimita notebook must pass `progress = False`**: tqdm
  writes one stream record per refresh and nbconvert stores all of them, which was 149
  frames each in notebooks 00 and 02. Do *not* reach for `TQDM_DISABLE` instead -- tqdm
  binds it at import time, so it only works from a cell running before the first
  `gammaraytoys` import and breaks silently if cells are reordered.
- **Re-execute notebooks with the inline backend, never `MPLBACKEND=Agg`.** Agg
  overrides the inline backend, so `--execute --inplace` silently strips every figure
  from the committed `.ipynb` while exiting 0. The CI job sets Agg deliberately, which
  is fine there because it discards output via `--stdout`.

## Notebooks stay in CI

The maintainer measures the whole `notebooks` job at ~120 s on their machine, about
half what this container reports, and has decided the numbered notebooks stay in CI.
Do not propose moving them to `slow/` on wall-time grounds.

## Long-running notebooks live outside CI

`docs/examples/cosimita/slow/` holds notebooks that run real simulations to a few
hundred triggers per panel -- minutes each. The `notebooks` CI job globs
`docs/examples/cosimita/*.ipynb`, which does **not** recurse, so they are skipped by
construction; a README there and a comment on the job say so explicitly. Keep the
numbered notebooks fast enough to gate every push and put anything slower in `slow/`.
**Never re-execute a notebook with `MPLBACKEND=Agg`** -- it overrides the inline
backend and silently strips every figure while still exiting 0.

## Landed in PR 4, beyond the plan

- **`NearFieldSource.plot` now expands the axes limits.** PR 1 wrote it without the
  `_expand_axes_limits` call every far-field plot method makes, so a near source
  beyond `1.5 a` was drawn off-screen and vanished silently. PR 4 is what made the
  `s >= a` branch reachable, so it is fixed there.
- **`ExtendedSource` validates `width`.** `kappa = 1/width**2` above ~1e15 sends
  `scipy.stats.vonmises.rvs` into a rejection loop that never accepts, inside compiled
  code, so the process hangs and cannot be interrupted. `width = 0` reaches it, and the
  docstring's "at very small width this behaves like a PointSource" invites exactly
  that. PRs 5-7 adding any other sampled distribution should validate the same way.
- **Both new sources emit `direction` in degrees.** `Particle` preserves whatever unit
  it is handed, so a radian direction is visible downstream in `EventList.write`.

## Known issues — all three now resolved

- **`run_binned` dropped ~8.5% of an inertial run.** It filled `Nu = 270 deg - direction`
  while `direction` is wrapped to `[0,360)`, giving `Nu` in `(-90, 270]` against an
  axis of `[-180, 180]`; everything with `direction < 90 deg` fell off and histpy
  dropped it with a warning. **Fixed in PR 3** with `Angle(...).wrap_at(180 deg)`, at
  the maintainer's direction. Pre-existing and byte-identical on `main`; no cached
  response was affected, because every `photon_axes` use that feeds an `.h5` passes
  `'Ei'`, never `'Nu'`.
- **`SimpleTraditionalReconstructor` returned `triggered = True` with `psi = nan`**
  when every hit is in layer 0. Measured at 0.19% of triggers in the uniform-stack
  fixture and 0% in COMPTELito. **Verified mechanism**: the photon Comptons in layer 0
  (recorded), deposits below `energy_threshold` in a nearby layer (so no hit is
  recorded), backscatters, and interacts in layer 0 again -- giving recorded hits
  `[0, 0]` with an invisible step between them. Sub-threshold deposits of 6.9-10.0 keV
  against a 20 keV threshold were seen in all three sampled events. **Fixed in PR #16**
  (merged) by requiring `np.any(hits.layer > 0)`. The cached responses use COMPTELito,
  where the rate is zero, so they were **not** regenerated; a note in
  `docs/tutorials/compton_telescopes/data/README.md` records this, to be removed if
  they are ever regenerated.
- **A stale comment** in `tests/test_inertial_simulator.py`'s module docstring
  (`rho = 70.2513 deg`, true value `70.2074 deg`). **Fixed in PR 3**, along with six
  further constants arithmetically descended from it.

## Carried into PR 3

- **`is_occulted` is dominated by Quantity arithmetic**, done per photon, and it
  recomputes `arcsin(R_E/r)` every call although it is constant per interval. That is
  exactly the pattern SS3.5 warns about. **Done in PR 3**: a private `_is_occulted`
  fast path taking plain floats, unit-tested against the public method so the two
  cannot drift. Measured at **1.8 us against 23.9 us**, roughly **13x**.
  *Earlier figures in this file and in PR 3's first description (277x, 26x) were
  wrong* -- they compared the old public path against a hand-written float snippet
  rather than against the method as shipped. Repeat measurements of the real pair
  land in the 11-13x range; absolute numbers move with machine load.
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
- **A green suite says nothing about the notebooks.** pytest never imports one, so
  moving `transform.py` broke notebook 02's very first import with all 313 tests
  passing, and notebook 00 sat broken on the PR 3 branch (`DemoNearFieldSource.
  random_photon` never grew the `earth` parameter) through several rounds of review.
  Hence the CI job. Until a PR's notebooks have actually been executed, they are
  unverified.
- **Verify against the code under review, not whatever is installed.** A scratch venv
  whose editable install still pointed at the `main` checkout made a notebook run look
  like evidence for a branch it never touched. Confirm `package.__file__` resolves
  inside the worktree before believing any run.
- **Reclaim agent worktrees between PRs.** Each carries an ~800 MB-1 GB `.venv`; by the
  end of PR 3 there were 29 stale ones holding 22 GB, and PR 4's implementer hit a full
  disk (135 MB free) before it could install anything. Removing every worktree whose
  HEAD is an ancestor of `main` freed 20 GB and touched no branch ref:
  `for wt in .claude/worktrees/*/; do h=$(git -C "$wt" rev-parse HEAD); \
   git merge-base --is-ancestor "$h" main && git worktree remove --force "$wt"; done`
  Do this at the start of each PR. Check for unmerged worktrees first -- `pr2-fixes`
  is deliberately kept.
- **A one-sided assertion can be *reinforced* by the bug it should catch.** PR 4's
  "a near source at the centre triggers far more often than one far outside" only
  required the far source to trigger rarely -- so breaking its aim entirely, which
  makes it trigger even less, strengthened the assertion instead of failing it. Ask of
  every comparison test: does the bug I fear push this quantity toward the assertion or
  away from it?
- **A test helper must not invert the transform under test.** PR 4's sky-angle helper
  recovered `lambda` with `offaxis_to_sky_angle(..., pose.attitude)`, the exact inverse
  of the `sky_angle_to_offaxis(..., pose.attitude)` the code had just applied. Both are
  `A - x`, so the round trip cancelled any attitude error and the source could ignore
  attitude entirely with the suite green. Derive the expectation from the raw output
  (`photon.direction`), not by undoing the computation.
- **Isolating one effect can delete the effect you meant to test.** Every PR 4
  `ExtendedSource` test used a 1 m Earth so nothing was ever occulted -- which made
  occultation deletable outright with the suite green. When a fixture switches
  something off to isolate something else, check that the thing switched off is still
  tested somewhere.
- **A source that caches per-pose geometry needs a test that reuses ONE source across
  poses.** PR 5's albedo caches its geometry on the orbital radius. Dropping
  `orbit_radius` from the cache key -- so it freezes at whatever radius it first saw --
  passed all 367 tests while inflating an elliptical-orbit expected count by 53-57%.
  Every test that varied altitude built a *fresh* source inside the loop. Interleave
  the poses (lo, hi, lo, hi): a cache that updates but only ever forward survives a
  monotone sweep.
- **A source carrying its own `Earth` must be validated against the simulator's at
  construction.** `EarthAlbedoSource` re-checked per photon, which is too late: a run
  whose Poisson draw comes up empty finishes silently having computed every expected
  count from the wrong planet. The default `Earth()` is astropy's 6378.1 km while this
  project uses 6371 km, so the plain `EarthAlbedoSource(E, spectrum)` form is the
  mismatching one. Now checked in `InertialSimulator._validate_sources`.
- **Fix a wrong base-class docstring in the base class, not with a subclass override.**
  PR 5 documented `EarthAlbedoSource`'s pose requirement by overriding
  `simulated_rate` purely to carry a corrected docstring. The maintainer asked for the
  inherited one to be fixed instead, and was right for a reason worth remembering: the
  base docstring was not merely incomplete for the subclass, it was actively false --
  and it carried a separate stale claim ("every source in this package has a
  pose-independent rate") that an override would have left sitting there for every
  future reader.
- **`_expand_axes_limits` leaves 8% headroom** (`_PLOT_AXES_MARGIN`). It used to expand
  to exactly the radius requested, so a sky-circle star sat on the boundary half
  clipped and an arc drawn at that radius traced the frame itself.
- **A `true_*` column must mean *this event*, never a property of its source.** PR 6
  wrote `ExtendedSource`'s distribution centre into `true_sky_angle_deg`: a constant
  40 deg while the real per-photon angles spanned 181 deg, mean error 21 deg. A NaN
  would have been honest; a plausible constant was not. If per-event truth is not
  recoverable, write nothing rather than something that looks right.
- **A test fixture that removes an ambiguity often removes the test.** PR 6's event
  helper timestamped every photon at `interval.mid_time` so the drawing and lookup
  attitudes were unambiguously the same interval -- which is exactly why the attitude
  search was never exercised, and why three mutations in it passed the whole suite,
  the worst corrupting 43% of rows by up to 54 deg. Fixtures must reproduce what the
  code under test really produces; `run_events` draws timestamps uniform over the span.
- **Give abstract bases a class-level default for any new attribute.** PR 6's `scaling`
  was set only in each concrete `__init__`, so a third-party subclass -- including the
  demo source in the shipped notebook 00 -- crashed inside `run_events` with
  `AttributeError`. One class-level default keeps a new feature backward compatible.
- **The plan's expression-evaluator defence is insufficient, and PR 7's contract says
  so with evidence.** Whitelist-plus-reject-`__` blocks every attack the plan names,
  but three expressions carry no `__` at all: `9**9**9**9` hangs the process
  (measured: killed at 6 s), and `(lambda: 1)()` and `t.real.conjugate()` are simply
  allowed. The last matters most -- attribute access is permitted, and today's
  whitelist exposing nothing useful through it is luck, not a property. Use an AST
  whitelist rejecting `ast.Attribute` and `ast.Lambda`. For the arithmetic DoS,
  reject chained `**` and integer *literal* exponents above 64; a naive "exponent
  must be a literal <= 64" rule is too strict and rejects `2**t`.
- **Compare a new source's speed against the sibling that does the same work.** PR 4's
  implementer measured `ExtendedSource` at ~625 us against `PointSource`'s ~200 us and
  believed it had blown constraint 2's 50% budget. But a `PointSource` with a fixed
  `offaxis_angle` never re-aims, so its throwing-plane cache never misses. Against
  `IsotropicSource`, which re-aims every photon exactly as `ExtendedSource` does, and
  with occultation controlled to ~0% on both so neither gets cheap early-outs, it is
  585 us vs 540 us -- **+8.3%**. Control the occultation rate before comparing: an
  occulted draw returns before the expensive throwing-plane call and silently
  cheapens whichever source is more occulted at the chosen pose.
- **Agents must commit and push as soon as work is done**, not after polishing: a rate
  limit killed one agent mid-task and lost the entire round.
- **`git push` can fail transiently with `could not read Username`** while `git fetch`
  keeps working -- the proxy's injected write credential drops and comes back. Six
  consecutive attempts failed and the seventh succeeded with nothing changed. Commits
  are not lost when this happens; re-push rather than redoing the work.

## Lessons from PR 7's review round

- **A mutation battery cannot find a rule that is the wrong shape.** PR 7's expression
  evaluator guarded `**` with two rules on the *exponent*. Every mutation of those rules
  was caught; the tests were fine. But both rules inspect only the **right** operand, and
  all the growth in a left-nested tower is on the left: `(((2**64)**64)**64)**64` built a
  16,777,217-bit integer, and each further nesting *squared* that (5.7 s / 703 MB one
  deeper, 61 s / 3.7 GB / `MemoryError` two deeper). Mutation testing asks "are these
  rules pinned?", never "are these the right rules?". Only an adversarial reviewer whose
  brief says *try to defeat it* found it.
- **The attack was invisible.** `1 + 0*(((2**64)**64)**64)**64` returns exactly `1.0`, so
  the config loads in 9 ms, the scaling looks like a no-op, and the run silently takes
  577x longer. A test asserting the scaling's *value* would have passed.
- **Prefer removing the possibility to out-guessing the shape.** Patching "no `**` on the
  left either" was itself beaten in ten minutes by `((((2**64*1)**64*1)**64*1)**64*1)**64`.
  The fix that worked deleted both rules and made `**` float-only via an
  `ast.NodeTransformer` before `compile` — floats overflow instead of growing, so there is
  no tree shape left to guess. It also *stopped* a wrong refusal: `0.999**1000` was being
  rejected, and a survival fraction raised to a step count is ordinary physics.
- **A daemon thread cannot bound a big `**`.** A single arbitrary-precision power is one
  uninterruptible C call that never releases the GIL, so `join(timeout)` does not return
  until it finishes: measured, `join(0.3)` on `7**40000000` returned after **45.7 s**. A
  thread guard around `9**9**9**9` would hang CI, which is the exact failure it was meant
  to prevent. Bound it in a **child process** with a timeout and an `RLIMIT_AS`.
- **Cap the expression length, and measure the boundary before choosing the number.** The
  obvious ~1000 characters would have been decorative: the recursion limit through the two
  nested tree walks is **497 characters**. 250 closes both the recursion half and the
  integer-multiply half.
- **Round-trip tests are structurally blind to a field both writes omit.** `to_config`
  dropping a source's `scaling` passed every round-trip test — compare two writes to each
  other and a missing key is invisible. A source could lose its `FunctionScaling` and still
  "round-trip cleanly", then run at a constant rate. Assert named fields, not just equality.
- **A silent load that dies far away is the worst failure mode, and this schema had four.**
  A negative `flux` loaded fine, round-tripped verbatim, then died inside numpy with
  `lam < 0 or lam is NaN` naming nothing. `flux` beside `flux_pivot`, and either half of the
  pivot pair alone, were silently resolved. A `pivot_energy` outside the spectrum's range
  gave `inf`, and `to_config` wrote `flux: 'inf 1 / (cm s)'` back into the file, where it
  looks deliberate. Validate at the point the value is read, as `scaling.py` already did.
- **A "did you mean" can be worse than silence.** Adding `difflib` to the function-name
  error made `min(1, t)` suggest `'sin'` (edit distance 0.667) over `'minimum'` (0.60), and
  `cosine(t)` suggest `'sin'` over `'cos'`. Prefer prefix relationships for identifiers.
- **Notebooks are code that goes stale.** The capstone demonstrated the two deleted `**`
  rules, with their messages baked into its stored output; re-run, it printed
  `NOT rejected (!)`. Any PR that changes behaviour a notebook teaches has to re-run it.
- **Brief the reviewer to attack, not to read.** The three findings that mattered all came
  from the instruction "decide whether you can defeat it — try", with measurements demanded.

## Lessons from PR 7's maintainer review

- **The cheapest way to secure an input is to stop accepting it.** PR 7 spent an
  entire review round hardening a free-form expression evaluator, and the defence
  was beaten twice anyway. The maintainer's answer was to delete the feature: fixed
  scalings with named parameters (`Burst`, `Sinusoidal`) instead of a formula. That
  removed 508 lines of `config.py`, an 891-line test file, and the whole attack
  surface. When a review finds a hole in a defence, "is this input worth accepting
  at all?" comes before "how do we defend it better?".
- **Say what a trade cost, in the docs, rather than quietly dropping the feature.**
  Notebook 06's section on the evaluator became "Why there is no expression syntax",
  naming both sides: the defence that failed, and the price of fixed shapes (a new
  shape needs a subclass, not a line in the config).
- **A green suite does not mean a rename was safe.** A `\bword\b` rename during the
  classmethod refactor damaged four *strings* rather than identifiers, and no test
  asserted any of them, so it survived three commits green. Caught only by
  AST-diffing every moved function body against its original. For a pure move,
  diff the bodies -- do not trust the tests to notice.
- **Verify a "pure refactor" as one.** Beyond the suite, 25 paired cases (18 error
  paths, 7 round trips) were run through the old and new APIs and compared including
  exception type and message string. That is what makes "behaviour is unchanged" a
  claim rather than a hope.
- **Breaking an import cycle beats deferring it.** Moving `*_from_config` onto the
  classes created a cycle (`config.py` imports the classes; the classes want its
  validators). Extracting the primitives into a dependency-free `config_utils.py`
  removed the cycle instead of hiding it behind function-body imports.
- **Guard inherited dispatch classmethods.** `Source.from_config` dispatches on
  `type`, so `PointSource.from_config` would silently return an `IsotropicSource`.
  The base checks `cls` and refuses a mismatch.
- **A half-open window is a decision, not a detail.** `BurstScaling` uses
  `start <= t < start + duration` specifically to agree with `TabulatedScaling`'s
  right-continuity. Two scalings disagreeing about their edges would be a trap.
- **Validate a shape that can go negative at construction.** `SinusoidalScaling`
  refuses `amplitude > mean`, because otherwise it is a negative Poisson mean deep
  inside `run_events` -- the same failure the review found for a negative `flux`.
- **CI's version matrix catches what one venv cannot.** The schema's array-with-unit
  string silently required a newer astropy than `pyproject.toml` asks for: 39 tests
  failed on Python 3.10 only. When a project supports a version range with unpinned
  deps, a local green suite proves one point in that space. First fixed by parsing
  the bracket by hand; **superseded in the third review round** -- see below.
- **A language-version floor does not imply a library floor.** The maintainer asked
  for Python >= 3.12 so the hand-rolled bracket parser could go. Measured, that alone
  was not enough: `u.Quantity("[0, 5, 10] mm")` fails on astropy 6.1.7 and every 7.x,
  and those install happily on 3.12 -- the same 39 failures, on a newer interpreter.
  8.0.0 is the first release that parses it, so `astropy>=8.0` is the real
  requirement and is what makes deleting the workaround sound. When a request is
  phrased as "raise the version so X can go", check which dependency actually
  gates X before deleting anything.
- **Keep more than one version in the matrix.** After the bump the matrix is
  `["3.12", "3.13"]`, not a single pin: a matrix spanning a range is the thing that
  caught this bug in the first place, and it costs one runner to keep that property.
- **Diff the old and new implementations before deleting a parser.** 36 inputs through
  both: every valid form identical (`"[5] mm"` still a one-element array), all 12
  differences confined to malformed or exotic input. Two were worth reporting rather
  than burying -- `"[1 2] mm"` is now accepted (a genuine loosening), and `"[inf] mm"`
  now parses and is caught one step later by the finiteness check instead.
- **Stale `.pyc` hides `SyntaxWarning`.** A cached-bytecode run reported "no warnings";
  cold, the suite shows 5 pre-existing `invalid escape sequence` warnings
  (`materials/material.py:68,72`, `tracker_2d.py:317`, `fast_norm_fit.py:8,145`).
  They are compile-time, so only a cold run sees them. Clear `__pycache__` before
  quoting a warning count.
- **A stale PR description misleads the review.** #21's body twice said a thing was
  "not fixed" that had been fixed at the maintainer's request, and omitted the
  portability fix entirely. Rewrite the body when the branch moves, not just when
  the PR opens.

## Known environment traps for agents

- The system Python lacks pandas, and `pip install -e .` fails against Debian's
  managed `packaging`. Use
  `python -m venv --system-site-packages .venv && .venv/bin/pip install -e ".[test,lint]"`,
  created **inside** the agent's own worktree so the editable install points at its copy.
- Add `.venv/` to the worktree's `.git/info/exclude`.
- A branch already checked out in another worktree cannot be checked out again. Work on
  a local branch and push to the target name.
- Test budget: whole suite under 120 s. Currently ~24 s.
