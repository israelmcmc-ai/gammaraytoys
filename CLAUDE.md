# gammaraytoys

Teaching toys for gamma-ray analysis. Students read this code as much as they run
it, so **clarity beats cleverness and speed everywhere in this repo**.

## What the package holds

| Module | Contents |
|---|---|
| `physics/` | Compton kinematics |
| `materials/` | `Material` (cross sections, attenuation) |
| `detectors/tracker/` | `ToyTracker2D` -- the layered Compton tracker |
| `detectors/coded_mask/` | 2D and 3D coded-mask toys |
| `coordinates/` | `Cartesian2D`, frame transforms (`transform.py` lives **here**, not in `sims/`) |
| `sims/` | Sources, spectra, `Earth`, `SpacecraftHistory`, simulators, YAML config, event CSV, reconstructors |
| `analysis/` | Likelihoods, response, Richardson-Lucy, likelihood grids, fast norm fit |

Docs: `docs/tutorials/compton_telescopes/` (the student-facing course),
`docs/examples/cosimita/` (simulation examples; `slow/` holds long-running ones CI
skips), `docs/dev/` (design plans).

## The two toy detectors

Both are **`ToyTracker2D` instances configured in the tutorials**, not classes in the
package. Don't go looking for a `COSIta` class, and don't invent one without asking.

- **COSIta** -- COSI-like. Germanium, `layer_length = 16 cm`, twelve layers in four
  groups of three (`[0,5,10, 20,25,30, 40,45,50, 60,65,70] mm`),
  `layer_thickness = 5 mm`, `energy_resolution = 0.01`, `energy_threshold = 20 keV`.
- **COMPTELito** -- COMPTEL-like, the "traditional" two-plane geometry.
  `layer_length = 10 m`, one layer at 300 cm plus ten at 0..9 cm,
  `layer_thickness = 1 cm`, `energy_resolution = 0.03`.

"cosimita" is neither -- it is the name of the *simulation project* whose plan is
`docs/dev/inertial_sim_plan.md` (complete). Its coordination file was removed once the
project closed; `git log --diff-filter=D -- .claude/cosimita-progress.md` finds the
commit that deleted it if the evidence behind any rule here is ever wanted.

## Reconstructors

`sims/reco.py` holds the `Reconstructor` ABC (with `from_config`/`to_config` already
wired into the YAML schema, so a new reconstructor gets config support nearly free),
`SimpleTraditionalReconstructor`, and `RecoEvent`/`RecoCompton`.

**`SimpleTraditionalReconstructor` does not work on COSIta**, and this is measured,
not suspected. It needs the first hit in layer 0 with a calorimeter below it, which
COSIta's layer ordering does not provide:

| detector | trigger fraction | psi std |
|---|---|---|
| COSIta | 0.38% | 142.9 deg (no cone at all) |
| COMPTELito | 10.35% | 30.9 deg (properly centred) |

This is why every cosimita notebook uses COMPTELito, and it is the gap a COSIta
reconstructor exists to fill.

A related fixed bug worth knowing: the reconstructor used to return
`triggered = True` with `psi = nan` when every hit was in layer 0 (photon Comptons in
layer 0, deposits sub-threshold in a nearby layer so no hit is recorded, backscatters,
interacts in layer 0 again). Fixed in #16 by requiring `np.any(hits.layer > 0)`. The
rate is geometry-dependent: 0.19% of triggers on a uniform 5 mm-pitch stack, **0% on
COMPTELito**, whose 3 m standoff is far wider than the backscatter step. Because it is
zero for that geometry the cached responses were not regenerated for this fix; the note
in `docs/tutorials/compton_telescopes/data/README.md` says so and should be deleted
whenever they are next regenerated for any other reason.

**Cached responses: check before trusting.** `docs/tutorials/compton_telescopes/data/`
holds three `.h5` responses generated before an inverted-interaction-probability bug in
`ToyTracker2D.simulate_event` was fixed. Only `response_energy_onaxis_traddet.h5` has
been regenerated; `response_energy_relative_onaxis_traddet.h5` and
`response_imaging_chiral_relative_1MeV_traddet.h5` were still pending as of that
README's last update. Read it (and its git history) before building analysis on top of
them. `regenerate_responses.py` in that directory rebuilds any of them -- roughly 2 min,
25-30 min, and 1.5 h respectively.

## Physics conventions that bite

**This is a 2D "flatland" world.** Fluxes are per unit *length* per time
(`1/cm/s`), never per area. Earth albedo emissivity is `1/(cm s)`. Getting this
wrong produces numbers that look entirely plausible and are wrong forever.

**Frames.** `attitude` A is the inertial angle of the detector's +y axis;
`sky_angle` is λ; `orbit_angle` is θ. Then `Nu = A - λ` and
`direction_det = 270° - Nu`. Nadir is `orbit_angle + 180°`. `attitude` and
`orbit_angle` are left unwrapped past 360°; `Nu = A - λ` owns the wrapping.

**`phi` and `psi` are reserved for Compton Data Space.** Never reuse those names for
spacecraft geometry -- doing so has caused real confusion here. When plotting CDS,
it is `Histogram([psi_axis, phi_axis])`: Psi on x, Phi on y.

**The Earth albedo source is not occultable** (`occultable = False`) -- it *is* the
Earth. Occultation fraction is `2ρ/360°` with `ρ = arcsin(R_E/r)`; at 400 km that is
ρ = 70.2°, about 39% of the sky. Occultation is a per-photon rejection drawn from the
*unocculted* mean, and it belongs to the **source** (`random_photon` returns
`Photon | None`), not to the simulator recovering λ from `photon.direction`.

This project's Earth is **6371 km**; astropy's default `Earth()` is 6378.1 km. A
source carrying its own mismatched `Earth` silently computes every expected count
from the wrong planet, so `InertialSimulator` validates them at construction.

## Style

Tutorial-grade code: explicit, simple, unoptimized. **Up to a ~50% performance
penalty is an accepted trade for readability.** Don't vectorize a loop, add a cache,
or introduce a dependency to make something faster unless asked.

**`analysis/` has a stricter rule.** `likelihoods.py` (28 lines), `richardson_lucy.py`
(39), `response.py` (44) are deliberately short enough that a student can open them
and read the whole thing. New analysis capability must not make these longer or more
abstract -- build alongside them, don't refactor them into a framework.

## API facts worth knowing before editing `sims/`

- `random_photon(detector, pose = None, earth = None)` -- the three-parameter form.
- **A `PointSource` built with `sky_angle` mutates `self.offaxis_angle` on every
  draw.** Never share one instance between two runs or two simulators.
- `SpacecraftHistory.intervals(tstart, tstop)` owns window clipping and the livetime
  rescale `live * (hi-lo)/(stop-start)`. Don't re-implement it.
- Use `SpacecraftHistory`'s `earth` property rather than constructing your own, or the
  validated and simulated Earths can differ.
- `Source.normalization` is abstract and resolves to `flux` for `FarFieldSource`,
  `rate` for `NearFieldSource`. Dispatch through it rather than `isinstance`.
- **An abstract `@property` cannot be satisfied by `self.x = value` in `__init__`** --
  a property with no setter is a data descriptor and blocks the assignment. Back it
  with `self._x` and override the property.
- **Give an abstract base a class-level default for any new attribute.** A value set
  only in each concrete `__init__` breaks every third-party subclass, including the
  demo source in notebook 00.
- **Validate the parameters of any sampled distribution.** `ExtendedSource` with
  `width = 0` gives `kappa = 1/width**2` ~ 1e15, which sends
  `scipy.stats.vonmises.rvs` into a rejection loop inside compiled code that never
  accepts -- the process hangs and cannot be interrupted.
- `ToyTracker2D.plot()` hardcodes **centimetres** and sets axes limits to ±1.5x the
  surrounding radius. Anything drawn on top must use cm and must expand those limits,
  or it lands off-screen silently. `_expand_axes_limits` leaves 8% headroom.

## Testing

Expected values are derived **from first principles or back-of-envelope
calculation, never copied from running the implementation**. A test that records
what the code currently does proves nothing.

Mutation testing is the quality gate: **inject the specific bug your test targets and
show the test fails.** Passing a mutation check only proves the suite catches the
mutations you happened to pick -- and a battery cannot find a rule that is the *wrong
shape*, only one that is missing.

Failure modes that have actually slipped through here:

- **A one-sided assertion can be reinforced by the bug it should catch.** Ask of every
  comparison test: does the bug I fear push this quantity toward the assertion or away?
- **A test helper must not invert the transform under test.** Recovering λ with the
  exact inverse of what the code just applied cancels the error. Derive expectations
  from raw output (`photon.direction`), not by undoing the computation.
- **Isolating one effect can delete the effect.** A 1 m Earth so nothing occults makes
  occultation deletable with the suite green. When a fixture switches something off,
  check it is still tested somewhere.
- **A cache needs one object reused across varied inputs, interleaved** (lo, hi, lo,
  hi). Every test building a fresh object inside the loop misses a frozen cache
  entirely; a monotone sweep is survived by a cache that only updates forward.
- **Fixtures must reproduce what the code really produces.** Timestamping every photon
  at `interval.mid_time` made an attitude search unreachable; `run_events` draws
  timestamps uniform over the span.
- **A `true_*` column means *this event*, never a property of its source.** If
  per-event truth is not recoverable, write NaN rather than a plausible constant.
- **A green suite says nothing about the notebooks.** pytest never imports one. Until
  a PR's notebooks have been executed, they are unverified.
- **Verify against the code under review, not whatever is installed.** Confirm
  `package.__file__` resolves inside the worktree before believing any run.

Run: `pytest` (570 tests, ~90 s).

## Notebooks

- **Pass `progress = False`** to `run_events`/`run_binned` in every notebook. tqdm
  writes one stream record per refresh and nbconvert stores all of them (149 frames
  each in notebooks 00 and 02). Do **not** use `TQDM_DISABLE` instead -- tqdm binds it
  at import time, so it breaks silently if cells are reordered.
- **Never re-execute a notebook with `MPLBACKEND=Agg`.** It overrides the inline
  backend and silently strips every figure from the committed `.ipynb` while exiting
  0. CI sets Agg deliberately, which is fine there because it discards output via
  `--stdout`.
- The numbered notebooks stay in CI (the maintainer measures the job at ~120 s). Put
  anything slower in `docs/examples/cosimita/slow/`, which the non-recursive CI glob
  skips by construction.

## Working process

Two agents:

- **Main agent (me)** -- plans, writes the tests, reviews. Also does the independent
  verification before anything is pushed.
- **Implementation agent** -- a cheaper model, writes the code to the plan.

Agents **never open PRs and never force-push**; the orchestrator opens PRs after its
own verification. The maintainer reviews and merges every PR. Develop on
`claude/<topic>` branches cut from `main`.

Ask rather than assume, and ask before adding features beyond what was requested.

**Commit and push as soon as work is done**, not after polishing -- a rate limit
killing an agent mid-task has lost a whole round before.

## Environment

Python **>= 3.12**, **`astropy>=8.0`** (8.0 is the first release whose
`Quantity(str)` parses the bracketed array form `"[30, 0, 1] cm"` the YAML schema
uses -- a language-version floor does not imply a library floor).

CI: `test` on matrix `["3.12", "3.13"]`, `lint` (ruff) on 3.12, `notebooks` on 3.12
executing `docs/examples/cosimita/*.ipynb` non-recursively. Keep two versions in the
matrix -- a matrix spanning a range is what caught a real astropy portability bug
that a single pinned version would have missed.

Agent traps:

- The system Python lacks pandas and `pip install -e .` fails against Debian's
  managed `packaging`. Use
  `python -m venv --system-site-packages .venv && .venv/bin/pip install -e ".[test,lint]"`,
  created **inside** the agent's own worktree so the editable install points at its copy.
  Add `.venv/` to that worktree's `.git/info/exclude`.
- **Reclaim worktrees between features.** Each carries an ~800 MB-1 GB `.venv`; 29
  stale ones once held 22 GB and filled the disk mid-task. Remove every worktree whose
  HEAD is an ancestor of `main` -- it touches no branch ref. Check for unmerged ones first.
- `git push` can fail transiently with `could not read Username` while `git fetch`
  keeps working. Commits are not lost; re-push rather than redoing the work.
- **Clear `__pycache__` before quoting a warning count.** `SyntaxWarning` is emitted at
  compile time, so a warm run hides it. Cold, the suite shows 5 pre-existing
  `invalid escape sequence` warnings (`materials/material.py:68,72`,
  `tracker/tracker_2d.py:317`, `analysis/fast_norm_fit.py:8,145`).

## Deliberate quirks -- do not "fix" these

- The **doppler-broadening inconsistency** in `tracker_2d.py` is intentional: it is a
  fudged illustration of the effect for teaching. Fixing it would require
  regenerating responses and re-checking low-energy secondaries. Leave it.
- **Layers are infinitesimal planes.** The near-horizontal blind wedge is expected and
  is explained in the tutorials.
- `SimpleTraditionalReconstructor`'s **asymmetric up-going acceptance** is correct.
- **Eccentricity defaults to 0** (circular); `e = 1` is parabolic and has no period.
- The package has a **pre-existing import cycle**: `tracker_2d.py` does
  `from gammaraytoys import Material` while `gammaraytoys/__init__.py` imports
  `.detectors`. The import graph is *not* acyclic; don't claim otherwise.
