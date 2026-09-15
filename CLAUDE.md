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
| `coordinates/` | `Cartesian2D`, frame transforms |
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
`docs/dev/inertial_sim_plan.md` (complete; see `.claude/cosimita-progress.md`).

## Physics conventions that bite

**This is a 2D "flatland" world.** Fluxes are per unit *length* per time
(`1/cm/s`), never per area. Earth albedo emissivity is `1/(cm s)`. Getting this
wrong produces numbers that look entirely plausible and are wrong forever.

**Frames.** `attitude` A is the inertial angle of the detector's +y axis;
`sky_angle` is λ; `orbit_angle` is θ. Then `Nu = A - λ` and
`direction_det = 270° - Nu`. Nadir is `orbit_angle + 180°`.

**`phi` and `psi` are reserved for Compton Data Space.** Never reuse those names for
spacecraft geometry -- doing so has caused real confusion here. When plotting CDS,
it is `Histogram([psi_axis, phi_axis])`: Psi on x, Phi on y.

**The Earth albedo source is not occultable** (`occultable = False`) -- it *is* the
Earth. Occultation fraction is `2ρ/360°` with `ρ = arcsin(R_E/r)`; at 400 km that is
ρ = 70.2°, about 39% of the sky.

## Style

Tutorial-grade code: explicit, simple, unoptimized. **Up to a ~50% performance
penalty is an accepted trade for readability.** Don't vectorize a loop, add a cache,
or introduce a dependency to make something faster unless asked.

**`analysis/` has a stricter rule.** `likelihoods.py` (28 lines), `richardson_lucy.py`
(39), `response.py` (44) are deliberately short enough that a student can open them
and read the whole thing. New analysis capability must not make these longer or more
abstract -- build alongside them, don't refactor them into a framework.

## Testing

Expected values are derived **from first principles or back-of-envelope
calculation, never copied from running the implementation**. A test that records
what the code currently does proves nothing. Mutation testing (inject a deliberate
bug, confirm a test fails) is the quality gate -- but note it cannot catch a rule
that is the *wrong shape*, only one that is missing.

Run: `pytest` (570 tests, ~90 s).

## Working process

Two agents:

- **Main agent (me)** -- plans, writes the tests, reviews. Also does the independent
  verification before anything is pushed.
- **Implementation agent** -- a cheaper model, writes the code to the plan.

Agents **never open PRs and never force-push**; the orchestrator opens PRs after its
own verification. The maintainer reviews and merges every PR. Develop on
`claude/<topic>` branches cut from `main`.

Ask rather than assume, and ask before adding features beyond what was requested.

## Environment and CI

Python **>= 3.12**, **`astropy>=8.0`** (8.0 is the first release whose
`Quantity(str)` parses the bracketed array form `"[30, 0, 1] cm"` the YAML schema
uses -- a language-version floor does not imply a library floor).

CI: `test` on matrix `["3.12", "3.13"]`, `lint` (ruff) on 3.12, `notebooks` on 3.12
executing `docs/examples/cosimita/*.ipynb` non-recursively. Keep two versions in the
matrix -- a matrix spanning a range is what caught a real astropy portability bug
that a single pinned version would have missed.

**Clear `__pycache__` before quoting a warning count.** `SyntaxWarning` is emitted at
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
- The package has a **pre-existing import cycle**: `tracker_2d.py` does
  `from gammaraytoys import Material` while `gammaraytoys/__init__.py` imports
  `.detectors`. The import graph is *not* acyclic; don't claim otherwise.
