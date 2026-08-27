# Inertial-frame simulation for the 2D toy Compton detector

Implementation plan and requirements. Target reader: the agentic coder implementing
this, working PR by PR with the maintainer reviewing each one.

The nickname for this subsystem is **cosimita** — a toy educational counterpart to
MEGAlib's COSIMA. The name appears only in the notebook directory
(`docs/examples/cosimita/`); all code goes in the existing `gammaraytoys/sims/`.


## 1. Goal

Today the simulator throws photons at a detector that is the centre of the universe.
Sources are specified by their off-axis angle, and nothing moves.

We want to add a second mode in which the sky is fixed and the *spacecraft* moves:
it follows an orbit around a 2D Earth, changes attitude, has finite livetime, and is
periodically blocked by the Earth. Sources live at fixed inertial sky angles, or on
the Earth's surface, or bolted to the spacecraft itself.

The existing detector-frame simulation must keep working exactly as it does now. The
new inertial simulator sits on top of it and reaches it through a coordinate
transformation, rather than replacing it.


## 2. Hard constraints

These are not negotiable without asking the maintainer first.

1. **The tutorials keep working unchanged.** `docs/tutorials/compton_telescopes/`
   is pedagogical material: explicit, simple, unoptimised on purpose. No API it
   touches may change shape. Specifically, tutorials 01 and 02 call
   `source.random_photon(det)` *and* `source.random_photon(detector = det)`, so
   `detector` must stay the first positional-or-keyword parameter, and anything new
   goes after it with a default. Tutorials 02, 03 and 06 use
   `det.throwing_plane_size`; it stays.
2. **Readability beats speed.** Up to a 50% overall slowdown is acceptable if it
   buys clearer code. Do not vectorise the per-photon path, do not introduce a
   compiled dependency, do not restructure the detector walk for speed.
3. **The full test suite stays under 120 s.** It is ~20 s today. Statistical tests
   must be sized to fit that budget, and must be seeded (there is already an autouse
   `_seed_random` fixture in `tests/conftest.py`).
4. **The detector stays as it is.** Layers remain infinitesimal planes with the
   thickness entering only through the optical depth. This is a deliberate
   pedagogical approximation explained in the tutorials. See §8.5 for what it costs.
5. **One PR per feature.** Stop when a feature is complete and reviewable, open the
   PR, and wait. Group two features only when both are small and obviously related,
   and say why in the PR description. A single large PR needs justifying up front.
6. **Every PR carries tests and, where §7 says so, a notebook.**


## 3. Conventions

Get this section right and the rest is bookkeeping. Get it wrong and everything
downstream is subtly rotated.

### 3.1 The world is 2D

Flatland. Fluxes are per unit *length* per time (`1/cm/s`, not `1/cm2/s`). The Earth
is a circle. The sky is a 1D circle of directions. "Solid angle" is just an angle.

### 3.2 Symbol names — read this before naming anything

The existing code already uses `phi` for the Compton scattering angle and `psi` for
the Compton scatter direction, both in the Compton Data Space. **Do not reuse either
symbol for spacecraft geometry.** Use these names, in code and in comments:

| Quantity | Name in code | Symbol in comments | Meaning |
|---|---|---|---|
| Source direction on the inertial sky | `sky_angle` | λ | CCW from inertial +X, pointing *toward* the source |
| Spacecraft attitude | `attitude` | A | Inertial angle of the detector's **+y** axis, CCW from inertial +X |
| Spacecraft orbital position angle | `orbit_angle` | θ | CCW from inertial +X, Earth centre at the origin |
| Spacecraft orbital radius | `radius` | r | From Earth centre |
| Earth angular radius from the SC | — | ρ | `arcsin(R_E / r)` |
| Off-axis angle in the detector frame | `offaxis_angle` | Nu | Existing convention, unchanged |

### 3.3 Frames

**Inertial frame**: origin at the Earth's centre, angles CCW from +X.
Spacecraft at `(r cos θ, r sin θ)`.

**Detector frame**: the existing `ToyTracker2D` frame. Detector "zenith" is +y.
A source at off-axis angle Nu lies along the unit vector `(sin Nu, cos Nu)`, and the
photon it emits flies along direction `270° - Nu`. (Check: Nu = 0 gives direction
270°, i.e. straight down from overhead. Nu = 90° gives direction 180°, a source on
the +x side.)

Attitude A is the inertial angle of detector +y, so in inertial components:

```
y_detector = ( cos A,  sin A)
x_detector = ( sin A, -cos A)      # +y rotated by -90 deg (right-handed, z out of page)
```

### 3.4 The transformations

These are the only three formulas the inertial simulator needs. Derive them once in
a helper module, test them directly, and use them everywhere.

**Sky direction to off-axis angle** (far-field sources):

```
Nu = A - λ            (wrapped to [-180, 180) deg)
```

Sanity check for the test suite: A = 90° means detector +y points along inertial +Y.
A source at λ = 90° is then on-axis (Nu = 0). A source at λ = 0° sits at Nu = 90°,
i.e. on the detector's +x side, which in inertial coordinates is +X. Both correct.

**Inertial position to detector-frame position** (near sources, albedo emission
points). With `d = P - C`, where C is the spacecraft position:

```
x_det = d_X sin A - d_Y cos A
y_det = d_X cos A + d_Y sin A
```

**Inertial flight direction to detector-frame flight direction**:

```
direction_det = direction_inertial - A + 90 deg
```

Consistency check that must appear in the tests: a far-field photon from λ flies at
`direction_inertial = λ + 180°`, which the formula sends to `270° - (A - λ) = 270° - Nu`
— exactly the existing detector-frame convention. If this identity fails, the
transform is wrong.

### 3.5 Units

Times in seconds, orbital distances in km, detector distances in cm/mm, energies in
MeV/keV. Everything stays an `astropy` `Quantity` at API boundaries. Inside hot loops
the existing code drops to plain floats in a fixed unit; follow that pattern only
where it already exists.


## 4. The spacecraft history file (`.ori`)

### 4.1 Format

A CSV file with a `.ori` extension. One header line, units baked into the column
names. `#` starts a comment line; blank lines ignored.

```
time_s,radius_km,orbit_angle_deg,attitude_deg,uptime_s
0.0,6771.0,0.0,90.0,9.5
10.0,6771.0,0.081,90.081,10.0
20.0,6771.0,0.162,90.162,7.2
...
5400.0,6771.0,360.0,450.0,0.0
```

### 4.2 Interval semantics — the part that gets miscoded

Rows define timestamps `t_0 < t_1 < ... < t_N`. There are **N intervals**, not N+1.

Interval `i` spans `[t_i, t_{i+1})` and takes **both its pose and its uptime from
row i**. Concretely:

- pose (radius, orbit_angle, attitude) = row `i`
- livetime `L_i` = row `i`'s `uptime_s`
- span `Δt_i = t_{i+1} - t_i`

**Row N is a pure terminator.** It contributes only `t_N`, to close the last
interval. Its pose and its uptime are never read. The reader should not silently
require them to be meaningful, but a writer should emit something sensible anyway.

Note this makes `uptime_s` **forward**-looking: row `i`'s uptime is the live seconds
during `[t_i, t_{i+1})`, not the live seconds since `t_{i-1}`.

Validation on read, all of which must raise with a clear message:

- timestamps strictly increasing
- at least 2 rows
- `0 <= L_i <= Δt_i` for every interval
- `radius > 0`

### 4.3 API

```python
class SpacecraftHistory:
    @classmethod
    def open(cls, filename): ...
    def write(self, filename): ...

    @classmethod
    def from_elliptical_orbit(cls, semi_major_axis, eccentricity, earth,
                              attitude_model, time_step, duration = None,
                              argument_of_periapsis = 0*u.deg,
                              initial_time = 0*u.s, livetime_fraction = 1.0): ...

    @property
    def nintervals(self): ...      # = nrows - 1
    @property
    def total_livetime(self): ...  # sum of L_i over intervals, terminator excluded
    def __iter__(self): ...        # yields SpacecraftInterval
    def plot(self, ax = None, earth = None, nposes = 12): ...
```

`SpacecraftInterval` is a small frozen dataclass: `start_time`, `stop_time`,
`livetime`, `radius`, `orbit_angle`, `attitude`, plus a `mid_time` property.

### 4.4 Orbit generation

Solve Kepler's equation properly — it is about fifteen readable lines and it is
honest, whereas a uniform-in-angle ellipse is neither.

```
n     = sqrt(mu / a^3)                     # mean motion, mu = G M_earth
M     = n (t - t_periapsis)                # mean anomaly
M     = E - e sin E                        # solve for E by Newton iteration
tan(nu/2) = sqrt((1+e)/(1-e)) tan(E/2)     # true anomaly
r     = a (1 - e cos E)
theta = nu + argument_of_periapsis
```

Take `mu` from `astropy.constants` (`G * M_earth`). Newton on `E` converges in a
handful of iterations for `e < 0.9`; iterate to a fixed tolerance and raise if it
fails to converge rather than silently returning garbage.

`duration` defaults to one full orbital period.

Attitude models, as small callables `(time, radius, orbit_angle) -> attitude`:

- `ZenithPointing()` — detector +y points radially outward: `A = θ`
- `NadirPointing()` — `A = θ + 180°`
- `InertialPointing(attitude)` — constant `A`
- `SpinPointing(rate, initial_attitude=0*u.deg)` — `A = A_0 + rate * t`

`livetime_fraction` is a scalar in `[0, 1]` filling the `uptime_s` column as
`fraction * Δt`. It is a convenience for generated files only; real files may vary
per row.

### 4.5 Earth

```python
class Earth:
    def __init__(self, radius = None): ...          # defaults to astropy's R_earth
    def angular_radius(self, spacecraft_radius): ...      # rho = arcsin(R/r)
    def is_occulted(self, sky_angle, orbit_angle, spacecraft_radius): ...
    def plot(self, ax = None): ...
```

A single radius, used for both occultation and albedo emission. No atmosphere shell.

Occultation test — a far-field photon from sky angle λ is blocked iff it comes from
within the Earth's disc:

```
nadir = orbit_angle + 180 deg
occulted  <=>  |wrap(λ - nadir)| < rho
```

Raise if `spacecraft_radius < radius`.


## 5. Source model

### 5.1 The split

`Source` stays as the abstract base holding the spectrum, the chirality machinery,
and the spectrum plotting/discretising helpers. Two abstract subclasses appear
underneath it:

```
Source (ABC)                       spectrum, chirality, plot_spectrum, discretize_spectrum
├── FarFieldSource (ABC)           normalisation is a flux  [1/cm/s]
│   ├── PointSource                fixed sky_angle or fixed offaxis_angle
│   ├── IsotropicSource            uniform over the whole sky
│   ├── ExtendedSource             von Mises on the sky
│   └── EarthAlbedoSource          see 5.5 — far-field by throwing, near-field by sampling
└── NearFieldSource (ABC)          normalisation is a rate  [1/s]
    └── NearPointSource            isotropic emitter at a fixed detector-frame position
```

### 5.2 `simulated_rate()`

Every source exposes

```python
def simulated_rate(self, detector, pose = None) -> Quantity  # [1/s]
```

the expected rate of photons *launched at the detector*, before occultation and
before any time-dependent scaling. This is what lets the simulator mix flux-normalised
and rate-normalised sources in one run: it sums rates, not fluxes.

For every far-field source this is uniformly

```
simulated_rate = sky_integrated_flux(pose) * detector.throwing_plane_size
```

where `throwing_plane_size = 2a` and `a` is the surrounding-circle radius. Note what
`flux` means for a far-field source: the flux **integrated over the whole sky**, not
a per-unit-angle brightness. That is already the existing convention — `IsotropicSource`
draws `nsim = flux * duration * 2a` with a uniform direction — and every new far-field
source must match it, so that `ExtendedSource` with a very small width reproduces
`PointSource` and with a very large width reproduces `IsotropicSource`.

`sky_integrated_flux` is pose-independent for everything except the albedo, whose
value depends on `r`.

**`Simulator` refactor.** The existing detector-frame `Simulator` currently computes
`nsim` from `total_flux * duration * throwing_plane_size` and picks between sources
with probabilities proportional to flux. Replace both with `simulated_rate()`:
`total_rate = sum(s.simulated_rate(detector) for s in sources)`, `nsim = total_rate *
duration`, and source selection weighted by rate. For a run containing only far-field
sources this is numerically identical to today's behaviour, which the tests must
assert. `total_flux` is referenced by `tests/test_sims.py`; keep the property
(summing far-field fluxes, `None` if any source is near-field) or update the test,
and say which in the PR.

### 5.3 `random_photon()`

```python
def random_photon(self, detector, pose = None) -> Photon | None
```

Returns a `Photon` in the **detector frame**, ready for `detector.simulate_event()`,
or `None` if the photon was occulted. `pose` is a `SpacecraftInterval`; when it is
`None` the source behaves in pure detector-frame mode (this is the path the tutorials
take, and it must not change).

Sky sources hold **one reusable internal `PointSource`** and re-aim it, rather than
building a fresh one per photon — the pattern already established for
`IsotropicSource`. Its throwing-plane cache is keyed on the off-axis angle, so
re-aiming is safe.

### 5.4 `NearPointSource`

An isotropic emitter at a fixed **detector-frame** position, with a total emission
rate `rate` [1/s] over the full 2π. It does not move with the sky; `pose` is ignored.
This is the instrumental-background source: activation, a calibration source, a hot
component on the bus.

With `s` the distance from the source to the surrounding-circle centre and `a` the
surrounding-circle radius:

- **`s >= a`** (source outside the circle): the circle subtends a half-angle
  `Δ = arcsin(a/s)`. Draw the flight direction uniformly in `[φ_c - Δ, φ_c + Δ]`,
  where `φ_c` points from the source to the circle centre. Acceptance fraction
  `f = arcsin(a/s) / π`, so `simulated_rate = rate * f`.
- **`s < a`** (source inside the circle): `f = 1`, draw the direction uniformly over
  the full circle, `simulated_rate = rate`.

There is no rejection step in either branch — every direction drawn is one that
reaches the surrounding circle by construction. Whether it then interacts is the
detector's business, and §8.5 explains why some directions never will.

### 5.5 `ExtendedSource`

A far-field source with a **von Mises** distribution on the sky. Von Mises rather
than a Gaussian because it wraps by construction: no truncation parameter, no
double-counting near 360°, exactly normalised over the circle.

```python
ExtendedSource(sky_angle, width, spectrum, flux = ..., chirality = ..., ...)
```

`width` is the σ a user thinks in; convert internally as `kappa = 1 / width**2`,
which is exact in the small-width limit. Say so in the docstring. Sample with
`scipy.stats.vonmises` (already available through the existing scipy dependency).
`flux` is the total integrated over the whole sky.

Two limits fall out for free and both belong in the tests: `width → ∞` (κ → 0) must
reproduce `IsotropicSource`, and `width → 0` must reproduce `PointSource`.

### 5.6 `EarthAlbedoSource`

Gamma rays from the Earth's surface — a stand-in for atmospheric scattering, not a
real albedo model.

**Normalisation.** A surface emissivity `E` in `[1/cm/s]`: photons emitted per unit
length of Earth surface per second, into the outward half-plane. This is a property
of the surface, independent of the orbit, so the same source object means the same
physics at any altitude.

**Two emission laws**, selectable, and both may be used at once by putting two
sources in the same run with their own emissivities:

- `'isotropic'` — uniform over the outward half-plane (π rad), so emission per unit
  angle is `E/π`.
- `'lambertian'` — `∝ cos θ` from the local normal, normalised so
  `∫ k cos θ dθ = E` over the half-plane gives `k = E/2`.

**Why this cannot be sampled naively.** Picking a surface point, picking an isotropic
direction, and testing whether it hits has an acceptance of order 10⁻¹⁶ (a 17 cm
detector 6400 km away). It must be importance-sampled.

**How to sample it.** Specific intensity is conserved along rays, so from the
spacecraft the Earth is a patch of sky of angular radius ρ with a known brightness
profile. Sample the **emission point on the surface**, then hand the resulting
direction to the ordinary far-field throwing plane. Exact to `O(a/s) ≈ 3e-8`.

Parametrise a surface point by the Earth-central angle β from the sub-satellite
point, `|β| < β_max = arccos(R_E/r)`:

```
s(β)      = sqrt(r^2 + R_E^2 - 2 r R_E cos β)         # surface point to spacecraft
cos θ(β)  = (r cos β - R_E) / s(β)                    # emission angle from local normal
```

- **Lambertian**: the radiance is `k = E/2`, *independent of angle* — the Earth is a
  uniform-brightness disc. So there is nothing to sample on the surface at all: draw
  the sky angle **uniformly** in `[nadir - ρ, nadir + ρ]`. This is the simple case;
  implement it as such rather than forcing it through the machinery below.
- **Isotropic**: brightness goes as `1/cos θ`, limb-brightened and formally divergent
  at the limb (integrably, as ε^(-1/2)). **Sample in β, never in sky angle** — the
  surface measure removes the divergence entirely. Draw β from
  `pdf(β) ∝ 1/s(β)` on `[-β_max, β_max]`, then convert to a sky angle.

For the isotropic sampler, tabulate the pdf on a β grid, build the CDF with
`scipy.integrate.cumulative_trapezoid`, and inverse-transform with `np.interp`. About
six readable lines. Cache the table on `r` — it is constant for a circular orbit and
changes only per interval otherwise.

Converting β to a sky angle: with `P` the surface point and `C` the spacecraft
position, both in inertial coordinates, the photon appears to arrive from
`λ = angle_of(P - C)`. Then `Nu = A - λ` as usual, and the existing `PointSource`
throwing plane takes over.

**Total rates**, needed for `simulated_rate()`:

```
Lambertian:  N_dot = 2 a E arcsin(R_E / r)                                   # closed form
Isotropic:   N_dot = (2 a E R_E / π) ∫ dβ / s(β)   over [-β_max, β_max]      # numeric
```

The Lambertian closed form has been verified two independent ways — direct surface
integration and radiance conservation — agreeing to six decimals from 100 km to
100 000 km altitude. Use it as the analytic anchor in the tests. The isotropic form
has no elementary closed form; compute it with `scipy.integrate.quad` and cache on
`r`. It can be cross-checked against the equivalent sky-angle integral
`(2 a E / π) ∫ dλ / cos θ(λ)` with `sin θ = (r/R_E) sin λ`, which is an independent
route to the same number and makes a good test.

For orientation: at 400 km the isotropic law gives ~23% more flux than the Lambertian
one at equal emissivity, and the Earth fills ~140° of the sky.

**Occultation must NOT be applied to this source.** Its photons all come from the
Earth's direction and would be rejected wholesale. See §8.1.

### 5.7 Time-dependent normalisation

A unitless scaling multiplying the source's flux or rate, evaluated at each interval's
**midpoint**.

```python
class SourceScaling(ABC):
    def __call__(self, time) -> float: ...

class ConstantScaling(SourceScaling): ...             # default, 1.0
class TabulatedScaling(SourceScaling): ...            # piecewise constant
class FunctionScaling(SourceScaling): ...             # wraps any callable
```

`TabulatedScaling.open(filename)` reads a two-column CSV, `time_s,scale`, interpreted
**piecewise constant** to match `.ori` interval semantics: the scale from the last row
whose time is `<= t`. Before the first row, use the first row's value; after the last,
the last row's.

Every source takes an optional `scaling = None` constructor argument defaulting to
`ConstantScaling(1.0)`.


## 6. The inertial simulator

```python
class InertialSimulator:
    def __init__(self, detector, sources, reconstructor, spacecraft_history,
                 earth, doppler_broadening = True): ...

    def run_events(self, tstart = None, tstop = None): ...
    def run_binned(self, ..., tstart = None, tstop = None): ...
```

The existing `Simulator` is untouched apart from the `simulated_rate()` refactor in
§5.2. `InertialSimulator` is a separate class; it reaches the detector-frame physics
through the source transformations, not by subclassing.

Per-interval loop:

```
for interval in history:                       # skips the terminator row
    t_mid = interval.mid_time
    for source in sources:
        mu = source.simulated_rate(detector, interval) * interval.livetime * source.scaling(t_mid)
        N  = poisson(mu)                       # always Poisson, never a fixed count
        for _ in range(N):
            t      = uniform(interval.start_time, interval.stop_time)
            photon = source.random_photon(detector, interval)
            if photon is None:                 # occulted
                continue
            sim_event  = detector.simulate_event(photon, doppler_broadening = ...)
            reco_event = reconstructor.reconstruct(sim_event)
            yield t, source, sim_event, reco_event
```

Points that matter:

- **Poisson always**, per (source, interval). Never a rounded expectation.
- **Timestamps are uniform over the full interval span**, not over the live part.
  Livetime only scales the count, so deadtime is spread evenly through the interval
  rather than parked as an artificial gap at its end.
- **Occultation is a per-photon rejection.** Draw `N` from the *unocculted* mean, then
  test each photon's inertial arrival direction and discard it if blocked. The test is
  cheap and happens before the detector walk. For a point source this degenerates
  exactly to "the source is on or off for the whole interval"; unlike an analytic
  visible-fraction, it also handles isotropic and extended sources with no bespoke
  truncation maths per class.
- **No sub-interval refinement.** The pose is frozen for the whole interval.
- **Termination is the `.ori` range**, optionally narrowed by `tstart`/`tstop`.
- For the progress bar, sum `mu` over all intervals and sources first — with Poisson
  draws the total is not known from a photon count alone.


## 7. PR breakdown

Each PR: implementation, tests, and where listed a notebook in
`docs/examples/cosimita/`. Stop and open the PR at the end of each. Notebooks are
pedagogical — explicit and simple, matching the tutorials' style.

### PR 1 — Source hierarchy and `simulated_rate()`

Pure refactor, no new physics. `Source` splits into `FarFieldSource` /
`NearFieldSource`; `simulated_rate()` lands on both; `Simulator` switches its
normalisation and source-selection from fluxes to rates.

*Tests*: for a far-field-only run, `nsim` from the new path is identical to the old
formula; `simulated_rate == flux * throwing_plane_size`; multi-source selection
weights match the rate ratios. Tutorials 01–07 must still import and run their
first cells.

### PR 2 — `Earth`, `SpacecraftHistory`, orbit generation

No simulator changes. Reader, writer, validation, Kepler solver, the four attitude
models, and the plotting helpers.

*Tests*: round-trip write/read; every validation error fires; a circular orbit has
constant `r` and uniform `dθ/dt`; an eccentric orbit conserves specific angular
momentum `r² dθ/dt` and energy to a set tolerance; the period matches
`2π sqrt(a³/μ)`; `ZenithPointing` gives `A == θ`; occultation geometry matches
`arcsin(R/r)`; interval semantics — `nintervals == nrows - 1`, terminator pose and
uptime unused, `total_livetime` correct.

*Notebook*: `01-spacecraft_orbit_and_attitude.ipynb`.

### PR 3 — `InertialSimulator`, transforms, occultation

The transformation helpers from §3.4, `PointSource` gaining a `sky_angle`, and the
per-interval loop. `IsotropicSource` comes along for free.

*Tests*: the identity in §3.4 (`direction_det == 270° - Nu` for a far-field photon);
round-trip inertial → detector → inertial for positions and directions; **the sky
stands still** — for a fixed `sky_angle` and a rotating spacecraft, `A(t) - Nu_reco`
is constant across the run within the reconstruction resolution; total counts are
Poisson about `flux * 2a * total_livetime` within 3σ; a source at the pole of a
circular zenith-pointing orbit is never occulted, one in the orbital plane is
occulted for a fraction `ρ/π` of the orbit.

*Notebook*: `02-inertial_simulation_and_occultation.ipynb`.

### PR 4 — `NearPointSource` and `ExtendedSource`

Two small independent source geometries, grouped because each is well under
200 lines and both are "a new way to place a source".

*Tests*: near-source rate matches `rate * arcsin(a/s) / π`; the inside-the-circle
branch gives `f = 1` and a uniform direction distribution; a near source at the
detector centre triggers far more often than the same rate placed far outside;
`ExtendedSource` sky angles pass a KS test against `scipy.stats.vonmises`; the two
limits of §5.5 reproduce `IsotropicSource` and `PointSource` rates within 3σ.

*Notebook*: `03-near_and_extended_sources.ipynb`, including a cell showing the
blind wedge of §8.5 honestly.

### PR 5 — `EarthAlbedoSource`

Its own PR: the subtlest physics in the set, and worth a focused review.

*Tests*: Lambertian rate equals `2 a E arcsin(R_E/r)` in closed form at several
altitudes; isotropic rate matches an independent `quad` of the sky-angle integral;
sampled sky angles all fall within `ρ` of nadir; the sampled β distribution passes a
KS test against `pdf ∝ 1/s(β)`; the Lambertian sky-angle distribution is uniform over
`[nadir-ρ, nadir+ρ]`; the isotropic one is limb-brightened relative to it; the
albedo is **not** suppressed by occultation.

*Notebook*: `04-earth_albedo.ipynb`.

### PR 6 — Time-dependent scaling and event CSV I/O

Two small bookkeeping features grouped together; both are pure plumbing with no
physics, and neither is large enough to merit its own review cycle.

The event file is CSV with a `#`-prefixed YAML metadata header, so one file is
self-describing and nothing can drift apart from its data:

```
# gammaraytoys_version: 0.1.0
# ori_file: iss.ori
# total_livetime_s: 5130.0
# nsim: {crab: 4821, albedo: 19233}
# triggered_only: false
event_id,time_s,source,true_x_cm,true_y_cm,true_direction_deg,true_sky_angle_deg,
true_offaxis_angle_deg,true_energy_MeV,true_chirality,triggered,
reco_energy_MeV,reco_phi_deg,reco_psi_deg
```

`true_sky_angle_deg` is empty for detector-frame-native near sources. The reco
columns are empty when `triggered` is false. Rows cover every photon **actually
launched** at the detector; occulted photons never appear, since they were never
launched and the livetime metadata already accounts for them.
`triggered_only = True` writes only the triggers. Read back with pandas
(`comment = '#'`), parsing the header block into a metadata dict.

The existing `EventList` YAML dump of the full interaction tree stays as it is —
different job, still the right tool for inspecting a single event.

*Tests*: round-trip with units and NaNs preserved; `triggered_only` drops exactly the
untriggered rows and nothing else; the metadata header survives; a file with no events
still reads back; piecewise-constant scaling returns the right value on both sides of
each breakpoint and outside the table; a run with `scaling = 2` yields twice the counts
of `scaling = 1` within 3σ.

*Notebook*: `05-time_dependence_and_event_files.ipynb`.

### PR 7 — YAML configuration

`Simulator.from_config(...)` / `InertialSimulator.from_config(...)`. Python API only;
no console script, no `python -m`. Sketch:

```yaml
detector:
  type: ToyTracker2D
  material: Ge
  layer_length: 16 cm
  layer_positions: [30, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9] cm
  layer_thickness: 5 mm
  energy_resolution: 0.01
  energy_threshold: 20 keV

earth:
  radius: 6371 km

spacecraft_history: iss.ori

random_seed: 42

sources:
  - name: crab
    type: PointSource
    sky_angle: 45 deg
    flux: 1e-3 1/(cm s)
    spectrum: {type: PowerLaw, index: -2, min_energy: 0.2 MeV, max_energy: 10 MeV}
    scaling: {type: Function, expression: "1 + 0.5*sin(2*pi*t/5400)"}
  - name: albedo
    type: EarthAlbedoSource
    emission_law: lambertian
    emissivity: 1e-4 1/(cm s)
    spectrum: {type: PowerLaw, index: -1.5, min_energy: 0.2 MeV, max_energy: 5 MeV}
```

Every unit-bearing value is a string astropy parses. Unknown keys are an error, not a
warning — a silently ignored typo in a config file is a debugging nightmare. Validate
by hand with clear messages; do not add a schema-validation dependency.

`FunctionScaling` from a config string is the one place with an obvious injection
hazard. Do **not** use bare `eval`. Restrict the namespace to `t` plus an explicit
whitelist of numpy functions and constants, and reject anything containing `__`.

*Tests*: every source and spectrum type round-trips config → object → config;
`random_seed` makes two runs byte-identical; unknown keys raise; a malformed unit
raises with a message naming the offending key; the expression evaluator rejects
`__import__` and friends.

*Notebook*: `06-full_simulation_from_yaml.ipynb` — the capstone, running everything
from a single file.


## 8. Known traps

Collected from reading the existing code and probing it. Each has bitten or will bite.

1. **Occultation must not touch `EarthAlbedoSource`.** Its photons arrive from the
   Earth's direction by construction and a blanket occultation test rejects all of
   them. Give far-field sources an `occultable` property, `False` on the albedo.
2. **`random_photon(detector, pose = None)`** — tutorials 01 and 02 call this both
   positionally and as `detector = det`. `pose` must be a trailing keyword with a
   default, and `pose = None` must mean exactly today's behaviour.
3. **Do not reuse `phi` or `psi`** for spacecraft geometry; they are taken by the
   Compton Data Space. See §3.2.
4. **`det.throwing_plane_size`** appears in tutorials 02, 03 and 06. It stays.
5. **Near-field sources have a blind wedge.** Because layers are infinitesimal
   planes (constraint 4), a photon whose layer crossings all fall outside
   `|x| < layer_length/2` never interacts — measured: a source between layers emits
   at 1° from horizontal and interacts 0% of the time, versus ~47–60% for steep
   angles. This is correct behaviour under the documented approximation, and it shows
   up as reduced efficiency, not as a wrong normalisation. Do not silently work around
   it; show it in the notebook.
6. **The isotropic albedo's `1/cos θ` limb divergence** is integrable but real.
   Sampling in sky angle needs a singular pdf; sampling in β does not. Sample in β.
7. **`Simulator.total_flux`** is asserted on in `tests/test_sims.py`. Decide whether
   to keep the property or update the test, and say which in PR 1.
8. **Photon generation is astropy-bound.** Profiling `IsotropicSource.random_photon`
   shows ~447 µs/photon, 57% of it inside `throwing_plane()` and `Cartesian2D`
   building `Quantity` and `CartesianRepresentation` objects. The per-interval
   transform adds nothing to this; per-photon re-aiming does. Benchmark before and
   after and report the number in each PR, against the 50% budget of constraint 2.
9. **`SimpleTraditionalReconstructor` has a strong directional acceptance** — it
   requires the first hit in layer index 0. With a sky full of sources this shows up
   as a large asymmetry between up-going and down-going events (measured trigger
   fractions 7e-4 at Nu = 180° versus 7.3e-3 at Nu = 0°). This is real instrument
   behaviour, not a bug: `psi` stays geometrically correct for photons entering from
   below. Do not "fix" it.
10. **`arcsin(a/s)` breaks for `s < a`.** A near source inside the surrounding circle
    needs the separate branch of §5.4.
