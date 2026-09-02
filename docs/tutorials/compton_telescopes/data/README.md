# TradDet

> **Regeneration status.** These `.h5` files were originally generated before
> the interaction-probability bug in `ToyTracker2D.simulate_event` was fixed
> (see the "Fix inverted interaction probability in ToyTracker2D.simulate_event"
> commit) -- that bug made photons interact at essentially the first layer
> they geometrically crossed regardless of material/thickness/energy.
>
> - `response_energy_onaxis_traddet.h5` -- **regenerated** with the fixed
>   simulator (see "Regenerate response_energy_onaxis_traddet.h5..." commit).
> - `response_energy_relative_onaxis_traddet.h5`,
>   `response_imaging_chiral_relative_1MeV_traddet.h5` -- still pending
>   regeneration as of their last update; check git history for this file to
>   see if that has since changed.
>
> Run `regenerate_responses.py` (in this directory) to (re)build any of them.
> The script is parallelized across CPU cores and includes the simulator
> performance fixes from this cleanup (see the script's docstring for
> details and current timing estimates):
>
> ```
> python regenerate_responses.py energy_onaxis              # ~2 min
> python regenerate_responses.py energy_relative_onaxis      # ~25-30 min
> python regenerate_responses.py imaging_chiral_relative      # ~1.5 h
> ```
>
> The last one is still long enough to want a background/overnight job on
> most machines (`nohup ... &`, see the script's docstring), but is no
> longer the multi-day proposition it would have been pre-optimization.

> **Reconstructor trigger guard (delete this note once the responses below
> are next regenerated).** `SimpleTraditionalReconstructor.reconstruct`
> (`gammaraytoys/sims/reco.py`) used to trigger on
> `hits.nhits >= 2 and hits.layer[0] == 0`, with no requirement that any hit
> land below the top layer. When every recorded hit fell in layer 0,
> `position_bottom = np.mean(hits.position[hits.layer > 0])` averaged an
> empty selection, `psi` came out NaN, and the event was still counted as
> triggered (there is a NaN guard for `phi` right below that line, but there
> was none for `psi`). The fix adds `and np.any(hits.layer > 0)` to the
> trigger condition.
>
> Mechanism (confirmed by dumping full interaction chains, including
> unrecorded interactions): the photon Comptons in layer 0, deposits energy
> below `energy_threshold` in a nearby layer (no hit recorded), backscatters,
> and interacts in layer 0 again -- recorded hits are `[0, 0]` with an
> invisible sub-threshold step between them. This needs closely spaced
> layers, so it is a geometry-dependent rate, not a fixed one:
>
> - Uniform 5 mm-pitch stack (the `tracker` fixture in `tests/conftest.py`):
>   **0.19% of triggers** were affected (4 NaN `psi` out of 2103 triggers
>   before the fix; 2099 triggers, 0 NaN, after -- 120000-photon
>   `IsotropicSource` run at 1 MeV, seed 0).
> - This "traditional detector" geometry (tracker 3 m above a calorimeter,
>   see below): **0%** affected (0 NaN `psi` out of 4716 triggers, identical
>   before and after the fix, same 120000-photon/seed-0 run) -- the 3 m
>   standoff is far wider than the sub-threshold backscatter step, so the
>   mechanism cannot occur.
>
> Because the measured rate for this geometry is zero, **the cached
> `.h5` responses below are unaffected by this fix and do not need
> regenerating for it** -- `regenerate_responses.py`'s `_traditional_detector()`
> uses exactly this geometry. This note exists only to record that; delete it
> whenever the responses are regenerated for any other reason.

The "traditional detector" corresponds to

```
det = ToyTracker2D(material = 'Ge',
                   layer_length = 10*u.m, 
                   layer_positions = np.append(300, np.arange(0,10,1))*u.cm, 
                   layer_thickness = 1*u.cm, 
                   energy_resolution = 0.03,
                   energy_threshold = 20*u.keV)
```

Using the SimpleTraditionalReconstructor and commit b5e2ebb

response_energy_onaxis_traddet.h5 was generated using an spectral index of -1.
