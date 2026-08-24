# TradDet

> **STALE — pending regeneration.** All three `.h5` files in this directory were
> generated before the interaction-probability bug in
> `ToyTracker2D.simulate_event` was fixed (see the "Fix inverted interaction
> probability in ToyTracker2D.simulate_event" commit). That bug made photons
> interact at essentially the first layer they geometrically crossed
> regardless of material/thickness/energy, so these cached responses do not
> reflect the corrected physics. They are being kept for now (regenerating
> the two larger ones takes hours — see below) but should be treated as
> reference/placeholder data until regenerated.
>
> Run `regenerate_responses.py` (in this directory) to rebuild them:
>
> ```
> python regenerate_responses.py energy_onaxis              # ~30 min
> python regenerate_responses.py energy_relative_onaxis      # ~4-5 h
> python regenerate_responses.py imaging_chiral_relative      # ~14-16 h
> ```
>
> The two longer ones are only practical as background/overnight jobs, and
> are good candidates to re-time after the simulator performance work planned
> for this cleanup — see the repo's cleanup notes.

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
