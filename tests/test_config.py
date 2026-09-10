"""Tests for `gammaraytoys.sims.config` (`docs/dev/inertial_sim_plan.md`, the
PR 7 entry in Section 7): `Simulator.from_config` / `InertialSimulator.from_config`
and the YAML schema behind them.

Every expected value below is derived from the schema documented in the
`gammaraytoys.sims.config` module docstring, from the plan itself, or from
the CONTRACT.md/TEST_BRIEF.md written before this PR's implementation was
read -- never by running the implementation and copying its output. Where a
test only needs to prove a round trip ("build an object, write it back out,
read that back in, write it out again"), the two writes are compared to each
other, which is format-agnostic: it does not depend on knowing this module's
exact string formatting, only that it is self-consistent.

House style, matching `tests/test_scaling.py` and `tests/test_event_csv.py`:
statistical assertions state their sigma and assert at 4 sigma.
"""

import numpy as np
import pytest
import yaml
import astropy.units as u
from astropy.constants import R_earth

from gammaraytoys.sims import (
    ConstantScaling, Earth, EarthAlbedoSource, ExtendedSource,
    InertialPointing, InertialSimulator, IsotropicSource, MonoenergeticSpectrum,
    MultiComponentSpectrum, NadirPointing, NearPointSource, ObservationStrategy,
    PointSource,
    PowerLawSpectrum, SimpleTraditionalReconstructor, SpacecraftHistory,
    Source, SourceScaling, SpinPointing, Simulator, Spectrum,
    TabulatedScaling, TargetedPointing, ZenithPointing,
    detector_from_config, detector_to_config,
    load_config, reconstructor_from_config, reconstructor_to_config,
)
# Reading an array-with-unit string back is the one thing `u.Quantity` cannot
# do on every astropy this project supports: parsing a bracketed list out of a
# string is a recent feature, and `pyproject.toml` pins no version. The library
# parses that form itself for exactly this reason, so the three assertions
# below that read one back use the same parser rather than `u.Quantity`.
from gammaraytoys.sims.config_utils import _parse_quantity


# ===========================================================================
# Shared fixtures (plain dicts, not YAML text: `from_config` accepts an
# already-parsed mapping directly, and every unit-bearing value still has to
# be the string astropy parses -- that part of the schema does not depend on
# whether the surrounding structure came from YAML or a literal dict).
# ===========================================================================

# Same detector geometry as `tests/conftest.py`'s `tracker` fixture: known
# valid (no overlapping layers), so any failure below is the config layer's
# fault, not the detector's.
DETECTOR_BLOCK = {
    'type': 'ToyTracker2D',
    'material': 'Ge',
    'layer_length': '16 cm',
    'layer_positions': '[0, 5, 10, 20, 25, 30] mm',
    'layer_thickness': '5 mm',
    'energy_resolution': 0.01,
    'energy_threshold': '20 keV',
}

# The plan's own Section 7 sketch value -- deliberately not astropy's
# default `R_earth` (6378.1 km), so a config that sets `earth` is
# distinguishable from one that lets it default (see
# `test_earth_default_radius_is_astropy_r_earth_not_the_plans_6371`).
EARTH_BLOCK = {'radius': '6371 km'}

EARTH = Earth(radius=6371 * u.km)


def _spectrum_block(energy='1 MeV'):
    return {'type': 'Monoenergetic', 'energy': energy}


def _minimal_source_block():
    return {'type': 'IsotropicSource', 'flux': '1e-3 1/(cm s)',
            'spectrum': _spectrum_block()}


def _minimal_detector_frame_config():
    return {'detector': dict(DETECTOR_BLOCK), 'sources': [_minimal_source_block()]}


# ===========================================================================
# Part A -- spectra round-trip (CONTRACT.md inventory: 3 spectrum types)
# ===========================================================================

def test_monoenergetic_spectrum_round_trips():
    block = {'type': 'Monoenergetic', 'energy': '511 keV'}
    spectrum = Spectrum.from_config(block)

    assert isinstance(spectrum, MonoenergeticSpectrum)
    assert spectrum.energy == 511 * u.keV

    out = spectrum.to_config()
    assert out['type'] == 'Monoenergetic'
    assert u.Quantity(out['energy']) == 511 * u.keV

    spectrum2 = Spectrum.from_config(out)
    assert spectrum2.to_config() == out


def test_monoenergetic_spectrum_accepts_full_class_name_alias():
    spectrum = Spectrum.from_config({'type': 'MonoenergeticSpectrum', 'energy': '1 MeV'})
    assert isinstance(spectrum, MonoenergeticSpectrum)
    # The short name is always what is written back, regardless of which
    # spelling was read (module docstring, "Round trips").
    assert spectrum.to_config()['type'] == 'Monoenergetic'


def test_powerlaw_spectrum_round_trips():
    block = {'type': 'PowerLaw', 'index': -2, 'min_energy': '0.2 MeV', 'max_energy': '10 MeV'}
    spectrum = Spectrum.from_config(block)

    assert isinstance(spectrum, PowerLawSpectrum)
    assert spectrum.index == -2
    assert spectrum.min_energy == 0.2 * u.MeV
    assert spectrum.max_energy == 10 * u.MeV

    out = spectrum.to_config()
    assert out['type'] == 'PowerLaw'
    assert out['index'] == -2.0
    assert u.Quantity(out['min_energy']) == 0.2 * u.MeV
    assert u.Quantity(out['max_energy']) == 10 * u.MeV

    spectrum2 = Spectrum.from_config(out)
    assert spectrum2.to_config() == out


def test_powerlaw_spectrum_rejects_nonpositive_min_energy():
    block = {'type': 'PowerLaw', 'index': -2, 'min_energy': '0 MeV', 'max_energy': '10 MeV'}
    with pytest.raises(ValueError, match='min_energy'):
        Spectrum.from_config(block)


def test_powerlaw_spectrum_rejects_max_not_above_min():
    block = {'type': 'PowerLaw', 'index': -2, 'min_energy': '5 MeV', 'max_energy': '5 MeV'}
    with pytest.raises(ValueError, match='max_energy'):
        Spectrum.from_config(block)


def test_multicomponent_spectrum_round_trips_with_equal_weights_omitted():
    block = {'type': 'MultiComponent',
             'components': [_spectrum_block('511 keV'), _spectrum_block('1275 keV')]}
    spectrum = Spectrum.from_config(block)

    assert isinstance(spectrum, MultiComponentSpectrum)
    assert spectrum.ncomponents == 2
    np.testing.assert_allclose(spectrum.weights, [0.5, 0.5])

    out = spectrum.to_config()
    assert 'weights' not in out  # equal weights are the default: omitted

    spectrum2 = Spectrum.from_config(out)
    assert spectrum2.to_config() == out


def test_multicomponent_spectrum_round_trips_with_unequal_weights_kept():
    block = {'type': 'MultiComponent',
             'components': [_spectrum_block('511 keV'), _spectrum_block('1275 keV')],
             'weights': [1, 3]}
    spectrum = Spectrum.from_config(block)

    np.testing.assert_allclose(spectrum.weights, [0.25, 0.75])

    out = spectrum.to_config()
    assert out['weights'] == pytest.approx([0.25, 0.75])

    spectrum2 = Spectrum.from_config(out)
    assert spectrum2.to_config() == out
    # NOTE: `MultiComponentSpectrum.random_energy()` is a known pre-existing
    # bug (returns shape-(1,) where the rest of this codebase expects a
    # scalar) that breaks `simulate_event`. Per TEST_BRIEF.md this is
    # confirmed independent of PR 7; round-tripping the config is fine, but
    # no test here simulates one.


def test_multicomponent_spectrum_rejects_mismatched_weights_length():
    block = {'type': 'MultiComponent',
             'components': [_spectrum_block('511 keV'), _spectrum_block('1275 keV')],
             'weights': [1, 2, 3]}
    with pytest.raises(ValueError, match='weights'):
        Spectrum.from_config(block)


def test_spectrum_unknown_type_raises():
    with pytest.raises(ValueError, match='unknown spectrum type'):
        Spectrum.from_config({'type': 'Gaussian', 'energy': '511 keV'})


def test_spectrum_unknown_key_raises():
    block = {'type': 'Monoenergetic', 'energy': '511 keV', 'bogus': 1}
    with pytest.raises(ValueError, match='bogus'):
        Spectrum.from_config(block)


# ===========================================================================
# Part B -- scalings round-trip (CONTRACT.md inventory: 3 scaling types)
# ===========================================================================

def test_constant_scaling_round_trips():
    scaling = SourceScaling.from_config({'type': 'Constant', 'scale': 2.5})
    assert isinstance(scaling, ConstantScaling)
    assert scaling.scale == 2.5
    assert scaling.to_config() == {'type': 'Constant', 'scale': 2.5}


def test_constant_scaling_default_scale_is_one():
    scaling = SourceScaling.from_config({'type': 'Constant'})
    assert scaling.scale == 1.0


def test_tabulated_scaling_inline_round_trips():
    block = {'type': 'Tabulated', 'time': '[0, 100, 200] s', 'scale': [1.0, 2.0, 0.5]}
    scaling = SourceScaling.from_config(block)

    assert isinstance(scaling, TabulatedScaling)
    np.testing.assert_allclose(scaling.time.to_value(u.s), [0, 100, 200])
    np.testing.assert_allclose(scaling.scale, [1.0, 2.0, 0.5])

    out = scaling.to_config()
    scaling2 = SourceScaling.from_config(out)
    assert scaling2.to_config() == out


def test_tabulated_scaling_single_row_round_trips():
    # `_number`'s scalar `scale` is wrapped into a one-element list before
    # `TabulatedScaling` sees it (`SourceScaling.from_config`); pin
    # that this actually works for the smallest legal table.
    block = {'type': 'Tabulated', 'time': '[0] s', 'scale': 7.0}
    scaling = SourceScaling.from_config(block)
    assert scaling(0 * u.s) == 7.0
    assert scaling(1e9 * u.s) == 7.0


def test_tabulated_scaling_file_round_trips_and_is_written_inline(tmp_path):
    path = tmp_path / 'lightcurve.csv'
    path.write_text('time_s,scale\n0,1.0\n100,2.0\n200,0.5\n')

    scaling = SourceScaling.from_config({'type': 'Tabulated', 'file': str(path)})
    assert isinstance(scaling, TabulatedScaling)

    out = scaling.to_config()
    assert 'file' not in out
    assert 'time' in out and 'scale' in out

    scaling2 = SourceScaling.from_config(out)
    assert scaling2.to_config() == out


def test_tabulated_scaling_file_and_inline_together_raises(tmp_path):
    path = tmp_path / 'lightcurve.csv'
    path.write_text('time_s,scale\n0,1.0\n100,2.0\n')
    block = {'type': 'Tabulated', 'file': str(path), 'time': '[0, 100] s', 'scale': [1.0, 2.0]}
    with pytest.raises(ValueError, match='not both'):
        SourceScaling.from_config(block)


def test_tabulated_scaling_neither_file_nor_inline_raises():
    with pytest.raises(ValueError, match='needs either'):
        SourceScaling.from_config({'type': 'Tabulated'})


def test_scaling_unknown_type_raises():
    with pytest.raises(ValueError, match='unknown scaling type'):
        SourceScaling.from_config({'type': 'Sinusoid', 'scale': 1.0})


def test_scaling_unknown_key_raises():
    with pytest.raises(ValueError, match='bogus'):
        SourceScaling.from_config({'type': 'Constant', 'scale': 1.0, 'bogus': True})


# ===========================================================================
# Part C -- sources round-trip (CONTRACT.md inventory: 5 source types,
# PointSource counted twice for its two mutually exclusive forms)
# ===========================================================================

def test_pointsource_offaxis_form_round_trips():
    block = {'name': 'crab', 'type': 'PointSource', 'offaxis_angle': '30 deg',
             'flux': '1e-3 1/(cm s)', 'spectrum': _spectrum_block()}
    source = Source.from_config(block)

    assert isinstance(source, PointSource)
    assert source.sky_angle is None
    assert source.offaxis_angle == 30 * u.deg
    # Value and unit are checked separately: `pytest.approx` compares numbers,
    # and multiplying its result by a unit is not a comparison at all. The
    # unit is the one the block was written in -- `_quantity` returns a value
    # in the unit it was written with, it does not convert (module docstring).
    flux = source.flux()
    assert flux.unit == u.Unit('1/(cm s)')
    assert flux.value == pytest.approx(1e-3)

    out = source.to_config(name='crab')
    assert out['name'] == 'crab'
    assert out['type'] == 'PointSource'
    assert 'offaxis_angle' in out and 'sky_angle' not in out

    source2 = Source.from_config(out)
    assert source2.to_config(name='crab') == out


def test_pointsource_sky_angle_form_round_trips():
    block = {'type': 'PointSource', 'sky_angle': '45 deg', 'flux': '1e-3 1/(cm s)',
             'spectrum': _spectrum_block()}
    source = Source.from_config(block)

    assert source.offaxis_angle is None
    assert source.sky_angle == 45 * u.deg

    out = source.to_config()
    assert 'sky_angle' in out and 'offaxis_angle' not in out

    source2 = Source.from_config(out)
    assert source2.to_config() == out


def test_pointsource_flux_pivot_form_round_trips_to_a_flux():
    # A pivot flux is a *differential* flux, so resolving it to a total flux
    # divides by the spectrum's PDF at the pivot energy. That needs a spectrum
    # that has a PDF: a power law, not a monoenergetic line (whose PDF is a
    # delta and which refuses PDF evaluation outright -- see
    # `test_flux_pivot_with_a_monoenergetic_spectrum_is_refused`).
    #
    # The expected flux is worked out by hand from the power-law definition,
    # not read off the implementation. For index n over [Emin, Emax] the
    # normalization is
    #     norm = (1 + n) / (Emax*(Emax/Emin)**n - Emin)
    # and the PDF is norm*(E/Emin)**n. With n = -2, Emin = 1 MeV and
    # Emax = 10 MeV,
    #     norm = -1 / (10*10**-2 - 1) MeV^-1 = -1/(-0.9) MeV^-1 = 10/9 MeV^-1
    # and at the pivot E = Emin = 1 MeV the PDF is that same 10/9 per MeV.
    # A pivot flux of 1e-6 1/(cm s keV) is 1e-3 1/(cm s MeV), so
    #     flux = 1e-3 / (10/9) = 9e-4 1/(cm s).
    spec = {'type': 'PowerLaw', 'index': -2,
            'min_energy': '1 MeV', 'max_energy': '10 MeV'}
    block = {'type': 'PointSource', 'sky_angle': '0 deg',
             'flux_pivot': '1e-6 1/(cm s keV)', 'pivot_energy': '1 MeV', 'spectrum': spec}
    source = Source.from_config(block)

    flux = source.flux()
    assert flux.to_value('1/(cm s)') == pytest.approx(9e-4)

    # `to_config` writes the *resolved* flux, not flux_pivot/pivot_energy
    # (module docstring, "Round trips").
    out = source.to_config()
    assert 'flux' in out and 'flux_pivot' not in out
    assert 'pivot_energy' not in out

    source2 = Source.from_config(out)
    assert source2.flux().to_value('1/(cm s)') == pytest.approx(9e-4)
    assert source2.to_config() == out


def test_flux_pivot_with_a_monoenergetic_spectrum_is_refused():
    # A monoenergetic spectrum's PDF is a delta function, and
    # `MonoenergeticSpectrum` refuses to evaluate it. So a pivot flux -- which
    # can only be resolved to a total flux through that PDF -- cannot be used
    # with one, and the configuration layer must surface that refusal rather
    # than build a source with a silently missing normalization.
    block = {'type': 'PointSource', 'sky_angle': '0 deg',
             'flux_pivot': '1e-6 1/(cm s keV)', 'pivot_energy': '1 MeV',
             'spectrum': _spectrum_block('1 MeV')}

    with pytest.raises(ValueError, match='PDF'):
        Source.from_config(block)


def test_pointsource_both_forms_raises():
    block = {'type': 'PointSource', 'offaxis_angle': '10 deg', 'sky_angle': '20 deg',
             'flux': '1e-3 1/(cm s)', 'spectrum': _spectrum_block()}
    with pytest.raises(ValueError, match='exactly one'):
        Source.from_config(block)


def test_pointsource_neither_form_raises():
    block = {'type': 'PointSource', 'flux': '1e-3 1/(cm s)', 'spectrum': _spectrum_block()}
    with pytest.raises(ValueError, match='exactly one'):
        Source.from_config(block)


def test_isotropic_source_round_trips():
    block = {'type': 'IsotropicSource', 'flux': '2e-4 1/(cm s)', 'spectrum': _spectrum_block()}
    source = Source.from_config(block)
    assert isinstance(source, IsotropicSource)

    out = source.to_config()
    source2 = Source.from_config(out)
    assert source2.to_config() == out


def test_near_point_source_round_trips():
    block = {'type': 'NearPointSource', 'position': {'x': '0 cm', 'y': '1 cm'},
             'rate': '5 1/s', 'spectrum': _spectrum_block()}
    source = Source.from_config(block)

    assert isinstance(source, NearPointSource)
    assert source.position.x == 0 * u.cm
    assert source.position.y == 1 * u.cm
    assert source.rate == 5 * u.Hz

    out = source.to_config()
    source2 = Source.from_config(out)
    assert source2.to_config() == out


def test_near_point_source_missing_position_raises():
    block = {'type': 'NearPointSource', 'rate': '5 1/s', 'spectrum': _spectrum_block()}
    with pytest.raises(ValueError, match='position'):
        Source.from_config(block)


def test_extended_source_round_trips():
    block = {'type': 'ExtendedSource', 'sky_angle': '10 deg', 'width': '5 deg',
             'flux': '1e-3 1/(cm s)', 'spectrum': _spectrum_block()}
    source = Source.from_config(block)

    assert isinstance(source, ExtendedSource)
    assert source.sky_angle == 10 * u.deg
    assert source.width == 5 * u.deg

    out = source.to_config()
    source2 = Source.from_config(out)
    assert source2.to_config() == out


def test_earth_albedo_source_lambertian_round_trips():
    block = {'type': 'EarthAlbedoSource', 'emissivity': '1e-4 1/(cm s)',
             'spectrum': _spectrum_block()}
    source = Source.from_config(block, earth=EARTH)

    assert isinstance(source, EarthAlbedoSource)
    assert source.law == 'lambertian'
    assert source.earth is EARTH

    out = source.to_config()
    assert 'law' not in out  # default, omitted (module docstring)

    source2 = Source.from_config(out, earth=EARTH)
    assert source2.to_config() == out


def test_earth_albedo_source_isotropic_law_round_trips():
    block = {'type': 'EarthAlbedoSource', 'emissivity': '1e-4 1/(cm s)',
             'law': 'isotropic', 'spectrum': _spectrum_block()}
    source = Source.from_config(block, earth=EARTH)
    assert source.law == 'isotropic'
    assert source.to_config()['law'] == 'isotropic'


def test_earth_albedo_source_accepts_emission_law_alias():
    # `emission_law` is the plan's Section 7 sketch spelling; `law` is the
    # constructor argument (module docstring).
    block = {'type': 'EarthAlbedoSource', 'emissivity': '1e-4 1/(cm s)',
             'emission_law': 'isotropic', 'spectrum': _spectrum_block()}
    source = Source.from_config(block, earth=EARTH)
    assert source.law == 'isotropic'


def test_earth_albedo_source_law_and_emission_law_together_raises():
    block = {'type': 'EarthAlbedoSource', 'emissivity': '1e-4 1/(cm s)',
             'law': 'isotropic', 'emission_law': 'lambertian', 'spectrum': _spectrum_block()}
    with pytest.raises(ValueError, match='not both'):
        Source.from_config(block, earth=EARTH)


def test_earth_albedo_source_rejects_bad_law():
    block = {'type': 'EarthAlbedoSource', 'emissivity': '1e-4 1/(cm s)',
             'law': 'bogus', 'spectrum': _spectrum_block()}
    with pytest.raises(ValueError):
        Source.from_config(block, earth=EARTH)


def test_source_chirality_round_trips():
    block = {'type': 'IsotropicSource', 'flux': '1e-3 1/(cm s)', 'spectrum': _spectrum_block(),
             'chirality': 1, 'chirality_degree': 0.7}
    source = Source.from_config(block)

    assert source.chirality == 1
    assert source.chirality_degree == 0.7

    out = source.to_config()
    assert out['chirality'] == 1
    assert out['chirality_degree'] == 0.7

    source2 = Source.from_config(out)
    assert source2.to_config() == out


def test_source_scaling_is_written_back_out_by_to_config():
    # A round trip that only compares two `to_config` writes to each other is
    # blind to a field both writes leave out, so this one names the block it
    # expects. A source that silently lost its scaling would still round-trip
    # "cleanly" and then run at a constant rate -- exactly the kind of quiet
    # wrong answer this schema is meant to prevent.
    scaling_block = {'type': 'Tabulated', 'time': '[0.0, 100.0] s',
                     'scale': [1.0, 2.0]}
    block = {'type': 'IsotropicSource', 'flux': '1e-3 1/(cm s)',
             'spectrum': _spectrum_block(),
             'scaling': scaling_block}
    source = Source.from_config(block)

    assert isinstance(source.scaling, TabulatedScaling)

    out = source.to_config()
    assert out['scaling'] == scaling_block


def test_a_non_default_constant_scaling_is_written_back_out_too():
    block = {'type': 'IsotropicSource', 'flux': '1e-3 1/(cm s)',
             'spectrum': _spectrum_block(),
             'scaling': {'type': 'Constant', 'scale': 0.25}}
    source = Source.from_config(block)

    assert source.to_config()['scaling'] == {'type': 'Constant', 'scale': 0.25}


def test_a_default_scaling_is_the_only_one_omitted_from_to_config():
    # The other side of the same decision: `ConstantScaling(1.0)` is what a
    # source with no `scaling` key gets, so writing it back out would be
    # noise. Anything else must survive.
    source = Source.from_config(_minimal_source_block())

    assert isinstance(source.scaling, ConstantScaling)
    assert source.scaling.scale == 1.0
    assert 'scaling' not in source.to_config()


def test_source_default_chirality_is_omitted_from_to_config():
    source = Source.from_config(_minimal_source_block())
    out = source.to_config()
    assert 'chirality' not in out
    assert 'chirality_degree' not in out


def test_source_invalid_chirality_raises():
    block = dict(_minimal_source_block())
    block['chirality'] = 2
    with pytest.raises(ValueError, match='chirality'):
        Source.from_config(block)


def test_source_unknown_type_raises():
    with pytest.raises(ValueError, match='unknown source type'):
        Source.from_config({'type': 'Blazar', 'spectrum': _spectrum_block()})


def test_source_unknown_key_raises():
    block = dict(_minimal_source_block())
    block['bogus'] = 1
    with pytest.raises(ValueError, match='bogus'):
        Source.from_config(block)


def test_source_missing_spectrum_raises():
    with pytest.raises(ValueError, match='spectrum'):
        Source.from_config({'type': 'IsotropicSource', 'flux': '1e-3 1/(cm s)'})


# ===========================================================================
# Part D -- observation strategies round-trip (CONTRACT.md inventory: 5)
# ===========================================================================

def test_zenith_pointing_round_trips():
    strategy = ObservationStrategy.from_config({'type': 'ZenithPointing'})
    assert isinstance(strategy, ZenithPointing)
    assert strategy.to_config() == {'type': 'ZenithPointing'}


def test_nadir_pointing_round_trips():
    strategy = ObservationStrategy.from_config({'type': 'NadirPointing'})
    assert isinstance(strategy, NadirPointing)
    assert strategy.to_config() == {'type': 'NadirPointing'}


def test_inertial_pointing_round_trips():
    strategy = ObservationStrategy.from_config({'type': 'InertialPointing', 'attitude': '30 deg'})
    assert isinstance(strategy, InertialPointing)
    assert strategy.attitude == 30 * u.deg

    out = strategy.to_config()
    assert u.Quantity(out['attitude']) == 30 * u.deg


def test_spin_pointing_round_trips_default_initial_attitude_omitted():
    strategy = ObservationStrategy.from_config({'type': 'SpinPointing', 'rate': '0.1 deg/s'})
    assert isinstance(strategy, SpinPointing)
    assert strategy.initial_attitude == 0 * u.deg

    out = strategy.to_config()
    assert 'initial_attitude' not in out


def test_spin_pointing_round_trips_explicit_initial_attitude():
    strategy = ObservationStrategy.from_config(
        {'type': 'SpinPointing', 'rate': '0.1 deg/s', 'initial_attitude': '15 deg'})
    out = strategy.to_config()
    assert u.Quantity(out['initial_attitude']) == 15 * u.deg


def test_targeted_pointing_round_trips_with_earth():
    strategy = ObservationStrategy.from_config(
        {'type': 'TargetedPointing', 'sky_angle': '45 deg'}, earth=EARTH)
    assert isinstance(strategy, TargetedPointing)
    assert strategy.sky_angle == 45 * u.deg
    assert strategy.earth is EARTH

    out = strategy.to_config()
    assert u.Quantity(out['sky_angle']) == 45 * u.deg


def test_targeted_pointing_without_earth_raises():
    with pytest.raises(ValueError, match='Earth'):
        ObservationStrategy.from_config({'type': 'TargetedPointing', 'sky_angle': '45 deg'})


def test_observation_strategy_unknown_type_raises():
    with pytest.raises(ValueError, match='unknown observation strategy type'):
        ObservationStrategy.from_config({'type': 'SlewPointing'})


def test_observation_strategy_unknown_key_raises():
    with pytest.raises(ValueError, match='bogus'):
        ObservationStrategy.from_config({'type': 'ZenithPointing', 'bogus': 1})


# ===========================================================================
# Part E -- detector, Earth, reconstructor
# ===========================================================================

def test_detector_round_trips():
    detector = detector_from_config(DETECTOR_BLOCK)

    out = detector_to_config(detector)
    assert out['material'] == 'Ge'
    assert u.Quantity(out['layer_length']) == 16 * u.cm
    assert u.Quantity(out['layer_thickness']) == 5 * u.mm
    assert out['energy_resolution'] == 0.01
    assert u.Quantity(out['energy_threshold']) == 20 * u.keV

    detector2 = detector_from_config(out)
    assert detector_to_config(detector2) == out


def test_detector_missing_key_raises():
    block = dict(DETECTOR_BLOCK)
    del block['layer_length']
    with pytest.raises(ValueError, match='layer_length'):
        detector_from_config(block)


def test_detector_unknown_key_raises():
    block = dict(DETECTOR_BLOCK)
    block['bogus'] = 1
    with pytest.raises(ValueError, match='bogus'):
        detector_from_config(block)


def test_detector_unknown_type_raises():
    block = dict(DETECTOR_BLOCK)
    block['type'] = 'ToyTracker3D'
    with pytest.raises(ValueError, match='unknown detector type'):
        detector_from_config(block)


def test_earth_round_trips():
    earth = Earth.from_config(EARTH_BLOCK)
    assert earth.radius == 6371 * u.km

    out = earth.to_config()
    assert u.Quantity(out['radius']) == 6371 * u.km

    earth2 = Earth.from_config(out)
    assert earth2.to_config() == out


def test_earth_default_radius_is_astropy_r_earth_not_the_plans_6371():
    # module docstring: an empty `earth` block gives astropy's nominal
    # R_earth (6378.1 km), *not* the 6371 km the plan's own sketch uses.
    earth = Earth.from_config({})
    assert earth.radius == R_earth.to(u.km)
    assert earth.radius != 6371 * u.km


def test_earth_unknown_key_raises():
    with pytest.raises(ValueError, match='bogus'):
        Earth.from_config({'radius': '6371 km', 'bogus': 1})


def test_earth_nonpositive_radius_raises():
    with pytest.raises(ValueError, match='positive'):
        Earth.from_config({'radius': '-1 km'})


def test_reconstructor_round_trips():
    reconstructor = reconstructor_from_config({'type': 'SimpleTraditionalReconstructor'})
    assert isinstance(reconstructor, SimpleTraditionalReconstructor)
    assert reconstructor_to_config(reconstructor) == {'type': 'SimpleTraditionalReconstructor'}


def test_reconstructor_unknown_key_raises():
    with pytest.raises(ValueError, match='bogus'):
        reconstructor_from_config({'type': 'SimpleTraditionalReconstructor', 'bogus': 1})


def test_reconstructor_unknown_type_raises():
    with pytest.raises(ValueError, match='unknown reconstructor type'):
        reconstructor_from_config({'type': 'MLReconstructor'})


# ===========================================================================
# Part F -- malformed units name the offending key (CONTRACT.md, "Anchors
# for the round-trip tests")
# ===========================================================================

def test_malformed_unit_names_the_offending_key():
    block = dict(DETECTOR_BLOCK)
    block['layer_length'] = '16 banana'
    with pytest.raises(ValueError, match='layer_length'):
        detector_from_config(block)


def test_wrong_physical_type_unit_names_the_offending_key():
    block = dict(DETECTOR_BLOCK)
    block['layer_length'] = '16 kg'  # mass, not length
    with pytest.raises(ValueError, match='layer_length'):
        detector_from_config(block)


def test_bare_number_without_unit_names_the_offending_key():
    block = dict(DETECTOR_BLOCK)
    block['layer_length'] = 16  # no unit string at all
    with pytest.raises(ValueError, match='layer_length'):
        detector_from_config(block)


def test_unquoted_yaml_list_with_unit_names_the_offending_key():
    # A YAML list carries no unit of its own; the module docstring's
    # documented fix is a quoted string, e.g. "[30, 0, 1] cm".
    block = dict(DETECTOR_BLOCK)
    block['layer_positions'] = [30, 0, 1]  # a real YAML list, not a string
    with pytest.raises(ValueError, match='layer_positions'):
        detector_from_config(block)


def test_malformed_unit_in_a_source_block_names_the_key_not_just_astropys_message():
    block = dict(_minimal_source_block())
    block['flux'] = '1e-3 parsecs'  # a unit, but not one convertible to 1/(cm s)
    with pytest.raises(ValueError, match='flux'):
        Source.from_config(block)


# ===========================================================================
# Part G -- unknown keys raise at EVERY level (TEST_BRIEF item 3: one test
# per level, not one test overall)
# ===========================================================================

def test_unknown_key_at_top_level_raises():
    config = _minimal_detector_frame_config()
    config['bogus'] = 1
    with pytest.raises(ValueError, match='bogus'):
        Simulator.from_config(config)


def test_unknown_key_in_detector_block_raises():
    config = _minimal_detector_frame_config()
    config['detector'] = dict(config['detector'], bogus=1)
    with pytest.raises(ValueError, match='bogus'):
        Simulator.from_config(config)


def test_unknown_key_in_earth_block_raises():
    config = _minimal_detector_frame_config()
    config['earth'] = {'radius': '6371 km', 'bogus': 1}
    with pytest.raises(ValueError, match='bogus'):
        Simulator.from_config(config)


def test_unknown_key_in_source_block_raises():
    config = _minimal_detector_frame_config()
    config['sources'][0]['bogus'] = 1
    with pytest.raises(ValueError, match='bogus'):
        Simulator.from_config(config)


def test_unknown_key_in_spectrum_block_raises():
    config = _minimal_detector_frame_config()
    config['sources'][0]['spectrum']['bogus'] = 1
    with pytest.raises(ValueError, match='bogus'):
        Simulator.from_config(config)


def test_unknown_key_in_scaling_block_raises():
    config = _minimal_detector_frame_config()
    config['sources'][0]['scaling'] = {'type': 'Constant', 'scale': 1.0, 'bogus': 1}
    with pytest.raises(ValueError, match='bogus'):
        Simulator.from_config(config)


def _minimal_inertial_config(observation_strategy_extra=None, history_extra=None):
    history = {'type': 'elliptical_orbit', 'semi_major_axis': '6771 km',
               'duration': '10 s', 'time_step': '10 s'}
    if observation_strategy_extra is not None:
        history['observation_strategy'] = dict(
            {'type': 'ZenithPointing'}, **observation_strategy_extra)
    if history_extra is not None:
        history.update(history_extra)

    return {
        'detector': dict(DETECTOR_BLOCK),
        'earth': dict(EARTH_BLOCK),
        'sources': [{'type': 'PointSource', 'sky_angle': '0 deg', 'flux': '1e-3 1/(cm s)',
                     'spectrum': _spectrum_block()}],
        'spacecraft_history': history,
    }


def test_unknown_key_in_observation_strategy_block_raises():
    config = _minimal_inertial_config(observation_strategy_extra={'bogus': 1})
    with pytest.raises(ValueError, match='bogus'):
        InertialSimulator.from_config(config)


def test_unknown_key_in_spacecraft_history_block_raises():
    config = _minimal_inertial_config(history_extra={'bogus': 1})
    with pytest.raises(ValueError, match='bogus'):
        InertialSimulator.from_config(config)


def test_detector_frame_simulator_rejects_spacecraft_history_key():
    config = _minimal_detector_frame_config()
    config['spacecraft_history'] = 'nonexistent.ori'
    with pytest.raises(ValueError, match='spacecraft_history'):
        Simulator.from_config(config)


def test_inertial_simulator_requires_spacecraft_history_key():
    config = _minimal_detector_frame_config()
    with pytest.raises(ValueError, match='spacecraft_history'):
        InertialSimulator.from_config(config)


# ===========================================================================
# Part H -- source name policy (TEST_BRIEF item 8)
# ===========================================================================

def test_source_name_with_hash_is_refused_at_load_time():
    config = _minimal_detector_frame_config()
    config['sources'][0]['name'] = 'bad#name'
    with pytest.raises(ValueError, match='#'):
        Simulator.from_config(config)


def test_duplicate_source_names_are_refused():
    config = _minimal_detector_frame_config()
    config['sources'].append(dict(_minimal_source_block()))
    config['sources'][0]['name'] = 'dup'
    config['sources'][1]['name'] = 'dup'
    with pytest.raises(ValueError, match='dup'):
        Simulator.from_config(config)


def test_empty_source_name_is_refused():
    config = _minimal_detector_frame_config()
    config['sources'][0]['name'] = ''
    with pytest.raises(ValueError, match='empty'):
        Simulator.from_config(config)


def test_two_sources_with_no_name_at_all_are_allowed():
    # Only an actual collision is refused; two nameless sources do not
    # collide with each other (module docstring: "sources with no `name`
    # key are absent from it").
    config = _minimal_detector_frame_config()
    config['sources'].append(dict(_minimal_source_block()))
    simulator = Simulator.from_config(config)
    assert simulator.source_names == {}


# ===========================================================================
# Part I -- the plan's own Section 7 sketch is not valid YAML (TEST_BRIEF
# item 10)
# ===========================================================================

def test_plans_own_sketch_layer_positions_line_is_invalid_yaml(tmp_path):
    # `layer_positions: [30, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9] cm`, unquoted, is
    # a YAML flow sequence followed by trailing scalar text -- a parser
    # error, not (as it visually suggests) "a list with a unit on the end".
    path = tmp_path / 'sketch.yaml'
    path.write_text('layer_positions: [30, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9] cm\n')

    with pytest.raises(ValueError, match='not valid YAML'):
        load_config(path)


def test_quoting_the_sketchs_array_line_fixes_it():
    # The module docstring's documented fix for the line above: once the
    # array-with-unit is a quoted YAML string, it parses fine as an ordinary
    # `layer_positions` value. This is the sketch's own line, verbatim, in the
    # sketch's own units -- 1 cm apart with 5 mm layers, so the geometry is
    # valid and the only thing standing between the sketch and a working
    # detector really is the pair of quotes.
    positions = '[30, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9] cm'
    block = dict(DETECTOR_BLOCK, layer_positions=positions,
                 layer_thickness='5 mm')
    detector = detector_from_config(block)

    assert detector.layer_positions.unit == u.cm
    assert list(detector.layer_positions.to_value(u.cm)) == [30, 0, 1, 2, 3, 4,
                                                             5, 6, 7, 8, 9]


def test_load_config_rejects_a_yaml_file_whose_top_level_is_not_a_mapping(tmp_path):
    # A file that parses as YAML but holds a list at its top level: the
    # contents are wrong, not the argument, so this is a ValueError
    # (`load_config`'s docstring: ValueError "if the file ... does not hold a
    # mapping at its top level").
    path = tmp_path / 'a_list.yaml'
    path.write_text('- 1\n- 2\n- 3\n')

    with pytest.raises(ValueError, match='key/value pairs'):
        load_config(path)


def test_load_config_rejects_a_list_argument_as_a_type_error():
    # A list handed to `load_config` directly is neither a path nor a mapping,
    # which is a TypeError about the *argument* -- distinct from the ValueError
    # above about the *contents* of a file. `load_config`'s docstring documents
    # the split, and the two cases are worth keeping apart: one is a caller
    # bug, the other is a bad configuration file.
    with pytest.raises(TypeError, match='path to a YAML file or an already-parsed'):
        load_config([1, 2, 3])


def test_load_config_rejects_a_type_that_is_neither_path_nor_mapping():
    with pytest.raises(TypeError):
        load_config(12345)


def test_load_config_rejects_an_empty_file(tmp_path):
    path = tmp_path / 'empty.yaml'
    path.write_text('')
    with pytest.raises(ValueError, match='empty'):
        load_config(path)


def test_load_config_does_not_mutate_the_callers_mapping():
    config = _minimal_detector_frame_config()
    original_keys = set(config.keys())
    loaded = load_config(config)
    loaded['bogus_added_after_the_fact'] = 1
    assert set(config.keys()) == original_keys


# ===========================================================================
# Part J -- a hand-built simulator's `to_config` names what it cannot
# recover (TEST_BRIEF item 9)
# ===========================================================================

def test_to_config_on_hand_built_inertial_simulator_raises():
    detector = detector_from_config(DETECTOR_BLOCK)
    earth = Earth(radius=6371 * u.km)
    history = SpacecraftHistory.from_elliptical_orbit(
        semi_major_axis=6771 * u.km, duration=100 * u.s, time_step=50 * u.s, earth=earth)
    source = PointSource(sky_angle=0 * u.deg, spectrum=MonoenergeticSpectrum(1 * u.MeV),
                         flux=1e-3 * u.Unit('1/(cm s)'))

    simulator = InertialSimulator(detector=detector, sources=[source],
                                  reconstructor=SimpleTraditionalReconstructor(),
                                  spacecraft_history=history, earth=earth)

    with pytest.raises(ValueError, match='from_config'):
        simulator.to_config()


def test_to_config_on_hand_built_detector_frame_simulator_does_not_raise():
    # The detector-frame `Simulator` has no spacecraft history to lose, so
    # `to_config` on one built directly succeeds -- it just has no source
    # names to write, which is a legitimate state (an unnamed source), not
    # a guess.
    detector = detector_from_config(DETECTOR_BLOCK)
    source = IsotropicSource(spectrum=MonoenergeticSpectrum(1 * u.MeV),
                             flux=1e-3 * u.Unit('1/(cm s)'))
    simulator = Simulator(detector=detector, sources=[source],
                          reconstructor=SimpleTraditionalReconstructor())

    out = simulator.to_config()
    assert 'name' not in out['sources'][0]


# ===========================================================================
# Part K -- the full inventory round-trips together, canonically and
# idempotently, and survives yaml.safe_dump/safe_load (TEST_BRIEF item 6;
# CONTRACT.md "Everything must round-trip")
# ===========================================================================

FULL_CONFIG = {
    'detector': dict(DETECTOR_BLOCK),
    'earth': dict(EARTH_BLOCK),
    'reconstructor': {'type': 'SimpleTraditionalReconstructor'},
    'doppler_broadening': True,
    'random_seed': 7,
    'spacecraft_history': {
        'type': 'elliptical_orbit',
        'semi_major_axis': '6771 km',
        'eccentricity': 0.01,
        'duration': '200 s',
        'time_step': '50 s',
        'argument_of_periapsis': '10 deg',
        'observation_strategy': {'type': 'SpinPointing', 'rate': '0.05 deg/s'},
    },
    'sources': [
        {'name': 'crab', 'type': 'PointSource', 'sky_angle': '45 deg',
         'flux': '1e-3 1/(cm s)',
         'spectrum': {'type': 'PowerLaw', 'index': -2,
                      'min_energy': '0.2 MeV', 'max_energy': '10 MeV'},
         'scaling': {'type': 'Sinusoidal', 'mean': 1.0, 'amplitude': 0.5,
                     'period': '5400.0 s'}},
        {'name': 'iso', 'type': 'IsotropicSource', 'flux': '5e-4 1/(cm s)',
         'spectrum': _spectrum_block('0.5 MeV')},
        {'name': 'near', 'type': 'NearPointSource',
         'position': {'x': '0 cm', 'y': '2 cm'}, 'rate': '3 1/s',
         'spectrum': _spectrum_block('0.1 MeV')},
        {'name': 'ext', 'type': 'ExtendedSource', 'sky_angle': '90 deg', 'width': '5 deg',
         'flux': '2e-4 1/(cm s)', 'spectrum': _spectrum_block('0.4 MeV')},
        {'name': 'albedo', 'type': 'EarthAlbedoSource', 'emissivity': '1e-4 1/(cm s)',
         'law': 'isotropic', 'spectrum': _spectrum_block('0.3 MeV')},
    ],
}


def test_full_config_round_trips_canonically_and_idempotently():
    simulator1 = InertialSimulator.from_config(FULL_CONFIG)
    out1 = simulator1.to_config()

    simulator2 = InertialSimulator.from_config(out1)
    out2 = simulator2.to_config()

    # Canonical: writing back what was just read gives the same thing again.
    assert out1 == out2

    # Idempotent: doing it a third time changes nothing further.
    simulator3 = InertialSimulator.from_config(out2)
    assert simulator3.to_config() == out2


def test_full_config_survives_yaml_safe_dump_and_safe_load():
    simulator1 = InertialSimulator.from_config(FULL_CONFIG)
    out1 = simulator1.to_config()

    # `to_config`'s own contract: "holding only plain strings, numbers,
    # lists and dictionaries, so it can be written straight out with
    # `yaml.safe_dump`".
    reloaded = yaml.safe_load(yaml.safe_dump(out1))

    simulator2 = InertialSimulator.from_config(reloaded)
    assert simulator2.to_config() == out1


def test_full_config_builds_every_source_type_in_the_inventory():
    simulator = InertialSimulator.from_config(FULL_CONFIG)
    types = {type(s).__name__ for s in simulator.sources}
    assert types == {'PointSource', 'IsotropicSource', 'NearPointSource',
                     'ExtendedSource', 'EarthAlbedoSource'}


def test_full_config_wires_the_same_earth_into_everything():
    # CONTRACT.md: "a config cannot produce a run with two different
    # planets" -- the single `earth` block reaches the history and the
    # albedo source alike.
    simulator = InertialSimulator.from_config(FULL_CONFIG)
    albedo = next(s for s in simulator.sources if isinstance(s, EarthAlbedoSource))
    assert albedo.earth.radius == simulator.earth.radius == 6371 * u.km


# ===========================================================================
# Part L -- `random_seed` makes two runs byte-identical (TEST_BRIEF item 2;
# mutation target (d))
# ===========================================================================

SEEDED_CONFIG = {
    'detector': dict(DETECTOR_BLOCK),
    'earth': dict(EARTH_BLOCK),
    'random_seed': 20260909,
    'spacecraft_history': {
        'type': 'elliptical_orbit',
        'semi_major_axis': '6771 km',
        'duration': '300 s',
        'time_step': '60 s',
        'observation_strategy': {'type': 'ZenithPointing'},
    },
    'sources': [
        {'name': 'crab', 'type': 'PointSource', 'sky_angle': '45 deg',
         'flux': '5e-2 1/(cm s)', 'spectrum': _spectrum_block('1 MeV'),
         'scaling': {'type': 'Sinusoidal', 'mean': 1.0, 'amplitude': 0.5,
                     'period': '5400.0 s'}},
        {'name': 'albedo', 'type': 'EarthAlbedoSource', 'emissivity': '2e-3 1/(cm s)',
         'spectrum': _spectrum_block('0.3 MeV')},
    ],
}


def _event_fields(simulator, events):
    """Every field the PR 6 event-file round trip compared, field by field,
    read straight off the live event stream rather than a CSV -- source is
    compared by name (or class name, for an unnamed source) since the two
    runs build entirely separate `Source` objects."""
    names = simulator.source_names or {}
    fields = []
    for time, source, sim_event, reco_event in events:
        fields.append((
            time.to_value(u.s),
            names.get(source, type(source).__name__),
            sim_event.energy.to_value(u.keV),
            sim_event.direction.to_value(u.deg),
            sim_event.chirality,
            reco_event.triggered,
            reco_event.energy.to_value(u.keV) if reco_event.triggered else None,
            reco_event.phi.to_value(u.deg) if reco_event.triggered else None,
            reco_event.psi.to_value(u.deg) if reco_event.triggered else None,
        ))
    return fields


def test_random_seed_makes_two_inertial_runs_byte_identical():
    # Deliberately disturb the global RNG between the two `from_config`
    # calls (TEST_BRIEF.md: "a test that does not disturb it would pass
    # even if the seed were never applied").
    np.random.seed(999)
    np.random.uniform(size=500)

    simulator1 = InertialSimulator.from_config(SEEDED_CONFIG)
    events1 = _event_fields(simulator1, list(simulator1.run_events(progress=False)))

    np.random.seed(12345)
    np.random.uniform(size=37)

    simulator2 = InertialSimulator.from_config(SEEDED_CONFIG)
    events2 = _event_fields(simulator2, list(simulator2.run_events(progress=False)))

    assert len(events1) > 0  # otherwise the comparison below is vacuous
    assert events1 == events2


def test_random_seed_makes_two_detector_frame_runs_byte_identical():
    config = dict(_minimal_detector_frame_config(), random_seed=555)

    np.random.seed(1)
    np.random.uniform(size=200)
    simulator1 = Simulator.from_config(config)
    events1 = [(e.energy.to_value(u.keV), e.direction.to_value(u.deg), e.chirality)
              for e, _ in simulator1.run_events(nsim=200, progress=False)]

    np.random.seed(2)
    np.random.uniform(size=71)
    simulator2 = Simulator.from_config(config)
    events2 = [(e.energy.to_value(u.keV), e.direction.to_value(u.deg), e.chirality)
              for e, _ in simulator2.run_events(nsim=200, progress=False)]

    assert events1 == events2


def test_random_seed_is_applied_after_everything_is_built(monkeypatch):
    # The ordering the module docstring commits to: the seed goes on *last*,
    # "after every object is built, so that what follows the call -- the run
    # itself -- always starts from the same RNG state, no matter what
    # construction did or did not draw."
    #
    # Nothing in today's construction path draws from the global RNG, so the
    # ordering is invisible unless something does. This test supplies exactly
    # that: one construction step is wrapped so that it draws, standing in for
    # any future step that starts to (a randomized default, a sampled table, a
    # spectrum that pre-draws). Seeded last, those draws are wiped out by the
    # seed; seeded first, they eat into the seeded stream and every run that
    # follows is shifted.
    real_from_config = Spectrum.from_config

    def spectrum_from_config_that_draws(*args, **kwargs):
        np.random.uniform(size=3)
        return real_from_config(*args, **kwargs)

    monkeypatch.setattr(Spectrum, 'from_config', spectrum_from_config_that_draws)

    config = dict(_minimal_detector_frame_config(), random_seed=2026)

    # Disturb the global RNG first, so a seed that is never applied at all
    # cannot pass this either.
    np.random.seed(999)
    np.random.uniform(size=17)

    Simulator.from_config(config)
    after_from_config = list(np.random.uniform(size=5))

    # What `np.random.seed(2026)` alone gives -- numpy's own guarantee, not
    # anything read off this package.
    np.random.seed(2026)
    expected = list(np.random.uniform(size=5))

    assert after_from_config == expected


def test_no_random_seed_means_seed_is_none_and_nothing_is_seeded():
    config = _minimal_detector_frame_config()
    simulator = Simulator.from_config(config)
    assert simulator.random_seed is None


def test_random_seed_round_trips_through_to_config():
    config = dict(_minimal_detector_frame_config(), random_seed=42)
    simulator = Simulator.from_config(config)
    assert simulator.random_seed == 42
    assert simulator.to_config()['random_seed'] == 42


# ===========================================================================
# Part M -- a PointSource's normalization: one form, whole, and usable
# ===========================================================================
#
# A `PointSource` is normalized either by `flux` -- a total flux -- or by the
# pair `flux_pivot` and `pivot_energy`, a differential flux and the energy it
# is quoted at. `PointSource` itself takes whichever it is given, quietly
# prefers `flux`, and leaves the flux unset when the pair is half there. None
# of that is visible from a configuration file, and all three of the ways it
# goes wrong end the same way: a file that loads, runs, and is written back
# out as a *different* file from the one that was read.

def _power_law_block():
    # A spectrum with a PDF, which a pivot flux needs (a monoenergetic line's
    # PDF is a delta and refuses to be evaluated at all).
    return {'type': 'PowerLaw', 'index': -2,
            'min_energy': '1 MeV', 'max_energy': '10 MeV'}


def test_flux_beside_flux_pivot_is_refused():
    # Given both, `PointSource` uses `flux` and drops the pivot pair without
    # saying so, and `to_config` then writes the file back out with the pivot
    # pair gone -- a silently different file.
    block = {'type': 'PointSource', 'sky_angle': '0 deg',
             'flux': '1e-3 1/(cm s)', 'flux_pivot': '1e-6 1/(cm s keV)',
             'pivot_energy': '1 MeV', 'spectrum': _power_law_block()}

    with pytest.raises(ValueError, match='not both'):
        Source.from_config(block)


def test_flux_beside_pivot_energy_alone_is_refused():
    # Half a pivot pair beside a `flux` is the same conflict: `pivot_energy`
    # means nothing without `flux_pivot`, and would be dropped just as
    # quietly.
    block = {'type': 'PointSource', 'sky_angle': '0 deg',
             'flux': '1e-3 1/(cm s)', 'pivot_energy': '1 MeV',
             'spectrum': _power_law_block()}

    with pytest.raises(ValueError, match='not both'):
        Source.from_config(block)


def test_flux_pivot_without_pivot_energy_is_refused():
    # The two are halves of one number. Given only one, `PointSource` leaves
    # the flux unset: the run draws from an unnormalized source and the lone
    # key vanishes from what is written back out.
    block = {'type': 'PointSource', 'sky_angle': '0 deg',
             'flux_pivot': '1e-6 1/(cm s keV)', 'spectrum': _power_law_block()}

    with pytest.raises(ValueError, match='pivot_energy'):
        Source.from_config(block)


def test_pivot_energy_without_flux_pivot_is_refused():
    block = {'type': 'PointSource', 'sky_angle': '0 deg',
             'pivot_energy': '1 MeV', 'spectrum': _power_law_block()}

    with pytest.raises(ValueError, match='flux_pivot'):
        Source.from_config(block)


def test_a_pointsource_with_no_normalization_at_all_is_still_allowed():
    # "No flux" is a legitimate thing to write: such a source can be drawn
    # from but not counted, and the module docstring says so. Only a *half*
    # normalization is refused.
    block = {'type': 'PointSource', 'sky_angle': '0 deg',
             'spectrum': _power_law_block()}
    source = Source.from_config(block)

    assert source.flux() is None

    out = source.to_config()
    assert 'flux' not in out
    assert Source.from_config(out).to_config() == out


def test_a_pivot_energy_where_the_spectrum_has_no_density_is_refused():
    # The pivot flux is resolved by dividing by the spectrum's probability
    # density at `pivot_energy`, and that density is exactly zero outside the
    # spectrum's own range. The division does not fail -- it returns infinity
    # -- so without a check the source is built with an infinite flux and
    # `to_config` writes `flux: inf ...` back into the file as if someone had
    # meant it. This spectrum runs from 1 to 10 MeV; 100 MeV is outside it.
    block = {'type': 'PointSource', 'sky_angle': '0 deg',
             'flux_pivot': '1e-6 1/(cm s keV)', 'pivot_energy': '100 MeV',
             'spectrum': _power_law_block()}

    with pytest.raises(ValueError, match='zero'):
        Source.from_config(block)


def test_the_refusal_names_the_energy_range_the_pivot_should_have_been_in():
    # The message has to say where the pivot could have gone, or the reader
    # is left to work out the spectrum's range for themselves.
    block = {'type': 'PointSource', 'sky_angle': '0 deg',
             'flux_pivot': '1e-6 1/(cm s keV)', 'pivot_energy': '0.5 MeV',
             'spectrum': _power_law_block()}

    with pytest.raises(ValueError, match='pivot_energy') as caught:
        Source.from_config(block)

    assert '1' in str(caught.value) and '10' in str(caught.value)


def test_a_pivot_energy_inside_the_range_is_still_fine():
    # The check must not have closed the door on the ordinary case: the same
    # hand-computed flux as `test_pointsource_flux_pivot_form_round_trips`,
    # at a pivot energy in the middle of the range rather than at its edge.
    # For index -2 over [1, 10] MeV the normalization is
    #     norm = (1 + n) / (Emax*(Emax/Emin)**n - Emin) = 10/9 per MeV,
    # and the PDF at E is norm*(E/Emin)**n, so at E = 2 MeV it is
    #     (10/9)*(1/4) = 10/36 per MeV.
    # A pivot flux of 1e-6 1/(cm s keV) is 1e-3 1/(cm s MeV), so
    #     flux = 1e-3 / (10/36) = 3.6e-3 1/(cm s).
    block = {'type': 'PointSource', 'sky_angle': '0 deg',
             'flux_pivot': '1e-6 1/(cm s keV)', 'pivot_energy': '2 MeV',
             'spectrum': _power_law_block()}
    source = Source.from_config(block)

    assert source.flux().to_value('1/(cm s)') == pytest.approx(3.6e-3)


# ===========================================================================
# Part N -- the quantities that cannot sensibly be negative are bounded
# ===========================================================================
#
# Before this, `flux: "-1e-3 1/(cm s)"` loaded without a word, round-tripped
# verbatim, and died at the first interval of the run inside numpy, with
# `lam < 0 or lam is NaN` and not one word about which key, which source or
# which file. The bound belongs where the file is read.

def test_a_negative_point_source_flux_is_refused_naming_the_key():
    block = {'type': 'PointSource', 'sky_angle': '0 deg',
             'flux': '-1e-3 1/(cm s)', 'spectrum': _spectrum_block()}

    with pytest.raises(ValueError, match="'flux'") as caught:
        Source.from_config(block)

    assert '>= 0' in str(caught.value)
    # The offending value, so the reader can find the line.
    assert '-1e-3 1/(cm s)' in str(caught.value)


def test_a_negative_isotropic_flux_is_refused():
    block = {'type': 'IsotropicSource', 'flux': '-2e-4 1/(cm s)',
             'spectrum': _spectrum_block()}

    with pytest.raises(ValueError, match="'flux'"):
        Source.from_config(block)


def test_a_negative_extended_source_flux_is_refused():
    block = {'type': 'ExtendedSource', 'sky_angle': '90 deg', 'width': '5 deg',
             'flux': '-2e-4 1/(cm s)', 'spectrum': _spectrum_block()}

    with pytest.raises(ValueError, match="'flux'"):
        Source.from_config(block)


def test_a_negative_flux_pivot_or_pivot_energy_is_refused():
    negative_pivot = {'type': 'PointSource', 'sky_angle': '0 deg',
                      'flux_pivot': '-1e-6 1/(cm s keV)', 'pivot_energy': '2 MeV',
                      'spectrum': _power_law_block()}
    with pytest.raises(ValueError, match="'flux_pivot'"):
        Source.from_config(negative_pivot)

    negative_energy = {'type': 'PointSource', 'sky_angle': '0 deg',
                       'flux_pivot': '1e-6 1/(cm s keV)', 'pivot_energy': '-2 MeV',
                       'spectrum': _power_law_block()}
    with pytest.raises(ValueError, match="'pivot_energy'"):
        Source.from_config(negative_energy)


def test_a_negative_near_point_source_rate_is_refused():
    block = {'type': 'NearPointSource', 'position': {'x': '0 cm', 'y': '2 cm'},
             'rate': '-3 1/s', 'spectrum': _spectrum_block()}

    with pytest.raises(ValueError, match="'rate'"):
        Source.from_config(block)


def test_a_zero_flux_is_still_allowed():
    # The bound is inclusive, and it has to be: a source switched off for a
    # run is a reasonable thing to write, and zero photons is a perfectly
    # well-defined Poisson mean.
    block = {'type': 'IsotropicSource', 'flux': '0 1/(cm s)',
             'spectrum': _spectrum_block()}
    source = Source.from_config(block)

    assert source.flux().to_value('1/(cm s)') == 0.0


@pytest.mark.parametrize('key, value', [('semi_major_axis', '-6771 km'),
                                        ('duration', '-200 s'),
                                        ('time_step', '-50 s')])
def test_a_negative_orbit_size_or_span_of_time_is_refused(key, value):
    config = _minimal_inertial_config()
    config['spacecraft_history'][key] = value

    with pytest.raises(ValueError, match=f"'{key}'") as caught:
        InertialSimulator.from_config(config)

    assert '>= 0' in str(caught.value)


@pytest.mark.parametrize('key, value', [('argument_of_periapsis', '-10 deg'),
                                        ('initial_time', '-100 s')])
def test_an_angle_around_the_orbit_and_an_epoch_may_still_be_negative(key, value):
    # Both of these run backwards perfectly sensibly: an argument of periapsis
    # measured the other way round, and a run that starts before the orbit's
    # own zero of time. Bounding them at zero would refuse a legitimate file.
    config = _minimal_inertial_config()
    config['spacecraft_history'][key] = value

    simulator = InertialSimulator.from_config(config)

    written = simulator.to_config()['spacecraft_history']
    assert u.Quantity(written[key]) == u.Quantity(value)


def test_a_negative_width_keeps_extended_sources_own_better_message():
    # `width` is deliberately NOT bounded at the configuration layer:
    # `ExtendedSource` demands a *strictly* positive one and says what to do
    # instead, which is more use than "must be >= 0", and the configuration
    # layer already puts the block's name in front of it.
    block = {'type': 'ExtendedSource', 'sky_angle': '90 deg', 'width': '-5 deg',
             'flux': '2e-4 1/(cm s)', 'spectrum': _spectrum_block()}

    with pytest.raises(ValueError, match='PointSource') as caught:
        Source.from_config(block)

    assert 'strictly positive' in str(caught.value)


def test_a_negative_emissivity_keeps_earth_albedos_own_better_message():
    block = {'type': 'EarthAlbedoSource', 'emissivity': '-1e-4 1/(cm s)',
             'spectrum': _spectrum_block()}

    with pytest.raises(ValueError, match='strictly positive') as caught:
        Source.from_config(block, earth=EARTH)

    assert 'drop the source' in str(caught.value)


# ===========================================================================
# Part O -- a missing sibling file says which key named it
# ===========================================================================

def test_a_missing_scaling_file_names_the_key_and_stays_a_file_not_found():
    # Still a `FileNotFoundError` -- "the file is missing" is a different
    # problem from "the file is wrong" -- but a configuration with a dozen
    # scalings in it needs to say which one.
    with pytest.raises(FileNotFoundError) as caught:
        SourceScaling.from_config({'type': 'Tabulated', 'file': 'lc.csv'})

    assert "key 'file' = 'lc.csv' does not exist" in str(caught.value)


def test_a_missing_spacecraft_history_file_names_the_key():
    config = _minimal_inertial_config()
    config['spacecraft_history'] = 'x.ori'

    with pytest.raises(FileNotFoundError) as caught:
        InertialSimulator.from_config(config)

    assert "config.spacecraft_history = 'x.ori' does not exist" in str(caught.value)


# ===========================================================================
# Part P -- `_suggest` still answers for keys, types and choices
# ===========================================================================
#
# `_suggest` is the plain edit-distance suggestion shared by the three
# messages below -- a mistyped key, a mistyped type, a mistyped choice.
# (It once had a companion rule for expression function names; that went
# with the expression evaluator.)

def test_a_mistyped_key_is_suggested():
    with pytest.raises(ValueError, match="Did you mean 'radius'") as caught:
        Earth.from_config({'radious': '6371 km'})

    assert 'radious' in str(caught.value)


def test_a_mistyped_type_is_suggested():
    with pytest.raises(ValueError, match="Did you mean 'PointSource'"):
        Source.from_config({'type': 'PointSorce', 'sky_angle': '0 deg',
                            'flux': '1e-3 1/(cm s)',
                            'spectrum': _spectrum_block()})


def test_a_mistyped_choice_is_suggested():
    with pytest.raises(ValueError, match="Did you mean 'isotropic'"):
        Source.from_config({'type': 'EarthAlbedoSource',
                            'emissivity': '1e-4 1/(cm s)', 'law': 'isotropci',
                            'spectrum': _spectrum_block()}, earth=EARTH)


# ===========================================================================
# Part Q -- the run's Earth reaches the spacecraft history, both ways in
# ===========================================================================
#
# CONTRACT.md: "do not let a config produce a run with two different planets".
# Two wiring points were never exercised, and a run with two planets is the
# exact bug PR 5 shipped:
#
#   * a `spacecraft_history` naming a real `.ori` file. Every test until now
#     named one that does not exist, so the `earth =` argument on the open
#     was never used for anything;
#   * a `TargetedPointing` inside a full configuration. It is the one strategy
#     whose whole job is deciding when the target is behind the Earth, so it
#     is the one strategy that cannot be handed the wrong Earth without the
#     answers changing.

def _write_ori_file(path, orbit_radius_km):
    """A three-row `.ori` file, written by hand rather than by
    `SpacecraftHistory.write`, so that this test does not depend on the writer
    it is about to read with. Columns are Section 4.1's, units baked into the
    names. Two intervals, 100 s each, at a fixed radius and a slow prograde
    drift; the third row is the terminator, whose pose and uptime are never
    read."""

    path.write_text(
        "time_s,orbit_radius_km,orbit_angle_deg,attitude_deg,uptime_s\n"
        f"0,{orbit_radius_km},0,0,100\n"
        f"100,{orbit_radius_km},5,5,100\n"
        f"200,{orbit_radius_km},10,10,0\n")

    return str(path)


def test_a_history_read_from_a_file_is_given_the_runs_own_earth(tmp_path):
    config = _minimal_inertial_config()
    config['spacecraft_history'] = _write_ori_file(tmp_path / 'iss.ori', 6771.0)

    simulator = InertialSimulator.from_config(config)

    # Identity, not "a planet of the same size": the history stores the Earth
    # it was validated against and hands it on to everything downstream.
    assert simulator.spacecraft_history.earth is simulator.earth
    assert simulator.earth.radius == 6371 * u.km


def test_a_history_file_is_validated_against_the_runs_earth_not_a_default_one(tmp_path):
    # 6375 km sits deliberately between the configuration's Earth (6371 km,
    # the plan's own value) and astropy's nominal `R_earth` (6378.1 km), which
    # is what a default `Earth()` would use. This file is a legal orbit around
    # the planet the file asks for, and an impossible one -- underground --
    # around the planet a stray `Earth()` would invent, so a run that built
    # its own here would refuse a file it has no business refusing.
    assert 6371 < 6375 < R_earth.to_value(u.km)

    config = _minimal_inertial_config()
    config['spacecraft_history'] = _write_ori_file(tmp_path / 'low.ori', 6375.0)

    simulator = InertialSimulator.from_config(config)

    assert simulator.spacecraft_history.nintervals == 2


def test_a_history_read_from_a_file_is_written_back_out_as_that_path(tmp_path):
    path = _write_ori_file(tmp_path / 'iss.ori', 6771.0)
    config = _minimal_inertial_config()
    config['spacecraft_history'] = path

    simulator = InertialSimulator.from_config(config)
    out = simulator.to_config()

    assert out['spacecraft_history'] == path

    # And reading that back gives the same file again: the path is the
    # provenance, and the rows are not copied into the configuration.
    simulator2 = InertialSimulator.from_config(out)
    assert simulator2.to_config() == out


# A planet at exactly half the orbit radius. Its angular radius is then
# arcsin(6771/2 / 6771) = arcsin(0.5) = 30 deg exactly, which makes the
# occultation arithmetic below something a reader can check in their head --
# and makes it wildly different from the 70.5 deg a default `Earth()` would
# give at the same altitude, so the two cannot be confused.
TARGETED_ORBIT_RADIUS_KM = 6771.0
TARGETED_PLANET_RADIUS_KM = TARGETED_ORBIT_RADIUS_KM / 2
TARGET_SKY_ANGLE_DEG = 0.0


def _targeted_config():
    return {
        'detector': dict(DETECTOR_BLOCK),
        'earth': {'radius': f'{TARGETED_PLANET_RADIUS_KM} km'},
        'sources': [{'type': 'PointSource', 'sky_angle': '0 deg',
                     'flux': '1e-3 1/(cm s)', 'spectrum': _spectrum_block()}],
        'spacecraft_history': {
            'type': 'elliptical_orbit',
            'semi_major_axis': f'{TARGETED_ORBIT_RADIUS_KM} km',
            'eccentricity': 0.0,
            # Just under one period (5545 s for this semi-major axis), sampled
            # often enough that the target rises and sets within the run.
            'duration': '5400 s',
            'time_step': '300 s',
            'observation_strategy': {'type': 'TargetedPointing',
                                     'sky_angle': f'{TARGET_SKY_ANGLE_DEG} deg'},
        },
    }


def _attitude_expected_by_hand(orbit_angle_deg, planet_radius_km):
    """The attitude `TargetedPointing` is documented to produce, worked out
    here from the geometry rather than read off the implementation.

    The target is occulted exactly when its sky direction falls within the
    planet's angular radius `rho = arcsin(R/r)` of nadir, and nadir is
    `orbit_angle + 180 deg`; the difference is wrapped to [-180, 180) deg.
    Pointing at the target means `attitude = sky_angle`; while it is occulted
    the strategy falls back to zenith pointing, `attitude = orbit_angle`."""

    rho_deg = np.degrees(np.arcsin(planet_radius_km / TARGETED_ORBIT_RADIUS_KM))

    nadir_deg = orbit_angle_deg + 180.0
    offset_deg = (TARGET_SKY_ANGLE_DEG - nadir_deg + 180.0) % 360.0 - 180.0

    return orbit_angle_deg if abs(offset_deg) < rho_deg else TARGET_SKY_ANGLE_DEG


def test_a_targeted_pointing_in_a_full_config_occults_against_the_runs_earth():
    simulator = InertialSimulator.from_config(_targeted_config())

    orbit_angles = []
    attitudes = []

    for interval in simulator.spacecraft_history:
        # A circular orbit, so every row is at the semi-major axis.
        assert interval.orbit_radius.to_value(u.km) == pytest.approx(
            TARGETED_ORBIT_RADIUS_KM)
        orbit_angles.append(interval.orbit_angle.to_value(u.deg))
        attitudes.append(interval.attitude.to_value(u.deg))

    expected = [_attitude_expected_by_hand(angle, TARGETED_PLANET_RADIUS_KM)
                for angle in orbit_angles]

    assert attitudes == pytest.approx(expected)

    # The test only proves anything if the target both rises and sets inside
    # the run: an "always visible" sample would match any Earth at all.
    assert TARGET_SKY_ANGLE_DEG in expected
    assert any(value != TARGET_SKY_ANGLE_DEG for value in expected)

    # And it only tells the two planets apart if they disagree somewhere on
    # this sample. They do: 30 deg of angular radius against 70.5 deg.
    if_it_built_its_own_earth = [
        _attitude_expected_by_hand(angle, R_earth.to_value(u.km))
        for angle in orbit_angles]
    assert expected != if_it_built_its_own_earth


def test_a_targeted_pointing_in_a_full_config_round_trips():
    simulator = InertialSimulator.from_config(_targeted_config())
    out1 = simulator.to_config()

    assert out1['spacecraft_history']['observation_strategy']['type'] == 'TargetedPointing'
    assert u.Quantity(
        out1['spacecraft_history']['observation_strategy']['sky_angle']) == 0 * u.deg

    simulator2 = InertialSimulator.from_config(out1)
    assert simulator2.to_config() == out1


# ===========================================================================
# Part R -- a genuinely per-layer detector round-trips layer by layer
# ===========================================================================
#
# `detector_to_config` collapses a per-layer array back to the single value it
# was written as, so a configuration that gave one number reads back as one
# number rather than six copies of it. Every detector in this file until now
# gave one number, which means a collapse that threw the other five away
# would have gone unnoticed.

# The same geometry as `DETECTOR_BLOCK`, with each of the three per-layer
# keys given a different value on every layer. Layers sit at 0, 5, 10, 20, 25
# and 30 mm, so consecutive thicknesses have to average less than the gap
# between their layers (5 mm, except 10 mm in the middle) for the layers not
# to overlap: 1, 2, 3, 4, 3, 2 mm does that with room to spare.
PER_LAYER_DETECTOR_BLOCK = {
    'type': 'ToyTracker2D',
    'material': 'Ge',
    'layer_length': '16 cm',
    'layer_positions': '[0, 5, 10, 20, 25, 30] mm',
    'layer_thickness': '[1.0, 2.0, 3.0, 4.0, 3.0, 2.0] mm',
    'energy_resolution': [0.01, 0.02, 0.03, 0.04, 0.03, 0.02],
    'energy_threshold': '[20.0, 25.0, 30.0, 35.0, 30.0, 25.0] keV',
}


def test_a_per_layer_detector_keeps_every_layers_own_value():
    detector = detector_from_config(PER_LAYER_DETECTOR_BLOCK)

    out = detector_to_config(detector)

    # Layer by layer, and in order: a detector written back out with only its
    # first layer's numbers would be a different instrument.
    assert _parse_quantity(out['layer_thickness']).to_value(u.mm) == pytest.approx(
        [1.0, 2.0, 3.0, 4.0, 3.0, 2.0])
    assert out['energy_resolution'] == pytest.approx(
        [0.01, 0.02, 0.03, 0.04, 0.03, 0.02])
    assert _parse_quantity(out['energy_threshold']).to_value(u.keV) == pytest.approx(
        [20.0, 25.0, 30.0, 35.0, 30.0, 25.0])


def test_a_per_layer_detector_round_trips():
    detector = detector_from_config(PER_LAYER_DETECTOR_BLOCK)
    out1 = detector_to_config(detector)

    detector2 = detector_from_config(out1)
    out2 = detector_to_config(detector2)

    assert out1 == out2

    # And the object really did come back the same, not just its description.
    assert np.all(detector2.layer_thickness == detector.layer_thickness)
    assert np.all(detector2.energy_threshold == detector.energy_threshold)
    assert np.all(np.asarray(detector2.energy_resolution)
                  == np.asarray(detector.energy_resolution))


def test_a_per_layer_detector_round_trips_inside_a_whole_configuration():
    config = dict(_minimal_detector_frame_config(),
                  detector=dict(PER_LAYER_DETECTOR_BLOCK))

    simulator = Simulator.from_config(config)
    out1 = simulator.to_config()

    assert _parse_quantity(out1['detector']['layer_thickness']).to_value(u.mm) == (
        pytest.approx([1.0, 2.0, 3.0, 4.0, 3.0, 2.0]))

    assert Simulator.from_config(out1).to_config() == out1
