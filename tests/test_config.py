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
    ConstantScaling, Earth, EarthAlbedoSource, ExtendedSource, FunctionScaling,
    InertialPointing, InertialSimulator, IsotropicSource, MonoenergeticSpectrum,
    MultiComponentSpectrum, NadirPointing, NearPointSource, PointSource,
    PowerLawSpectrum, SimpleTraditionalReconstructor, SpacecraftHistory,
    SpinPointing, Simulator, TabulatedScaling, TargetedPointing, ZenithPointing,
    detector_from_config, detector_to_config, earth_from_config, earth_to_config,
    load_config, observation_strategy_from_config, observation_strategy_to_config,
    reconstructor_from_config, reconstructor_to_config, scaling_from_config,
    scaling_to_config, source_from_config, source_to_config, spectrum_from_config,
    spectrum_to_config,
)


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
    spectrum = spectrum_from_config(block)

    assert isinstance(spectrum, MonoenergeticSpectrum)
    assert spectrum.energy == 511 * u.keV

    out = spectrum_to_config(spectrum)
    assert out['type'] == 'Monoenergetic'
    assert u.Quantity(out['energy']) == 511 * u.keV

    spectrum2 = spectrum_from_config(out)
    assert spectrum_to_config(spectrum2) == out


def test_monoenergetic_spectrum_accepts_full_class_name_alias():
    spectrum = spectrum_from_config({'type': 'MonoenergeticSpectrum', 'energy': '1 MeV'})
    assert isinstance(spectrum, MonoenergeticSpectrum)
    # The short name is always what is written back, regardless of which
    # spelling was read (module docstring, "Round trips").
    assert spectrum_to_config(spectrum)['type'] == 'Monoenergetic'


def test_powerlaw_spectrum_round_trips():
    block = {'type': 'PowerLaw', 'index': -2, 'min_energy': '0.2 MeV', 'max_energy': '10 MeV'}
    spectrum = spectrum_from_config(block)

    assert isinstance(spectrum, PowerLawSpectrum)
    assert spectrum.index == -2
    assert spectrum.min_energy == 0.2 * u.MeV
    assert spectrum.max_energy == 10 * u.MeV

    out = spectrum_to_config(spectrum)
    assert out['type'] == 'PowerLaw'
    assert out['index'] == -2.0
    assert u.Quantity(out['min_energy']) == 0.2 * u.MeV
    assert u.Quantity(out['max_energy']) == 10 * u.MeV

    spectrum2 = spectrum_from_config(out)
    assert spectrum_to_config(spectrum2) == out


def test_powerlaw_spectrum_rejects_nonpositive_min_energy():
    block = {'type': 'PowerLaw', 'index': -2, 'min_energy': '0 MeV', 'max_energy': '10 MeV'}
    with pytest.raises(ValueError, match='min_energy'):
        spectrum_from_config(block)


def test_powerlaw_spectrum_rejects_max_not_above_min():
    block = {'type': 'PowerLaw', 'index': -2, 'min_energy': '5 MeV', 'max_energy': '5 MeV'}
    with pytest.raises(ValueError, match='max_energy'):
        spectrum_from_config(block)


def test_multicomponent_spectrum_round_trips_with_equal_weights_omitted():
    block = {'type': 'MultiComponent',
             'components': [_spectrum_block('511 keV'), _spectrum_block('1275 keV')]}
    spectrum = spectrum_from_config(block)

    assert isinstance(spectrum, MultiComponentSpectrum)
    assert spectrum.ncomponents == 2
    np.testing.assert_allclose(spectrum.weights, [0.5, 0.5])

    out = spectrum_to_config(spectrum)
    assert 'weights' not in out  # equal weights are the default: omitted

    spectrum2 = spectrum_from_config(out)
    assert spectrum_to_config(spectrum2) == out


def test_multicomponent_spectrum_round_trips_with_unequal_weights_kept():
    block = {'type': 'MultiComponent',
             'components': [_spectrum_block('511 keV'), _spectrum_block('1275 keV')],
             'weights': [1, 3]}
    spectrum = spectrum_from_config(block)

    np.testing.assert_allclose(spectrum.weights, [0.25, 0.75])

    out = spectrum_to_config(spectrum)
    assert out['weights'] == pytest.approx([0.25, 0.75])

    spectrum2 = spectrum_from_config(out)
    assert spectrum_to_config(spectrum2) == out
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
        spectrum_from_config(block)


def test_spectrum_unknown_type_raises():
    with pytest.raises(ValueError, match='unknown spectrum type'):
        spectrum_from_config({'type': 'Gaussian', 'energy': '511 keV'})


def test_spectrum_unknown_key_raises():
    block = {'type': 'Monoenergetic', 'energy': '511 keV', 'bogus': 1}
    with pytest.raises(ValueError, match='bogus'):
        spectrum_from_config(block)


# ===========================================================================
# Part B -- scalings round-trip (CONTRACT.md inventory: 3 scaling types)
# ===========================================================================

def test_constant_scaling_round_trips():
    scaling = scaling_from_config({'type': 'Constant', 'scale': 2.5})
    assert isinstance(scaling, ConstantScaling)
    assert scaling.scale == 2.5
    assert scaling_to_config(scaling) == {'type': 'Constant', 'scale': 2.5}


def test_constant_scaling_default_scale_is_one():
    scaling = scaling_from_config({'type': 'Constant'})
    assert scaling.scale == 1.0


def test_tabulated_scaling_inline_round_trips():
    block = {'type': 'Tabulated', 'time': '[0, 100, 200] s', 'scale': [1.0, 2.0, 0.5]}
    scaling = scaling_from_config(block)

    assert isinstance(scaling, TabulatedScaling)
    np.testing.assert_allclose(scaling.time.to_value(u.s), [0, 100, 200])
    np.testing.assert_allclose(scaling.scale, [1.0, 2.0, 0.5])

    out = scaling_to_config(scaling)
    scaling2 = scaling_from_config(out)
    assert scaling_to_config(scaling2) == out


def test_tabulated_scaling_single_row_round_trips():
    # `_number`'s scalar `scale` is wrapped into a one-element list before
    # `TabulatedScaling` sees it (config.py's `scaling_from_config`); pin
    # that this actually works for the smallest legal table.
    block = {'type': 'Tabulated', 'time': '[0] s', 'scale': 7.0}
    scaling = scaling_from_config(block)
    assert scaling(0 * u.s) == 7.0
    assert scaling(1e9 * u.s) == 7.0


def test_tabulated_scaling_file_round_trips_and_is_written_inline(tmp_path):
    path = tmp_path / 'lightcurve.csv'
    path.write_text('time_s,scale\n0,1.0\n100,2.0\n200,0.5\n')

    scaling = scaling_from_config({'type': 'Tabulated', 'file': str(path)})
    assert isinstance(scaling, TabulatedScaling)

    out = scaling_to_config(scaling)
    assert 'file' not in out
    assert 'time' in out and 'scale' in out

    scaling2 = scaling_from_config(out)
    assert scaling_to_config(scaling2) == out


def test_tabulated_scaling_file_and_inline_together_raises(tmp_path):
    path = tmp_path / 'lightcurve.csv'
    path.write_text('time_s,scale\n0,1.0\n100,2.0\n')
    block = {'type': 'Tabulated', 'file': str(path), 'time': '[0, 100] s', 'scale': [1.0, 2.0]}
    with pytest.raises(ValueError, match='not both'):
        scaling_from_config(block)


def test_tabulated_scaling_neither_file_nor_inline_raises():
    with pytest.raises(ValueError, match='needs either'):
        scaling_from_config({'type': 'Tabulated'})


def test_function_scaling_round_trips_expression_exactly():
    expression = '1 + 0.5*sin(2*pi*t/5400)'
    scaling = scaling_from_config({'type': 'Function', 'expression': expression})

    assert isinstance(scaling, FunctionScaling)
    assert scaling_to_config(scaling) == {'type': 'Function', 'expression': expression}


def test_function_scaling_from_config_rejects_dangerous_expression():
    # `scaling_from_config` must forward `TimeExpression`'s rejection rather
    # than swallowing or replacing it; the expression evaluator's own
    # defence is pinned exhaustively in `test_config_expression.py`.
    with pytest.raises(ValueError):
        scaling_from_config({'type': 'Function',
                              'expression': "__import__('os').system('id')"})


def test_scaling_unknown_type_raises():
    with pytest.raises(ValueError, match='unknown scaling type'):
        scaling_from_config({'type': 'Sinusoid', 'scale': 1.0})


def test_scaling_unknown_key_raises():
    with pytest.raises(ValueError, match='bogus'):
        scaling_from_config({'type': 'Constant', 'scale': 1.0, 'bogus': True})


# ===========================================================================
# Part C -- sources round-trip (CONTRACT.md inventory: 5 source types,
# PointSource counted twice for its two mutually exclusive forms)
# ===========================================================================

def test_pointsource_offaxis_form_round_trips():
    block = {'name': 'crab', 'type': 'PointSource', 'offaxis_angle': '30 deg',
             'flux': '1e-3 1/(cm s)', 'spectrum': _spectrum_block()}
    source = source_from_config(block)

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

    out = source_to_config(source, name='crab')
    assert out['name'] == 'crab'
    assert out['type'] == 'PointSource'
    assert 'offaxis_angle' in out and 'sky_angle' not in out

    source2 = source_from_config(out)
    assert source_to_config(source2, name='crab') == out


def test_pointsource_sky_angle_form_round_trips():
    block = {'type': 'PointSource', 'sky_angle': '45 deg', 'flux': '1e-3 1/(cm s)',
             'spectrum': _spectrum_block()}
    source = source_from_config(block)

    assert source.offaxis_angle is None
    assert source.sky_angle == 45 * u.deg

    out = source_to_config(source)
    assert 'sky_angle' in out and 'offaxis_angle' not in out

    source2 = source_from_config(out)
    assert source_to_config(source2) == out


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
    source = source_from_config(block)

    flux = source.flux()
    assert flux.to_value('1/(cm s)') == pytest.approx(9e-4)

    # `source_to_config` writes the *resolved* flux, not flux_pivot/pivot_energy
    # (module docstring, "Round trips").
    out = source_to_config(source)
    assert 'flux' in out and 'flux_pivot' not in out
    assert 'pivot_energy' not in out

    source2 = source_from_config(out)
    assert source2.flux().to_value('1/(cm s)') == pytest.approx(9e-4)
    assert source_to_config(source2) == out


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
        source_from_config(block)


def test_pointsource_both_forms_raises():
    block = {'type': 'PointSource', 'offaxis_angle': '10 deg', 'sky_angle': '20 deg',
             'flux': '1e-3 1/(cm s)', 'spectrum': _spectrum_block()}
    with pytest.raises(ValueError, match='exactly one'):
        source_from_config(block)


def test_pointsource_neither_form_raises():
    block = {'type': 'PointSource', 'flux': '1e-3 1/(cm s)', 'spectrum': _spectrum_block()}
    with pytest.raises(ValueError, match='exactly one'):
        source_from_config(block)


def test_isotropic_source_round_trips():
    block = {'type': 'IsotropicSource', 'flux': '2e-4 1/(cm s)', 'spectrum': _spectrum_block()}
    source = source_from_config(block)
    assert isinstance(source, IsotropicSource)

    out = source_to_config(source)
    source2 = source_from_config(out)
    assert source_to_config(source2) == out


def test_near_point_source_round_trips():
    block = {'type': 'NearPointSource', 'position': {'x': '0 cm', 'y': '1 cm'},
             'rate': '5 1/s', 'spectrum': _spectrum_block()}
    source = source_from_config(block)

    assert isinstance(source, NearPointSource)
    assert source.position.x == 0 * u.cm
    assert source.position.y == 1 * u.cm
    assert source.rate == 5 * u.Hz

    out = source_to_config(source)
    source2 = source_from_config(out)
    assert source_to_config(source2) == out


def test_near_point_source_missing_position_raises():
    block = {'type': 'NearPointSource', 'rate': '5 1/s', 'spectrum': _spectrum_block()}
    with pytest.raises(ValueError, match='position'):
        source_from_config(block)


def test_extended_source_round_trips():
    block = {'type': 'ExtendedSource', 'sky_angle': '10 deg', 'width': '5 deg',
             'flux': '1e-3 1/(cm s)', 'spectrum': _spectrum_block()}
    source = source_from_config(block)

    assert isinstance(source, ExtendedSource)
    assert source.sky_angle == 10 * u.deg
    assert source.width == 5 * u.deg

    out = source_to_config(source)
    source2 = source_from_config(out)
    assert source_to_config(source2) == out


def test_earth_albedo_source_lambertian_round_trips():
    block = {'type': 'EarthAlbedoSource', 'emissivity': '1e-4 1/(cm s)',
             'spectrum': _spectrum_block()}
    source = source_from_config(block, earth=EARTH)

    assert isinstance(source, EarthAlbedoSource)
    assert source.law == 'lambertian'
    assert source.earth is EARTH

    out = source_to_config(source)
    assert 'law' not in out  # default, omitted (module docstring)

    source2 = source_from_config(out, earth=EARTH)
    assert source_to_config(source2) == out


def test_earth_albedo_source_isotropic_law_round_trips():
    block = {'type': 'EarthAlbedoSource', 'emissivity': '1e-4 1/(cm s)',
             'law': 'isotropic', 'spectrum': _spectrum_block()}
    source = source_from_config(block, earth=EARTH)
    assert source.law == 'isotropic'
    assert source_to_config(source)['law'] == 'isotropic'


def test_earth_albedo_source_accepts_emission_law_alias():
    # `emission_law` is the plan's Section 7 sketch spelling; `law` is the
    # constructor argument (module docstring).
    block = {'type': 'EarthAlbedoSource', 'emissivity': '1e-4 1/(cm s)',
             'emission_law': 'isotropic', 'spectrum': _spectrum_block()}
    source = source_from_config(block, earth=EARTH)
    assert source.law == 'isotropic'


def test_earth_albedo_source_law_and_emission_law_together_raises():
    block = {'type': 'EarthAlbedoSource', 'emissivity': '1e-4 1/(cm s)',
             'law': 'isotropic', 'emission_law': 'lambertian', 'spectrum': _spectrum_block()}
    with pytest.raises(ValueError, match='not both'):
        source_from_config(block, earth=EARTH)


def test_earth_albedo_source_rejects_bad_law():
    block = {'type': 'EarthAlbedoSource', 'emissivity': '1e-4 1/(cm s)',
             'law': 'bogus', 'spectrum': _spectrum_block()}
    with pytest.raises(ValueError):
        source_from_config(block, earth=EARTH)


def test_source_chirality_round_trips():
    block = {'type': 'IsotropicSource', 'flux': '1e-3 1/(cm s)', 'spectrum': _spectrum_block(),
             'chirality': 1, 'chirality_degree': 0.7}
    source = source_from_config(block)

    assert source.chirality == 1
    assert source.chirality_degree == 0.7

    out = source_to_config(source)
    assert out['chirality'] == 1
    assert out['chirality_degree'] == 0.7

    source2 = source_from_config(out)
    assert source_to_config(source2) == out


def test_source_scaling_is_written_back_out_by_to_config():
    # A round trip that only compares two `to_config` writes to each other is
    # blind to a field both writes leave out, so this one names the block it
    # expects. A source that silently lost its scaling would still round-trip
    # "cleanly" and then run at a constant rate -- exactly the kind of quiet
    # wrong answer this schema is meant to prevent.
    expression = '1 + 0.5*sin(2*pi*t/5400)'
    block = {'type': 'IsotropicSource', 'flux': '1e-3 1/(cm s)',
             'spectrum': _spectrum_block(),
             'scaling': {'type': 'Function', 'expression': expression}}
    source = source_from_config(block)

    assert isinstance(source.scaling, FunctionScaling)

    out = source_to_config(source)
    assert out['scaling'] == {'type': 'Function', 'expression': expression}


def test_a_non_default_constant_scaling_is_written_back_out_too():
    block = {'type': 'IsotropicSource', 'flux': '1e-3 1/(cm s)',
             'spectrum': _spectrum_block(),
             'scaling': {'type': 'Constant', 'scale': 0.25}}
    source = source_from_config(block)

    assert source_to_config(source)['scaling'] == {'type': 'Constant', 'scale': 0.25}


def test_a_default_scaling_is_the_only_one_omitted_from_to_config():
    # The other side of the same decision: `ConstantScaling(1.0)` is what a
    # source with no `scaling` key gets, so writing it back out would be
    # noise. Anything else must survive.
    source = source_from_config(_minimal_source_block())

    assert isinstance(source.scaling, ConstantScaling)
    assert source.scaling.scale == 1.0
    assert 'scaling' not in source_to_config(source)


def test_source_default_chirality_is_omitted_from_to_config():
    source = source_from_config(_minimal_source_block())
    out = source_to_config(source)
    assert 'chirality' not in out
    assert 'chirality_degree' not in out


def test_source_invalid_chirality_raises():
    block = dict(_minimal_source_block())
    block['chirality'] = 2
    with pytest.raises(ValueError, match='chirality'):
        source_from_config(block)


def test_source_unknown_type_raises():
    with pytest.raises(ValueError, match='unknown source type'):
        source_from_config({'type': 'Blazar', 'spectrum': _spectrum_block()})


def test_source_unknown_key_raises():
    block = dict(_minimal_source_block())
    block['bogus'] = 1
    with pytest.raises(ValueError, match='bogus'):
        source_from_config(block)


def test_source_missing_spectrum_raises():
    with pytest.raises(ValueError, match='spectrum'):
        source_from_config({'type': 'IsotropicSource', 'flux': '1e-3 1/(cm s)'})


# ===========================================================================
# Part D -- observation strategies round-trip (CONTRACT.md inventory: 5)
# ===========================================================================

def test_zenith_pointing_round_trips():
    strategy = observation_strategy_from_config({'type': 'ZenithPointing'})
    assert isinstance(strategy, ZenithPointing)
    assert observation_strategy_to_config(strategy) == {'type': 'ZenithPointing'}


def test_nadir_pointing_round_trips():
    strategy = observation_strategy_from_config({'type': 'NadirPointing'})
    assert isinstance(strategy, NadirPointing)
    assert observation_strategy_to_config(strategy) == {'type': 'NadirPointing'}


def test_inertial_pointing_round_trips():
    strategy = observation_strategy_from_config({'type': 'InertialPointing', 'attitude': '30 deg'})
    assert isinstance(strategy, InertialPointing)
    assert strategy.attitude == 30 * u.deg

    out = observation_strategy_to_config(strategy)
    assert u.Quantity(out['attitude']) == 30 * u.deg


def test_spin_pointing_round_trips_default_initial_attitude_omitted():
    strategy = observation_strategy_from_config({'type': 'SpinPointing', 'rate': '0.1 deg/s'})
    assert isinstance(strategy, SpinPointing)
    assert strategy.initial_attitude == 0 * u.deg

    out = observation_strategy_to_config(strategy)
    assert 'initial_attitude' not in out


def test_spin_pointing_round_trips_explicit_initial_attitude():
    strategy = observation_strategy_from_config(
        {'type': 'SpinPointing', 'rate': '0.1 deg/s', 'initial_attitude': '15 deg'})
    out = observation_strategy_to_config(strategy)
    assert u.Quantity(out['initial_attitude']) == 15 * u.deg


def test_targeted_pointing_round_trips_with_earth():
    strategy = observation_strategy_from_config(
        {'type': 'TargetedPointing', 'sky_angle': '45 deg'}, earth=EARTH)
    assert isinstance(strategy, TargetedPointing)
    assert strategy.sky_angle == 45 * u.deg
    assert strategy.earth is EARTH

    out = observation_strategy_to_config(strategy)
    assert u.Quantity(out['sky_angle']) == 45 * u.deg


def test_targeted_pointing_without_earth_raises():
    with pytest.raises(ValueError, match='Earth'):
        observation_strategy_from_config({'type': 'TargetedPointing', 'sky_angle': '45 deg'})


def test_observation_strategy_unknown_type_raises():
    with pytest.raises(ValueError, match='unknown observation strategy type'):
        observation_strategy_from_config({'type': 'SlewPointing'})


def test_observation_strategy_unknown_key_raises():
    with pytest.raises(ValueError, match='bogus'):
        observation_strategy_from_config({'type': 'ZenithPointing', 'bogus': 1})


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
    earth = earth_from_config(EARTH_BLOCK)
    assert earth.radius == 6371 * u.km

    out = earth_to_config(earth)
    assert u.Quantity(out['radius']) == 6371 * u.km

    earth2 = earth_from_config(out)
    assert earth_to_config(earth2) == out


def test_earth_default_radius_is_astropy_r_earth_not_the_plans_6371():
    # module docstring: an empty `earth` block gives astropy's nominal
    # R_earth (6378.1 km), *not* the 6371 km the plan's own sketch uses.
    earth = earth_from_config({})
    assert earth.radius == R_earth.to(u.km)
    assert earth.radius != 6371 * u.km


def test_earth_unknown_key_raises():
    with pytest.raises(ValueError, match='bogus'):
        earth_from_config({'radius': '6371 km', 'bogus': 1})


def test_earth_nonpositive_radius_raises():
    with pytest.raises(ValueError, match='positive'):
        earth_from_config({'radius': '-1 km'})


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
        source_from_config(block)


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
         'scaling': {'type': 'Function', 'expression': '1 + 0.5*sin(2*pi*t/5400)'}},
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
         'scaling': {'type': 'Function', 'expression': '1 + 0.5*sin(2*pi*t/5400)'}},
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
    from gammaraytoys.sims import config as config_module

    real_spectrum_from_config = config_module.spectrum_from_config

    def spectrum_from_config_that_draws(*args, **kwargs):
        np.random.uniform(size=3)
        return real_spectrum_from_config(*args, **kwargs)

    monkeypatch.setattr(config_module, 'spectrum_from_config',
                        spectrum_from_config_that_draws)

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
