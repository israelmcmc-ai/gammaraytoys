"""Tests for the inertial <-> detector-frame transform helpers of Section 3.4.

Every expected value in this file is derived from the plan's own formulas,
from the basis vectors of Section 3.3, or from elementary geometry -- never
from running the implementation and recording what it printed.

Conventions used throughout (Section 3.2):

    lambda  `sky_angle`      CCW from inertial +X, pointing *toward* the source
    A       `attitude`       inertial angle of the detector's +y axis
    Nu      `offaxis_angle`  CCW from detector zenith (+y)
    theta   `orbit_angle`    CCW from inertial +X, Earth centre at the origin
"""

import astropy.units as u
import numpy as np
import pytest

from gammaraytoys.coordinates import Cartesian2D
from gammaraytoys.sims import (sky_angle_to_offaxis, offaxis_to_sky_angle,
                               inertial_to_detector_position,
                               inertial_to_detector_direction,
                               spacecraft_position)


def _km(x, y):
    """A `Cartesian2D` from two plain floats, in km."""
    return Cartesian2D(x * u.km, y * u.km)


# --- local, independent re-implementations of the two wraps --------------
#
# Written out here rather than imported, so that a bug in the package's own
# wrapping cannot hide behind itself.

def _wrap180(angle_deg):
    """Wrap a plain-float angle in degrees to [-180, 180)."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def _detector_basis(attitude_deg):
    """The detector's +x and +y axes in inertial components (Section 3.3):

        y_detector = ( cos A,  sin A)
        x_detector = ( sin A, -cos A)
    """
    a = np.radians(attitude_deg)
    return (np.array([np.sin(a), -np.cos(a)]),
            np.array([np.cos(a), np.sin(a)]))


# A spread of attitudes that deliberately includes values below zero and
# well past 360 deg: Section 4.4's generated histories leave `attitude` and
# `orbit_angle` unwrapped (a SpinPointing run grows without bound), so the
# transforms themselves have to own the wrapping.
ATTITUDES_DEG = [0.0, 37.5, 90.0, 180.0, 270.0, 359.9,
                 -45.0, -200.0, 400.0, 725.0, 1080.0, -730.0]

SKY_ANGLES_DEG = [0.0, 15.0, 90.0, 123.4, 180.0, 245.0, 359.0, -30.0, 380.0]


# --- sky_angle_to_offaxis ------------------------------------------------

def test_sky_angle_to_offaxis_reproduces_the_plan_sanity_checks():
    # Section 3.4 spells these two out explicitly: A = 90 deg means detector
    # +y points along inertial +Y, so a source at lambda = 90 deg is on-axis,
    # and a source at lambda = 0 deg sits at Nu = 90 deg, i.e. on the
    # detector's +x side.
    assert sky_angle_to_offaxis(90 * u.deg, 90 * u.deg).to_value(u.deg) == pytest.approx(0.0)
    assert sky_angle_to_offaxis(0 * u.deg, 90 * u.deg).to_value(u.deg) == pytest.approx(90.0)


@pytest.mark.parametrize("sky_angle_deg, attitude_deg, expected_deg", [
    # Nu = wrap180(A - lambda), computed by hand for each row.
    (0.0, 0.0, 0.0),
    (90.0, 0.0, -90.0),
    (270.0, 0.0, 90.0),
    (180.0, 0.0, -180.0),      # +180 and -180 are the same direction; the
    (-180.0, 0.0, -180.0),     # half-open range [-180, 180) picks -180.
    (200.0, 30.0, -170.0),
    (30.0, 200.0, 170.0),
    (10.0, 400.0, 30.0),       # attitude past a full turn
    (400.0, 10.0, -30.0),      # sky angle past a full turn
    (45.0, -45.0, -90.0),
])
def test_sky_angle_to_offaxis_matches_the_hand_computed_formula(sky_angle_deg,
                                                               attitude_deg,
                                                               expected_deg):
    result = sky_angle_to_offaxis(sky_angle_deg * u.deg, attitude_deg * u.deg)

    assert result.to_value(u.deg) == pytest.approx(expected_deg, abs=1e-9)


def test_sky_angle_to_offaxis_output_lies_in_the_half_open_180_range():
    for attitude_deg in ATTITUDES_DEG:
        for sky_angle_deg in SKY_ANGLES_DEG:
            nu = sky_angle_to_offaxis(sky_angle_deg * u.deg,
                                      attitude_deg * u.deg).to_value(u.deg)

            assert -180.0 <= nu < 180.0


def test_sky_angle_to_offaxis_is_blind_to_full_turns_of_the_attitude():
    # PR 2 leaves `attitude` unwrapped -- a SpinPointing history hands over
    # attitudes of many thousands of degrees -- so adding whole turns must
    # not change the answer.
    for attitude_deg in ATTITUDES_DEG:
        base = sky_angle_to_offaxis(33.0 * u.deg, attitude_deg * u.deg).to_value(u.deg)

        for turns in (-3, -1, 1, 2, 10):
            shifted = sky_angle_to_offaxis(33.0 * u.deg,
                                           (attitude_deg + 360.0 * turns) * u.deg)

            assert shifted.to_value(u.deg) == pytest.approx(base, abs=1e-8)


def test_sky_angle_to_offaxis_is_blind_to_full_turns_of_the_sky_angle():
    for sky_angle_deg in SKY_ANGLES_DEG:
        base = sky_angle_to_offaxis(sky_angle_deg * u.deg, 71.0 * u.deg).to_value(u.deg)

        for turns in (-2, 1, 5):
            shifted = sky_angle_to_offaxis((sky_angle_deg + 360.0 * turns) * u.deg,
                                           71.0 * u.deg)

            assert shifted.to_value(u.deg) == pytest.approx(base, abs=1e-8)


def test_sky_angle_to_offaxis_accepts_radians():
    # Section 3.5: Quantities in, Quantities out -- the caller's unit choice
    # must not matter.
    in_deg = sky_angle_to_offaxis(30 * u.deg, 100 * u.deg).to_value(u.deg)
    in_rad = sky_angle_to_offaxis(np.radians(30) * u.rad,
                                  np.radians(100) * u.rad).to_value(u.deg)

    assert in_rad == pytest.approx(in_deg, abs=1e-9)
    assert in_deg == pytest.approx(70.0, abs=1e-9)


# --- offaxis_to_sky_angle ------------------------------------------------

@pytest.mark.parametrize("offaxis_deg, attitude_deg, expected_deg", [
    # lambda = A - Nu, compared modulo a full turn.
    (0.0, 90.0, 90.0),
    (90.0, 90.0, 0.0),
    (-90.0, 90.0, 180.0),
    (45.0, 0.0, -45.0),
    (-170.0, 30.0, 200.0),
    (30.0, 400.0, 10.0),
])
def test_offaxis_to_sky_angle_matches_the_hand_computed_formula(offaxis_deg,
                                                               attitude_deg,
                                                               expected_deg):
    result = offaxis_to_sky_angle(offaxis_deg * u.deg, attitude_deg * u.deg)

    assert _wrap180(result.to_value(u.deg) - expected_deg) == pytest.approx(0.0, abs=1e-9)


def test_offaxis_to_sky_angle_inverts_sky_angle_to_offaxis():
    for attitude_deg in ATTITUDES_DEG:
        for sky_angle_deg in SKY_ANGLES_DEG:
            nu = sky_angle_to_offaxis(sky_angle_deg * u.deg, attitude_deg * u.deg)
            back = offaxis_to_sky_angle(nu, attitude_deg * u.deg)

            # Equal as directions on the sky, i.e. modulo a full turn.
            assert _wrap180(back.to_value(u.deg) - sky_angle_deg) == pytest.approx(
                0.0, abs=1e-8)


def test_sky_angle_to_offaxis_inverts_offaxis_to_sky_angle():
    for attitude_deg in ATTITUDES_DEG:
        for offaxis_deg in (-179.0, -90.0, -1.0, 0.0, 1.0, 90.0, 179.0):
            lam = offaxis_to_sky_angle(offaxis_deg * u.deg, attitude_deg * u.deg)
            back = sky_angle_to_offaxis(lam, attitude_deg * u.deg)

            assert back.to_value(u.deg) == pytest.approx(offaxis_deg, abs=1e-8)


# --- spacecraft_position -------------------------------------------------

@pytest.mark.parametrize("radius_km, angle_deg", [
    (6771.0, 0.0),
    (6771.0, 90.0),
    (6771.0, 180.0),
    (6771.0, 270.0),
    (7000.0, 37.0),
    (42164.0, -120.0),
    (6771.0, 370.0),      # unwrapped orbit angle, as PR 2 produces
    (6771.0, 1090.0),
])
def test_spacecraft_position_is_plain_polar_to_cartesian(radius_km, angle_deg):
    position = spacecraft_position(radius_km * u.km, angle_deg * u.deg)

    expected_x = radius_km * np.cos(np.radians(angle_deg))
    expected_y = radius_km * np.sin(np.radians(angle_deg))

    assert position.x.to_value(u.km) == pytest.approx(expected_x, abs=1e-9)
    assert position.y.to_value(u.km) == pytest.approx(expected_y, abs=1e-9)


def test_spacecraft_position_is_blind_to_full_turns_of_the_orbit_angle():
    base = spacecraft_position(6771 * u.km, 25 * u.deg)

    for turns in (-2, 1, 4):
        shifted = spacecraft_position(6771 * u.km, (25 + 360 * turns) * u.deg)

        assert shifted.x.to_value(u.km) == pytest.approx(base.x.to_value(u.km), abs=1e-6)
        assert shifted.y.to_value(u.km) == pytest.approx(base.y.to_value(u.km), abs=1e-6)


def test_spacecraft_position_norm_is_the_orbit_radius():
    for radius_km in (6771.0, 26600.0):
        for angle_deg in (0.0, 17.0, 123.0, 250.0, 359.0):
            position = spacecraft_position(radius_km * u.km, angle_deg * u.deg)

            norm = np.hypot(position.x.to_value(u.km), position.y.to_value(u.km))

            assert norm == pytest.approx(radius_km, rel=1e-12)


# --- inertial_to_detector_position ---------------------------------------

@pytest.mark.parametrize("attitude_deg, cx, cy, px, py, expected_x, expected_y", [
    # x_det = d_X sin A - d_Y cos A ; y_det = d_X cos A + d_Y sin A, with
    # d = P - C.  Each row worked out by hand.
    #
    # A = 0: detector +y is along inertial +X, detector +x along -Y.
    (0.0, 0.0, 0.0, 3.0, 4.0, -4.0, 3.0),
    # A = 90: detector frame coincides with the inertial frame.
    (90.0, 0.0, 0.0, 3.0, 4.0, 3.0, 4.0),
    # A = 180: detector +y along -X, detector +x along +Y.
    (180.0, 0.0, 0.0, 3.0, 4.0, 4.0, -3.0),
    # A = 270: detector +y along -Y, detector +x along -X.
    (270.0, 0.0, 0.0, 3.0, 4.0, -3.0, -4.0),
    # A = 90 with a displaced spacecraft: only d = P - C matters.
    (90.0, 10.0, -5.0, 13.0, -1.0, 3.0, 4.0),
    # Unwrapped attitude, same as A = 90.
    (450.0, 10.0, -5.0, 13.0, -1.0, 3.0, 4.0),
])
def test_inertial_to_detector_position_matches_the_hand_computed_formula(
        attitude_deg, cx, cy, px, py, expected_x, expected_y):
    result = inertial_to_detector_position(_km(px, py), _km(cx, cy),
                                           attitude_deg * u.deg)

    assert result.x.to_value(u.km) == pytest.approx(expected_x, abs=1e-9)
    assert result.y.to_value(u.km) == pytest.approx(expected_y, abs=1e-9)


def test_inertial_to_detector_position_maps_the_detector_axes_onto_themselves():
    # A point one unit along the detector's own +y axis (Section 3.3) must
    # land at (0, L) in the detector frame, and one along +x at (L, 0).
    length = 250.0
    centre = np.array([6771.0 * np.cos(0.3), 6771.0 * np.sin(0.3)])

    for attitude_deg in ATTITUDES_DEG:
        x_hat, y_hat = _detector_basis(attitude_deg)

        along_y = centre + length * y_hat
        along_x = centre + length * x_hat

        got_y = inertial_to_detector_position(_km(*along_y), _km(*centre),
                                              attitude_deg * u.deg)
        got_x = inertial_to_detector_position(_km(*along_x), _km(*centre),
                                              attitude_deg * u.deg)

        assert got_y.x.to_value(u.km) == pytest.approx(0.0, abs=1e-8)
        assert got_y.y.to_value(u.km) == pytest.approx(length, abs=1e-8)

        assert got_x.x.to_value(u.km) == pytest.approx(length, abs=1e-8)
        assert got_x.y.to_value(u.km) == pytest.approx(0.0, abs=1e-8)


def test_inertial_to_detector_position_is_an_isometry():
    # The transform is a rigid translation plus rotation, so it preserves
    # the distance from the spacecraft.
    centre = np.array([-3000.0, 5900.0])

    for attitude_deg in ATTITUDES_DEG:
        for point in ([0.0, 0.0], [1.0, 0.0], [6771.0, 200.0], [-40.0, 900.0]):
            point = np.asarray(point, dtype=float)

            result = inertial_to_detector_position(_km(*point), _km(*centre),
                                                   attitude_deg * u.deg)

            expected_norm = np.hypot(*(point - centre))
            got_norm = np.hypot(result.x.to_value(u.km), result.y.to_value(u.km))

            assert got_norm == pytest.approx(expected_norm, rel=1e-10, abs=1e-9)


def test_inertial_to_detector_position_round_trips():
    # The inverse is written out here from the Section 3.3 basis vectors
    # rather than taken from the package:
    #     P = C + x_det * x_hat + y_det * y_hat
    centre = np.array([4000.0, -5400.0])

    for attitude_deg in ATTITUDES_DEG:
        x_hat, y_hat = _detector_basis(attitude_deg)

        for point in ([0.0, 0.0], [4001.0, -5400.0], [-2000.0, 700.0], [12345.0, 6789.0]):
            point = np.asarray(point, dtype=float)

            det = inertial_to_detector_position(_km(*point), _km(*centre),
                                                attitude_deg * u.deg)

            back = (centre
                    + det.x.to_value(u.km) * x_hat
                    + det.y.to_value(u.km) * y_hat)

            assert back[0] == pytest.approx(point[0], abs=1e-7)
            assert back[1] == pytest.approx(point[1], abs=1e-7)


def test_inertial_to_detector_position_is_blind_to_full_turns_of_the_attitude():
    base = inertial_to_detector_position(_km(100.0, 200.0), _km(10.0, 20.0),
                                         33 * u.deg)

    for turns in (-2, 1, 3):
        shifted = inertial_to_detector_position(_km(100.0, 200.0), _km(10.0, 20.0),
                                                (33 + 360 * turns) * u.deg)

        assert shifted.x.to_value(u.km) == pytest.approx(base.x.to_value(u.km), abs=1e-6)
        assert shifted.y.to_value(u.km) == pytest.approx(base.y.to_value(u.km), abs=1e-6)


# --- inertial_to_detector_direction --------------------------------------

@pytest.mark.parametrize("direction_deg, attitude_deg, expected_deg", [
    # direction_det = direction_inertial - A + 90 deg, compared modulo a turn.
    (0.0, 90.0, 0.0),
    (90.0, 90.0, 90.0),
    (270.0, 0.0, 360.0),
    (0.0, 0.0, 90.0),
    (180.0, 45.0, 225.0),
    (10.0, 730.0, 10.0 - 730.0 + 90.0),
])
def test_inertial_to_detector_direction_matches_the_hand_computed_formula(
        direction_deg, attitude_deg, expected_deg):
    result = inertial_to_detector_direction(direction_deg * u.deg, attitude_deg * u.deg)

    assert _wrap180(result.to_value(u.deg) - expected_deg) == pytest.approx(0.0, abs=1e-9)


def test_inertial_to_detector_direction_round_trips():
    # Inverse written out from the plan's own formula:
    #     direction_inertial = direction_det + A - 90 deg
    for attitude_deg in ATTITUDES_DEG:
        for direction_deg in (0.0, 45.0, 123.4, 180.0, 270.0, 359.9, -22.0, 700.0):
            det = inertial_to_detector_direction(direction_deg * u.deg,
                                                 attitude_deg * u.deg)

            back = det.to_value(u.deg) + attitude_deg - 90.0

            assert _wrap180(back - direction_deg) == pytest.approx(0.0, abs=1e-8)


def test_inertial_to_detector_direction_is_blind_to_full_turns_of_the_attitude():
    base = inertial_to_detector_direction(17 * u.deg, 44 * u.deg).to_value(u.deg)

    for turns in (-3, 1, 6):
        shifted = inertial_to_detector_direction(
            17 * u.deg, (44 + 360 * turns) * u.deg).to_value(u.deg)

        assert _wrap180(shifted - base) == pytest.approx(0.0, abs=1e-8)


# --- the Section 3.4 consistency identity --------------------------------
#
# "a far-field photon from lambda flies at direction_inertial = lambda + 180
#  deg, which the formula sends to 270 deg - (A - lambda) = 270 deg - Nu --
#  exactly the existing detector-frame convention. If this identity fails,
#  the transform is wrong."
#
# This is the single most important assertion in PR 3: everything downstream
# is silently rotated if it fails.

def test_far_field_photon_direction_identity_is_270_minus_offaxis():
    for attitude_deg in ATTITUDES_DEG:
        for sky_angle_deg in SKY_ANGLES_DEG:
            offaxis = sky_angle_to_offaxis(sky_angle_deg * u.deg, attitude_deg * u.deg)

            direction_inertial = (sky_angle_deg + 180.0) * u.deg
            direction_det = inertial_to_detector_direction(direction_inertial,
                                                           attitude_deg * u.deg)

            expected_det = 270.0 - offaxis.to_value(u.deg)

            assert _wrap180(direction_det.to_value(u.deg) - expected_det) == pytest.approx(
                0.0, abs=1e-8)


def test_far_field_photon_direction_is_antiparallel_to_the_line_of_sight():
    # An independent route to the same identity, tying the *position*
    # transform to the *direction* transform. Put a source very far away
    # along lambda; the detector-frame unit vector pointing at it must be
    # (sin Nu, cos Nu) (Section 3.3), and the photon it emits must fly along
    # exactly the opposite unit vector.
    huge = 1e10   # km, effectively infinity next to any orbit radius

    centre = spacecraft_position(6771 * u.km, 41 * u.deg)
    cx = centre.x.to_value(u.km)
    cy = centre.y.to_value(u.km)

    for attitude_deg in ATTITUDES_DEG:
        for sky_angle_deg in SKY_ANGLES_DEG:
            lam = np.radians(sky_angle_deg)
            source = _km(cx + huge * np.cos(lam), cy + huge * np.sin(lam))

            to_source = inertial_to_detector_position(source, centre,
                                                      attitude_deg * u.deg)

            sx = to_source.x.to_value(u.km)
            sy = to_source.y.to_value(u.km)
            norm = np.hypot(sx, sy)
            los = np.array([sx / norm, sy / norm])

            nu = np.radians(sky_angle_to_offaxis(sky_angle_deg * u.deg,
                                                 attitude_deg * u.deg).to_value(u.deg))

            # The source direction in the detector frame is (sin Nu, cos Nu).
            assert los[0] == pytest.approx(np.sin(nu), abs=1e-6)
            assert los[1] == pytest.approx(np.cos(nu), abs=1e-6)

            direction_det = inertial_to_detector_direction(
                (sky_angle_deg + 180.0) * u.deg, attitude_deg * u.deg).to_value(u.deg)

            flight = np.array([np.cos(np.radians(direction_det)),
                               np.sin(np.radians(direction_det))])

            assert flight[0] == pytest.approx(-los[0], abs=1e-6)
            assert flight[1] == pytest.approx(-los[1], abs=1e-6)


def test_transform_helpers_are_exported_from_the_sims_package():
    import gammaraytoys.sims as sims

    for name in ('sky_angle_to_offaxis', 'offaxis_to_sky_angle',
                 'inertial_to_detector_position', 'inertial_to_detector_direction',
                 'spacecraft_position'):
        assert hasattr(sims, name), f"{name} is not exported from gammaraytoys.sims"
