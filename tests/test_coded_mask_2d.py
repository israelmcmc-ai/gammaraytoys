import astropy.units as u


def test_effective_area_zero_outside_fully_coded_fov(mask_detector):
    # Regression test: effective_area() used to compare
    # np.abs(angle > fov) (abs of a bool) instead of np.abs(angle) > fov,
    # so it never actually zeroed out large angles.
    fov = mask_detector.fully_coded_fov

    far_angle = fov + 30 * u.deg

    assert mask_detector.effective_area(far_angle) == 0 * u.cm
    assert mask_detector.effective_area(-far_angle) == 0 * u.cm


def test_effective_area_positive_within_fully_coded_fov(mask_detector):
    assert mask_detector.effective_area(0 * u.deg) > 0 * u.cm


def test_effective_area_computed_at_angles_near_fov_edge(mask_detector):
    # Regression test: point_source_response used to crash with an
    # IndexError whenever a mask pixel's shadow projected exactly onto the
    # detector's edge, which reliably happens somewhere across the sky_axis
    # scan that response() performs.
    fov = mask_detector.fully_coded_fov

    assert mask_detector.effective_area(fov / 2) >= 0 * u.cm
    assert mask_detector.effective_area(-fov / 2) >= 0 * u.cm


def test_fully_coded_smaller_than_partially_coded_fov(mask_detector):
    assert mask_detector.fully_coded_fov < mask_detector.partially_coded_fov
