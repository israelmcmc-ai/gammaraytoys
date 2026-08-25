#!/usr/bin/env python
"""
Regenerate the cached instrument-response files in this directory.

These files were generated with the simulator before the interaction-probability
bug was fixed (see the "Fix inverted interaction probability in
ToyTracker2D.simulate_event" commit) and are therefore stale relative to the
current physics. This script reproduces the exact generation logic that was
in tutorials 03, 06 and 07 for each cached file, so they can be regenerated
without hand-copying code out of the notebooks.

Work is parallelized across processes (one independent simulation chunk/bin
per worker; results are combined in the parent), since each bin/chunk is
statistically independent. Approximate runtimes on 4 cores, post-fix
(Beer-Lambert correction, Compton Doppler sampling fix, and simulate_event
loop-invariant hoisting -- see git history for this file), on-axis Ge tracker:
  - energy_onaxis            ~5 min    (ntrig=10000 split across workers)
  - energy_relative_onaxis   ~35-45 min (25 photon-energy bins, ntrig=4000 each)
  - imaging_chiral_relative  ~2-2.5 h  (17 offaxis-angle bins, ntrig=20000 each)

(Original single-core, pre-optimization baseline was ~30 min / ~4-5 h /
~14-16 h. These are estimates extrapolated from per-event profiling and a
partial timed run, not a full timed run of each target -- expect some
variance, particularly for imaging_chiral_relative since trigger efficiency
varies by off-axis angle. Scale roughly with min(bins, workers) additional
speedup from more cores, up to the number of independent bins.)

The larger ones are still impractical for an interactive session; run them
with nohup/tmux/a background job, e.g.:

    nohup python regenerate_responses.py imaging_chiral_relative > imaging.log 2>&1 &

Pass --ntrig to override the per-bin/per-run trigger count (e.g. for a fast,
noisier smoke test before committing to a full run), and --workers to
override the process count (default: all available cores).
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import astropy.units as u
from histpy import Histogram, Axis

from gammaraytoys import ToyTracker2D
from gammaraytoys.detectors import (
    Simulator,
    SimpleTraditionalReconstructor,
    PointSource,
    PowerLawSpectrum,
    MonoenergeticSpectrum,
)

DEFAULT_WORKERS = os.cpu_count() or 1


def _traditional_detector():
    return ToyTracker2D(
        material="Ge",
        layer_length=10 * u.m,
        layer_positions=np.append(300, np.arange(0, 10, 1)) * u.cm,
        layer_thickness=1 * u.cm,
        energy_resolution=0.03,
        energy_threshold=20 * u.keV,
    )


def _split_evenly(total, n):
    """Split `total` into `n` near-equal positive integer chunks summing to total."""

    n = max(1, min(n, total))
    base, extra = divmod(total, n)
    return [base + 1 if i < extra else base for i in range(n)]


def _worker_seeds(n):
    """Independent, well-separated integer seeds for n worker processes."""

    return [int(s.generate_state(1)[0]) for s in np.random.SeedSequence().spawn(n)]


# --- energy_onaxis ---------------------------------------------------------


def _worker_energy_onaxis(args):
    ntrig_chunk, seed = args
    np.random.seed(seed)

    det = _traditional_detector()
    source = PointSource(
        offaxis_angle=0 * u.deg,
        spectrum=PowerLawSpectrum(index=-1, min_energy=200 * u.keV, max_energy=50 * u.MeV),
        flux_pivot=1e2 / u.erg / u.cm / u.s,
        pivot_energy=1 * u.MeV,
    )
    reco = SimpleTraditionalReconstructor()
    sims = Simulator(detector=det, sources=source, reconstructor=reco)
    sims.photon_energy_axis = np.geomspace(200 * u.keV, 50 * u.MeV, 31)
    sims.measured_energy_axis = np.geomspace(100 * u.keV, 60 * u.MeV, 11)

    return sims.run_binned(ntrig=ntrig_chunk, axes="Em", photon_axes="Ei")


def regenerate_energy_onaxis(
    ntrig=10000, outfile="response_energy_onaxis_traddet.h5", workers=DEFAULT_WORKERS
):
    """
    On-axis spectral response, R(Em; Ei), for a power-law source (index -1).
    Matches docs/tutorials/03-instrument_response.ipynb.
    """

    det = _traditional_detector()

    chunks = _split_evenly(ntrig, workers)
    seeds = _worker_seeds(len(chunks))

    h_data_rsp = None
    h_nsim_rsp = None
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(chunks)) as ex:
        futures = [ex.submit(_worker_energy_onaxis, args) for args in zip(chunks, seeds)]
        for n, fut in enumerate(as_completed(futures), 1):
            hd, hn = fut.result()
            h_data_rsp = hd if h_data_rsp is None else h_data_rsp + hd
            h_nsim_rsp = hn if h_nsim_rsp is None else h_nsim_rsp + hn
            print(f"{n}/{len(chunks)} worker chunks done ({time.time() - t0:.0f}s elapsed)")

    response = h_data_rsp * (det.throwing_plane_size / h_nsim_rsp.contents)[:, None]
    response.write(outfile, overwrite=True)
    print(f"Wrote {outfile}")


# --- energy_relative_onaxis -------------------------------------------------


def _worker_energy_relative_bin(args):
    nEi, Ei, ntrig, seed = args
    np.random.seed(seed)

    det = _traditional_detector()
    source = PointSource(offaxis_angle=0 * u.deg, spectrum=MonoenergeticSpectrum(energy=Ei))
    reco = SimpleTraditionalReconstructor()
    sims = Simulator(detector=det, sources=source, reconstructor=reco)

    frac_errors = []
    nsim = 0
    for sim_event, reco_event in sims.run_events(ntrig=ntrig):
        nsim += 1
        if reco_event.triggered:
            frac_errors.append(((reco_event.energy - Ei) / Ei).to_value(""))

    return nEi, nsim, np.array(frac_errors)


def regenerate_energy_relative_onaxis(
    ntrig=4000, outfile="response_energy_relative_onaxis_traddet.h5", workers=DEFAULT_WORKERS
):
    """
    On-axis effective area Aeff(Ei) and fractional energy-error PDF, binned
    per true photon energy. Matches docs/tutorials/06-delta_response_spectrum.ipynb.

    Parallelized one photon-energy bin per worker (bins are independent).
    """

    det = _traditional_detector()

    Ei_axis = Axis(np.geomspace(0.2, 50, 26) * u.MeV, label="Ei", scale="log")
    EmFracError_axis = Axis(
        np.append(np.linspace(-0.5, -0.1, 16)[:-1], np.linspace(-0.1, 0.1, 16)),
        label="EmDeltaFrac",
    )

    h_data_delta = Histogram([Ei_axis, EmFracError_axis])
    h_nsim = Histogram(Ei_axis)

    nbins = Ei_axis.nbins
    seeds = _worker_seeds(nbins)
    tasks = [
        (nEi, Ei, ntrig, seed)
        for nEi, (Ei, seed) in enumerate(zip(Ei_axis.centers, seeds))
    ]

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_worker_energy_relative_bin, t) for t in tasks]
        for n, fut in enumerate(as_completed(futures), 1):
            nEi, nsim, frac_errors = fut.result()
            h_nsim[nEi] = nsim
            if frac_errors.size:
                Ei = Ei_axis.centers[nEi]
                h_data_delta.fill(Ei, frac_errors)
            print(f"{n}/{nbins} bins done ({time.time() - t0:.0f}s elapsed)")

    h_ntrig = h_data_delta.project("Ei")
    Aeff = (h_ntrig / h_nsim) * det.throwing_plane_size

    energy_dist = h_data_delta / h_ntrig.contents[:, None]
    phase_space = energy_dist.axes[0].centers[:, None] * energy_dist.axes[1].widths[None, :]
    energy_pdf = energy_dist / phase_space

    Aeff.write(outfile, overwrite=True, name="Aeff")
    energy_pdf.write(outfile, overwrite=True, name="PDF")
    print(f"Wrote {outfile}")


# --- imaging_chiral_relative -------------------------------------------------


def _phi_arm_phase_space(phi1, phi2, arm1, arm2):
    """Integrate phase space, accounting for phi+arm being limited to [0, pi]."""

    phi1 = phi1.to_value(u.rad)
    phi2 = phi2.to_value(u.rad)
    arm1 = arm1.to_value(u.rad)
    arm2 = arm2.to_value(u.rad)

    phi1, phi2, arm1, arm2 = np.broadcast_arrays(phi1, phi2, arm1, arm2)

    arm1 = np.choose((arm1 < -phi2) & (-phi2 < arm2), [arm1, -phi2])
    arm2 = np.choose((arm1 < np.pi - phi1) & (np.pi - phi1 < arm2), [arm2, np.pi - phi1])

    phi1 = np.choose((phi1 < -arm2) & (-arm2 < phi2), [phi1, -arm2])
    phi2 = np.choose((phi1 < np.pi - arm1) & (np.pi - arm1 < phi2), [phi2, np.pi - arm1])

    integral_rect = (arm2 - arm1) * (phi2 - phi1)

    phil = np.maximum(-arm2, phi1)
    phih = np.minimum(-arm1, phi2)
    arml = -phih
    armh = -phil
    unphys_lowerleft_integral = (armh - arml) * (phih - phil) / 2
    unphys_lowerleft_integral *= (phil + arm1) < 0
    integral = integral_rect - unphys_lowerleft_integral

    phil = np.maximum(np.pi - arm2, phi1)
    phih = np.minimum(np.pi - arm1, phi2)
    arml = np.pi - phih
    armh = np.pi - phil
    unphys_upperright_integral = (armh - arml) * (phih - phil) / 2
    unphys_upperright_integral *= (phih + arm2) > np.pi
    integral -= unphys_upperright_integral

    phase_units = u.rad * u.rad
    integral = integral * phase_units

    fully_phys = (phi1 + arm1 >= 0) & (phi2 + arm2 <= np.pi)
    fully_unphys = (phi2 + arm2 <= 0) | (phi1 + arm1 >= np.pi)
    integral_full = integral_rect * phase_units

    if integral.ndim == 0:
        if fully_phys:
            return integral_full
        if fully_unphys:
            return 0 * phase_units
    else:
        integral[fully_phys] = integral_full[fully_phys]
        integral[fully_unphys] = 0 * phase_units

    return integral


def _worker_imaging_bin(args):
    nNu, nu, ntrig, seed = args
    np.random.seed(seed)

    det = _traditional_detector()
    source = PointSource(offaxis_angle=nu, spectrum=MonoenergeticSpectrum(energy=1 * u.MeV))
    reco = SimpleTraditionalReconstructor()
    sims = Simulator(detector=det, sources=source, reconstructor=reco)

    k_all = []
    k_trig, phi_trig, arm_trig, zeta_trig = [], [], [], []

    for sim_event, reco_event in sims.run_events(ntrig=ntrig):
        k_all.append(sim_event.chirality)

        if reco_event.triggered:
            psi = reco_event.psi
            phi = reco_event.phi

            if psi > nu + 180 * u.deg:
                psi -= 360 * u.deg
            if psi < nu - 180 * u.deg:
                psi += 360 * u.deg

            zeta = -1 if (psi - nu) > 0 else 1
            arm = np.abs(psi - nu) - phi

            k_trig.append(sim_event.chirality)
            phi_trig.append(phi)
            arm_trig.append(arm)
            zeta_trig.append(zeta)

    return nNu, nu, np.array(k_all), np.array(k_trig), u.Quantity(phi_trig), u.Quantity(arm_trig), np.array(zeta_trig)


def regenerate_imaging_chiral_relative(
    ntrig=20000,
    outfile="response_imaging_chiral_relative_1MeV_traddet.h5",
    workers=DEFAULT_WORKERS,
):
    """
    Chirality-resolved imaging response (ARM/Phi/Zeta vs off-axis angle Nu) at
    1 MeV. Matches docs/tutorials/07-unbinned_ts_map.ipynb (the "In[1]" restart
    section, i.e. the second detector/axis definition in that notebook -- the
    notebook itself is exploratory and out of linear order, this script is the
    reproducible pipeline).

    Parallelized one off-axis angle bin per worker (bins are independent).
    """

    det = _traditional_detector()

    offaxis_angle_axis = Axis(np.linspace(-80, 80, 16 + 1 + 1) * u.deg, label="Nu")
    chirality_axis = Axis([-2, 0, 2], label="k")
    phi_axis = Axis(
        np.concatenate(
            [
                np.linspace(0, 30, 10 + 1)[:-1],
                np.linspace(30, 80, 10 + 1)[:-1],
                np.linspace(80, 180, 10 + 1),
            ]
        )
        * u.deg,
        label="Phi",
    )
    arm_axis = Axis(
        np.concatenate(
            [
                np.linspace(-180, -73, 10 + 1)[:-1],
                np.linspace(-73, -15, 10 + 1)[:-1],
                np.linspace(-15, -3, 6 + 1)[:-1],
                np.linspace(-3, 3, 7 + 1)[:-1],
                np.linspace(3, 9, 3 + 1)[:-1],
                np.linspace(9, 73, 11 + 1)[:-1],
                np.linspace(73, 180, 11 + 1),
            ]
        )
        * u.deg,
        label="ARM",
    )
    zeta_axis = Axis([-2, 0, 2], label="Zeta")

    h_data_delta = Histogram([offaxis_angle_axis, chirality_axis, phi_axis, arm_axis, zeta_axis])
    h_nsim = Histogram([offaxis_angle_axis, chirality_axis])

    nbins = offaxis_angle_axis.nbins
    seeds = _worker_seeds(nbins)
    tasks = [
        (nNu, nu, ntrig, seed)
        for nNu, (nu, seed) in enumerate(zip(offaxis_angle_axis.centers, seeds))
    ]

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_worker_imaging_bin, t) for t in tasks]
        for n, fut in enumerate(as_completed(futures), 1):
            nNu, nu, k_all, k_trig, phi_trig, arm_trig, zeta_trig = fut.result()

            h_nsim.fill(nu, k_all)
            if k_trig.size:
                h_data_delta.fill(nu, k_trig, phi_trig, arm_trig, zeta_trig)

            print(f"{n}/{nbins} bins done ({(time.time() - t0) / 60:.1f} min elapsed)")

    h_ntrig = h_data_delta.project("Nu", "k")
    Aeff = (h_ntrig / h_nsim) * det.throwing_plane_size
    Aeff.clear_underflow_and_overflow()

    phi_edges_mesh, arm_edges_mesh = np.broadcast_arrays(
        phi_axis.edges[:, None], arm_axis.edges[None, :], subok=True
    )
    phase_space = _phi_arm_phase_space(
        phi_edges_mesh[:-1, :-1],
        phi_edges_mesh[1:, :-1],
        arm_edges_mesh[:-1, :-1],
        arm_edges_mesh[:-1, 1:],
    )
    phase_space_wzeta = phase_space[:, :, None]

    chiral_img_dist = h_data_delta / h_ntrig.contents[:, :, None, None, None]
    chiral_img_pdf = chiral_img_dist / phase_space_wzeta[None, None, :, :, :]
    chiral_img_pdf._contents[np.isnan(chiral_img_pdf._contents)] = 0

    Aeff.write(outfile, overwrite=True, name="Aeff")
    chiral_img_pdf.write(outfile, overwrite=True, name="PDF")
    print(f"Wrote {outfile}")


REGENERATORS = {
    "energy_onaxis": regenerate_energy_onaxis,
    "energy_relative_onaxis": regenerate_energy_relative_onaxis,
    "imaging_chiral_relative": regenerate_imaging_chiral_relative,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("which", choices=list(REGENERATORS) + ["all"])
    parser.add_argument(
        "--ntrig", type=int, default=None, help="Override the default per-run/per-bin trigger count"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of worker processes (default: {DEFAULT_WORKERS}, all available cores)",
    )
    args = parser.parse_args()

    targets = REGENERATORS if args.which == "all" else {args.which: REGENERATORS[args.which]}

    for name, fn in targets.items():
        kwargs = {"workers": args.workers}
        if args.ntrig is not None:
            kwargs["ntrig"] = args.ntrig
        print(f"=== Regenerating {name} (workers={args.workers}) ===")
        fn(**kwargs)
