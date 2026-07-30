"""
Pipeline

1. Load the point cloud.
2. Get a rough center from points above a high-Z percentile.
3. Slice the cloud along Y (giving XZ cross sections) and along X
   (giving YZ cross sections). Within each slice, bin the cross-axis
   coordinate, take the max Z per bin to build a 1D profile, and run
   scipy.signal.find_peaks on it. Each peak is a candidate rim point.
4. Pool all candidate points from both slicing directions and split them
   into "inner" and "outer" groups using a 1D two-means split on their
   radial distance from the rough center.
5. Fit an ellipse (direct least-squares conic fit, Halir & Flusser 1998)
   to each group's (x, y) coordinates.
6. Fit a plane or quadratic surface to the true baseline (points safely
   outside the outer rim, i.e. excluding the ring and the cut interior)
   and subtract it from the whole cloud, so the baseline sits at Z=0
   and any tilt/bow across the sample is corrected.
7. Report max/min/std/mean/median of Z (post-flattening) for each edge,
   plus the fitted ellipse geometry, and optionally save the flattened
   point cloud and a diagnostic plot.

"""

import argparse
import re
import numpy as np
from scipy.signal import find_peaks, peak_widths




def load_xyz(path, skip_header=None, verbose=True):
    """
    Loading xyz file, 
    skip_header: force a specific number of header lines to skip. If
    None (default), auto-detects by scanning for the first line that
    parses as >=3 numeric fields.
    """
    def split_line(line):
        return re.split(r"[,\s]+", line.strip())

    if skip_header is None:
        peek_lines = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(1000):
                line = f.readline()
                if not line:
                    break
                peek_lines.append(line)

        skip_header = None
        for i, line in enumerate(peek_lines):
            parts = [p for p in split_line(line) if p != ""]
            if len(parts) < 3:
                continue
            try:
                float(parts[0]); float(parts[1]); float(parts[2])
            except ValueError:
                continue
            skip_header = i
            break
        if skip_header is None:
            raise ValueError(
                f"Could not find a numeric 'x y z' data row in the first "
                f"{len(peek_lines)} lines of {path}. Check that this is "
                f"really a point-cloud export and not, e.g., an image or "
                f"binary Zygo .dat/.datx file, or pass skip_header= "
                f"explicitly if the header is longer than that."
            )
        if verbose and skip_header > 0:
            print(f"Detected {skip_header} header line(s) before the data "
                  f"starts (first skipped line: {peek_lines[0].strip()!r})")

    # Fast path: vectorized parsing via pandas. Handles comma- or
    # whitespace-delimited files and coerces any non-numeric bad-pixel
    # markers to NaN (then drops them), without a slow per-line Python loop.
    pts = None
    try:
        import pandas as pd
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(skip_header + 1):
                sample = f.readline()
        if "," in sample:
            df = pd.read_csv(path, skiprows=skip_header, header=None,
                              sep=",", engine="c", usecols=[0, 1, 2],
                              on_bad_lines="skip")
        else:
            df = pd.read_csv(path, skiprows=skip_header, header=None,
                              sep=r"\s+", engine="python", usecols=[0, 1, 2],
                              on_bad_lines="skip")
        df = df.apply(pd.to_numeric, errors="coerce")
        arr = df.to_numpy(dtype=float)
        bad_mask = np.isnan(arr).any(axis=1)
        n_bad = int(bad_mask.sum())
        pts = arr[~bad_mask]
        if verbose and n_bad:
            print(f"Skipped {n_bad} row(s) that weren't valid numeric x y z "
                  f"triples (bad-pixel markers, short lines, etc.)")
    except Exception as e:
        if verbose:
            print(f"Fast (pandas) parser failed ({e!r}), falling back to "
                  f"the slower line-by-line parser.")
        pts = None

    if pts is None or len(pts) == 0:
        # Robust fallback: parse every line by hand, one at a time.
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        rows = []
        n_bad = 0
        for line in lines[skip_header:]:
            parts = [p for p in split_line(line) if p != ""]
            if len(parts) < 3:
                if parts:
                    n_bad += 1
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                n_bad += 1
                continue
            rows.append((x, y, z))
        if not rows:
            raise ValueError(f"No valid numeric x y z rows found in {path} "
                              f"after skipping {skip_header} header line(s).")
        if verbose and n_bad:
            print(f"Skipped {n_bad} row(s) that weren't valid numeric x y z "
                  f"triples (bad-pixel markers, short lines, etc.)")
        pts = np.array(rows, dtype=float)

    if verbose:
        print(f"Parsed {len(pts)} points from {path}")
    return pts

# Slicing + peak detection
def profile_peaks(cross_coord, z, n_bins, prominence, min_peak_distance):
    """
    Bins `cross_coord` into n_bins, take max(z) per bin, and finds peaks in that
    1D profile. Returns list of cross_coord_of_peak, z_of_peak, fwhm.
    """
    lo, hi = cross_coord.min(), cross_coord.max()
    if hi - lo < 1e-9 or len(cross_coord) < 5:
        return []

    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(cross_coord, edges) - 1, 0, n_bins - 1)
    bin_size = (hi - lo) / n_bins

    prof = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            prof[b] = z[mask].max()

    valid = ~np.isnan(prof)
    if valid.sum() < 5:
        return []
    # Interpolate small gaps so find_peaks doesn't get stuck on NaNs.
    prof_filled = prof.copy()
    prof_filled[~valid] = np.interp(
        centers[~valid], centers[valid], prof[valid]
    )

    peaks, _ = find_peaks(
        prof_filled, prominence=prominence, distance=max(1, min_peak_distance)
    )
    if len(peaks) == 0:
        return []

    #widths_samples, *_ = peak_widths(prof_filled, peaks, rel_height=0.5)
    #fwhm_physical = widths_samples * bin_size

    widths_samples, width_heights, left_ips, right_ips = peak_widths(
            prof_filled, peaks, rel_height=0.5
        )
    fwhm_physical = widths_samples * bin_size
    

    ''' #for debugging FWHM
    for i, w in enumerate(fwhm_physical):
        if w < 3:
            #print('plotting')
            import matplotlib.pyplot as plt
            p = peaks[i]
            left_x = centers[0] + left_ips[i] * bin_size
            right_x = centers[0] + right_ips[i] * bin_size
    
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(centers, prof_filled, "-", lw=1, color="steelblue", label="profile")
            ax.plot(centers[p], prof_filled[p], "ro", label="detected peak")
            ax.hlines(width_heights[i], left_x, right_x, color="red",
                          linestyle="--", label=f"FWHM span = {w:.2f}")
            ax.set_xlabel("cross_coord")
            ax.set_ylabel("z")
            ax.set_title(f"Suspicious peak: FWHM={w:.2f} at x={centers[p]:.4f}, "
                             f"z={prof_filled[p]:.4f}")
            ax.legend()
            fig.savefig(f"debug_fwhm_{i}.png")
    '''
    return [(centers[p], prof_filled[p], w) for p, w in zip(peaks, fwhm_physical)]

    

def slice_scan(points, slice_axis, cross_axis, other_axis,
               n_slices, profile_bins, prominence, min_peak_distance,
               min_pts):
    """
    Returns an (n by 4) array of candidate rim points: (x, y, z, fwhm) for each candidate point.
    """
    s = points[:, slice_axis]
    lo, hi = s.min(), s.max()
    edges = np.linspace(lo, hi, n_slices + 1)

    candidates = []
    for i in range(n_slices):
        m = (s >= edges[i]) & (s < edges[i + 1])
        if i == n_slices - 1:  # include the right-most edge point
            m = m | (s == edges[i + 1])
        if m.sum() < min_pts:
            continue

        sub = points[m]
        cross = sub[:, cross_axis]
        z = sub[:, 2]
        other_val = sub[:, other_axis].mean()

        for cross_peak, z_peak, fwhm in profile_peaks(
            cross, z, profile_bins, prominence, min_peak_distance
        ):
            row = [None, None, None, fwhm]
            row[cross_axis] = cross_peak
            row[other_axis] = other_val
            row[2] = z_peak
            candidates.append(row)

    if not candidates:
        return np.empty((0, 4))
    return np.array(candidates, dtype=float)


def select_by_angle(candidates, center, prefer):
    """
    Divides candidate points based on which cross section we should look at (i.e. xz if near the center vs yx if near the top/bottom
    prefer: "xz" or "yz" -- which direction's candidates to keep.
    """
    if len(candidates) == 0:
        return candidates
    dx = np.abs(candidates[:, 0] - center[0])
    dy = np.abs(candidates[:, 1] - center[1])
    xz_is_better = dy > dx
    return candidates[xz_is_better] if prefer == "xz" else candidates[~xz_is_better]

# Inner / outer classification (1D two-means on radius)

def kmeans_1d(values, k=2, n_iter=100, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    centers = rng.choice(values, size=k, replace=False)
    labels = np.zeros(len(values), dtype=int)
    for _ in range(n_iter):
        dist = np.abs(values[:, None] - centers[None, :])
        new_labels = dist.argmin(axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            labels = new_labels
            break
        labels = new_labels
        for c in range(k):
            if (labels == c).any():
                centers[c] = values[labels == c].mean()
    return labels, centers


def classify_inner_outer(candidates, center_xy):
    r = np.hypot(candidates[:, 0] - center_xy[0], candidates[:, 1] - center_xy[1])
    labels, centers = kmeans_1d(r, k=2)
    inner_label = np.argmin(centers)
    inner = candidates[labels == inner_label]
    outer = candidates[labels != inner_label]
    return inner, outer


def classify_and_refine(candidates, center_xy, n_iter=3):
    """
    Classify candidates into inner/outer by radius from initial guess of center_xy, then
    re-estimate the center from the classified points and repeat.
    """
    center = np.array(center_xy, dtype=float)
    inner, outer = classify_inner_outer(candidates, center)
    for _ in range(n_iter):
        parts = [g[:, :2] for g in (inner, outer) if len(g)]
        if not parts:
            break
        new_center = np.vstack(parts).mean(axis=0)
        if np.allclose(new_center, center, atol=1e-6):
            center = new_center
            break
        center = new_center
        inner, outer = classify_inner_outer(candidates, center)
    return inner, outer, center

# Ellipse fit (Halir & Flusser 1998 direct least-squares fit) 

def fit_ellipse_conic(x, y):
    x = x[:, None]
    y = y[:, None]
    D1 = np.hstack([x ** 2, x * y, y ** 2])
    D2 = np.hstack([x, y, np.ones_like(x)])
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    Cinv = np.array([[0, 0, 0.5], [0, -1, 0], [0.5, 0, 0]])
    M = Cinv @ M
    eigval, eigvec = np.linalg.eig(M)
    cond = 4 * eigvec[0] * eigvec[2] - eigvec[1] ** 2
    valid = np.where(cond > 0)[0]
    if len(valid) == 0:
        raise RuntimeError("Ellipse fit failed: no valid eigenvector found "
                            "(check that points actually form a ring, not a line).")
    a1 = eigvec[:, valid[0]]
    a2 = T @ a1
    return np.concatenate([a1, a2])  # A, B, C, D, E, F


def conic_to_geometric(coeffs):
    A, B, C, D, E, F = coeffs
    # Center
    M = np.array([[2 * A, B], [B, 2 * C]])
    rhs = np.array([-D, -E])
    x0, y0 = np.linalg.solve(M, rhs)

    # Axes / angle: evaluate semi-axes from eigenvalues of the quadratic
    # form after translating to the center (numerically simpler and more
    # robust than the closed-form conic-parameter formulas).
    Aq = np.array([[A, B / 2], [B / 2, C]])
    Fc = A * x0 ** 2 + B * x0 * y0 + C * y0 ** 2 + D * x0 + E * y0 + F
    eigval, eigvec = np.linalg.eigh(Aq)
    axes = np.sqrt(-Fc / eigval)
    axes = np.sort(axes)[::-1]  # major, minor

    major_vec = eigvec[:, np.argmin(eigval)]  # smaller eigenvalue -> larger axis
    angle = np.arctan2(major_vec[1], major_vec[0])

    return {
        "center": (x0, y0),
        "semi_major": axes[0],
        "semi_minor": axes[1],
        "angle_rad": angle,
        "angle_deg": np.degrees(angle),
    }


def fit_ellipse(x, y):
    coeffs = fit_ellipse_conic(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return conic_to_geometric(coeffs)


def ellipse_normalized_radius(x, y, ellipse):
    """
    Distance from a point to the ellipse center, normalized so that 1.0
    means "exactly on the ellipse". <1 is inside, >1 is outside.
    """
    x0, y0 = ellipse["center"]
    a, b, ang = ellipse["semi_major"], ellipse["semi_minor"], ellipse["angle_rad"]
    dx = np.asarray(x) - x0
    dy = np.asarray(y) - y0
    rx = dx * np.cos(ang) + dy * np.sin(ang)
    ry = -dx * np.sin(ang) + dy * np.cos(ang)
    return np.sqrt((rx / a) ** 2 + (ry / b) ** 2)


def reject_ellipse_outliers(points, ellipse, k=3.0):
    """
    Drop points whose (x, y) deviates too far from an already-fitted
    ellipse

    Returns (filtered_points, keep_mask) where keep_mask indexes into
    the original points array.
    """
    if len(points) == 0 or ellipse is None:
        return points, np.ones(len(points), dtype=bool)

    norm_r = ellipse_normalized_radius(points[:, 0], points[:, 1], ellipse)
    dev = norm_r - 1.0
    med = np.median(dev)
    mad = np.median(np.abs(dev - med))
    sigma = 1.4826 * mad
    if sigma <= 0:
        return points, np.ones(len(points), dtype=bool)

    keep = np.abs(dev - med) <= k * sigma
    return points[keep], keep


def fwhm_stats(points):
    """
    FWHM stats (full width at half prominence) for each group of
    edge points. 
    """
    if len(points) == 0 or points.shape[1] < 4:
        return None
    w = points[:, 3]
    from scipy import stats
    return {
        "n": len(w),
        "min": float(w.min()),
        "max": float(w.max()),
        "mean": float(w.mean()),
        "median": float(np.median(w)),
        "std": float(w.std()),
        "mode":float(stats.mode(w).mode)
    }


# --------------------------------------------------------------------------
# Baseline flattening: fit + subtract a low-order polynomial surface
# (plane or quadratic) using only points safely outside the cut, so the
# baseline sits at Z=0 and tilt/bow across the sample is corrected.
# --------------------------------------------------------------------------
def poly_design_matrix(x, y, order):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if order == 1:
        return np.column_stack([x, y, np.ones_like(x)])
    elif order == 2:
        return np.column_stack([x ** 2, y ** 2, x * y, x, y, np.ones_like(x)])
    else:
        raise ValueError("surface order must be 1 (plane) or 2 (quadratic)")


def fit_surface(x, y, z, order=2, robust_iters=3, robust_k=3.0):
    """
    Least-squares fit of a plane (order=1) or quadratic (order=2) surface
    to (x, y, z), with iterative outlier rejection (robust to stray noise
    or any ridge/edge points that leak into the baseline mask).
    Returns (coeffs, inlier_mask).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    mask = np.ones(len(x), dtype=bool)
    coeffs = None
    for it in range(robust_iters + 1):
        A = poly_design_matrix(x[mask], y[mask], order)
        coeffs, *_ = np.linalg.lstsq(A, z[mask], rcond=None)
        resid = z - poly_design_matrix(x, y, order) @ coeffs
        if it < robust_iters:
            med = np.median(resid[mask])
            mad = np.median(np.abs(resid[mask] - med))
            thresh = robust_k * 1.4826 * mad if mad > 0 else np.inf
            mask = np.abs(resid - med) < thresh
    return coeffs, mask


def evaluate_surface(coeffs, x, y, order):
    A = poly_design_matrix(x, y, order)
    return A @ coeffs


def flatten_baseline(points, outer_ellipse, order=2, margin=0.08,
                      robust_iters=3, robust_k=3.0):
    """
    Fit a plane/quadratic surface to points outside the outer rim
    and subtract it from every point in the
    cloud. Returns (points_flattened, coeffs, baseline_mask).

    margin: fraction beyond the outer ellipse's normalized radius (1.0)
            required for a point to be treated as baseline (i.e. not on the edge)
    """
    norm_r = ellipse_normalized_radius(points[:, 0], points[:, 1], outer_ellipse)
    baseline_mask = norm_r > (1.0 + margin)
    if baseline_mask.sum() < 20:
        raise RuntimeError(
            "Too few baseline points found outside the cut to fit a "
            "surface. Try lowering --baseline-margin."
        )

    coeffs, inliers = fit_surface(
        points[baseline_mask, 0], points[baseline_mask, 1], points[baseline_mask, 2],
        order=order, robust_iters=robust_iters, robust_k=robust_k,
    )

    surface_z = evaluate_surface(coeffs, points[:, 0], points[:, 1], order)
    flattened = points.copy()
    flattened[:, 2] = points[:, 2] - surface_z

    # combine the two masks (which baseline_mask points survived robust fitting)
    full_inlier_mask = np.zeros(len(points), dtype=bool)
    idx = np.where(baseline_mask)[0]
    full_inlier_mask[idx[inliers]] = True

    return flattened, coeffs, full_inlier_mask


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------
def z_stats(z):
    return {
        "n": len(z),
        "max": float(np.max(z)),
        "min": float(np.min(z)),
        "std": float(np.std(z)),
        "mean": float(np.mean(z)),
        "median": float(np.median(z)),
    }


# Main

def run_pipeline(xyz_file, skip_header=None, n_slices=200, profile_bins=300,
                  prominence=None, min_peak_distance=3, min_pts=15,
                  z_percentile=90, plot=None, surface_order=2,
                  baseline_margin=0.08, no_flatten=False, flatten_output=None,
                  subsample=15000, reject_outliers=True, outlier_k=3.0,
                  drop_negative_z=True, directional_split=True):
    """
    Run the full pipeline.

    Returns a dict with keys "inner" and "outer", each holding
    {"stats": {...}, "ellipse": {...}, "fwhm_stats": {...},
    "points": ndarray of (x, y, z, fwhm)}.
    """
    pts = load_xyz(xyz_file, skip_header=skip_header)
    print(f"Loaded {len(pts)} points from {xyz_file}")

    z = pts[:, 2]

    # Rough global detrend, used only to get a clean noise/residual signal
    # for (a) the rough center seed and (b) the auto-prominence estimate.
    # Tilt/bow in the raw data would otherwise swamp both.
    detrend_coeffs, _ = fit_surface(pts[:, 0], pts[:, 1], z, order=2, robust_iters=2)
    resid = z - evaluate_surface(detrend_coeffs, pts[:, 0], pts[:, 1], 2)

    if prominence is None:
        mad = np.median(np.abs(resid - np.median(resid)))
        sigma = 1.4826 * mad
        # Use ~5 sigma, not 3: with hundreds of slices x hundreds of
        # profile bins each, a 3-sigma threshold produces many spurious
        # peaks from the noise tail alone (multiple-comparisons effect).
        prominence = 5 * sigma if sigma > 0 else 0.1 * (resid.max() - resid.min())
        print(f"Auto prominence threshold (from detrended residual, "
              f"noise sigma~{sigma:.4g}): {prominence:.4g}")

    # Rough center: random value. A raw Z-percentile threshold can be
    # skewed by background waviness/noise far from the actual ring, so the
    # real center used for classification is refined from the detected
    # ridge candidates themselves (see classify_and_refine below).
    thresh = np.percentile(resid, z_percentile)
    hi_pts = pts[resid >= thresh]
    rough_center = (hi_pts[:, 0].mean(), hi_pts[:, 1].mean())
    print(f"Rough (Z-percentile-seeded) center estimate: {rough_center}")

    # XZ cross sections: slice along y (axis 1), profile along x (axis 0)
    cand_xz = slice_scan(pts, slice_axis=1, cross_axis=0, other_axis=1,
                          n_slices=n_slices, profile_bins=profile_bins,
                          prominence=prominence,
                          min_peak_distance=min_peak_distance,
                          min_pts=min_pts)

    # YZ cross sections: slice along x (axis 0), profile along y (axis 1)
    cand_yz = slice_scan(pts, slice_axis=0, cross_axis=1, other_axis=0,
                          n_slices=n_slices, profile_bins=profile_bins,
                          prominence=prominence,
                          min_peak_distance=min_peak_distance,
                          min_pts=min_pts)

    candidates_xz_raw = len(cand_xz)
    candidates_yz_raw = len(cand_yz)
    if directional_split:
        cand_xz = select_by_angle(cand_xz, rough_center, prefer="xz")
        cand_yz = select_by_angle(cand_yz, rough_center, prefer="yz")
        print(f"Directional split (perpendicular-only): kept {len(cand_xz)}/"
              f"{candidates_xz_raw} XZ candidates, {len(cand_yz)}/"
              f"{candidates_yz_raw} YZ candidates (each angular region "
              f"uses whichever slicing direction cuts it closer to "
              f"perpendicular, avoiding double-counting near the 45-degree "
              f"regions and near-tangent slices near the extremes)")

    candidates = np.vstack([cand_xz, cand_yz])
    print(f"Candidate rim points found: {len(candidates)} "
          f"({len(cand_xz)} from XZ slices, {len(cand_yz)} from YZ slices)")

    if len(candidates) < 10:
        raise RuntimeError("Too few candidate rim points detected. "
                            "Try lowering prominence or min_pts.")

    inner, outer, refined_center = classify_and_refine(candidates, rough_center)
    print(f"Refined center used for classification: "
          f"({refined_center[0]:.4f}, {refined_center[1]:.4f})")
    print(f"Inner-edge candidates: {len(inner)}, Outer-edge candidates: {len(outer)}")

    results = {}
    ellipses = {}
    for name, group in [("inner", inner), ("outer", outer)]:
        if len(group) < 5:
            print(f"WARNING: only {len(group)} points classified as {name}; "
                  "ellipse fit skipped.")
            results[name] = {"stats": z_stats(group[:, 2]) if len(group) else None,
                              "ellipse": None, "points": group}
            continue
        ellipse = fit_ellipse(group[:, 0], group[:, 1])
        ellipses[name] = ellipse
        results[name] = {"ellipse": ellipse, "points": group}

    # Reject outliers against the already-fitted ellipse (no refitting --
    # this just drops points from each group that sit too far from the
    # ellipse geometry already computed above, e.g. slice-detection
    # artifacts or points that got misclassified into the wrong group).
    if reject_outliers:
        for name in ("inner", "outer"):
            if name not in ellipses:
                continue
            group = results[name]["points"]
            filtered, keep = reject_ellipse_outliers(group, ellipses[name], k=outlier_k)
            n_removed = len(group) - len(filtered)
            if n_removed:
                print(f"{name}: rejected {n_removed} outlier point(s) "
                      f"(>{outlier_k} robust-sigma from the fitted ellipse) "
                      f"out of {len(group)}")
            results[name]["points"] = filtered

    flat_pts = None
    if not no_flatten:
        if "outer" not in ellipses:
            print("WARNING: no outer ellipse fit available, cannot flatten baseline.")
        else:
            flat_pts, surf_coeffs, baseline_inliers = flatten_baseline(
                pts, ellipses["outer"], order=surface_order,
                margin=baseline_margin,
            )
            baseline_resid = flat_pts[baseline_inliers, 2]
            print(f"\nBaseline flattening (order={surface_order} surface, "
                  f"{baseline_inliers.sum()} baseline points used):")
            print(f"  baseline residual after flattening: mean={baseline_resid.mean():.5g}  "
                  f"std={baseline_resid.std():.5g}")

            # Recompute each edge group's Z using the fitted surface at
            # its own (x, y), so stats reflect height above the true
            # (leveled, zeroed) baseline rather than raw Z.
            for name in ("inner", "outer"):
                group = results[name]["points"]
                if len(group) == 0:
                    continue
                surf_z = evaluate_surface(surf_coeffs, group[:, 0], group[:, 1],
                                           surface_order)
                group_flat = group.copy()
                group_flat[:, 2] = group[:, 2] - surf_z
                results[name]["points"] = group_flat

            if flatten_output:
                np.savetxt(flatten_output, flat_pts, fmt="%.6f")
                print(f"  Saved flattened point cloud to {flatten_output}")

    if drop_negative_z:
        for name in ("inner", "outer"):
            group = results[name]["points"]
            if len(group) == 0:
                continue
            pos_mask = group[:, 2] >= 0
            n_dropped = len(group) - int(pos_mask.sum())
            if n_dropped:
                print(f"{name}: dropped {n_dropped} point(s) with negative Z "
                      f"before computing statistics/width, out of {len(group)}")
            results[name]["points"] = group[pos_mask]

    for name in ("inner", "outer"):
        group = results[name]["points"]
        if len(group) < 5:
            continue
        stats = z_stats(group[:, 2])
        results[name]["stats"] = stats
        ellipse = results[name]["ellipse"]
        fwhm = fwhm_stats(group)
        results[name]["fwhm_stats"] = fwhm

        print(f"\n--- {name.upper()} EDGE ---")
        print(f"  Z stats{' (flattened)' if flat_pts is not None else ''}: "
              f"max={stats['max']:.4f}  min={stats['min']:.4f}  "
              f"mean={stats['mean']:.4f}  median={stats['median']:.4f}  "
              f"std={stats['std']:.4f}  (n={stats['n']})")
        print(f"  Ellipse: center=({ellipse['center'][0]:.4f}, "
              f"{ellipse['center'][1]:.4f})  "
              f"semi_major={ellipse['semi_major']:.4f}  "
              f"semi_minor={ellipse['semi_minor']:.4f}  "
              f"angle={ellipse['angle_deg']:.2f} deg")
        if fwhm is not None:
            print(f"  FWHM: mean={fwhm['mean']:.4f}  median={fwhm['median']:.4f}  "
                  f"std={fwhm['std']:.4f}  min={fwhm['min']:.4f}  max={fwhm['max']:.4f}  "
                  f"(n={fwhm['n']})")


    if plot:
        make_plot(flat_pts if flat_pts is not None else pts, results, plot,
                   subsample_n=subsample)
        print(f"\nSaved diagnostic plot to {plot}")

    return results


def make_plot(pts, results, path, subsample_n=15000, seed=0):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

    colors = {"inner": "red", "outer": "cyan"}

    # Subsample the background cloud -- plotting the full set (often
    # hundreds of thousands of points) in 3D is slow and not visually
    # useful anyway once ridges are only a couple of pixels wide.
    if len(pts) > subsample_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pts), size=subsample_n, replace=False)
        bg = pts[idx]
    else:
        bg = pts

    fig = plt.figure(figsize=(15, 7))

    # --- Panel 1: 2D top-down view -----------------------------------
    ax1 = fig.add_subplot(1, 2, 1)
    sc = ax1.scatter(bg[:, 0], bg[:, 1], c=bg[:, 2], s=1, cmap="viridis", alpha=0.4)
    fig.colorbar(sc, ax=ax1, label="Z", fraction=0.046, pad=0.04)
    for name, res in results.items():
        pts_g = res["points"]
        if len(pts_g):
            ax1.scatter(pts_g[:, 0], pts_g[:, 1], c=colors[name], s=8,
                        label=f"{name} edge points")
        ell = res["ellipse"]
        if ell is not None:
            t = np.linspace(0, 2 * np.pi, 300)
            x0, y0 = ell["center"]
            a, b, ang = ell["semi_major"], ell["semi_minor"], ell["angle_rad"]
            ex = x0 + a * np.cos(t) * np.cos(ang) - b * np.sin(t) * np.sin(ang)
            ey = y0 + a * np.cos(t) * np.sin(ang) + b * np.sin(t) * np.cos(ang)
            ax1.plot(ex, ey, c=colors[name], linewidth=2)
    ax1.set_aspect("equal")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title(f"Top-down view (background: {len(bg)} of {len(pts)} points)")
    ax1.legend(loc="upper right", fontsize=8)

    # --- Panel 2: 3D perspective view ---------------------------------
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(bg[:, 0], bg[:, 1], bg[:, 2], c=bg[:, 2], cmap="viridis",
                s=1, alpha=0.25, linewidths=0)
    for name, res in results.items():
        pts_g = res["points"]
        if len(pts_g):
            ax2.scatter(pts_g[:, 0], pts_g[:, 1], pts_g[:, 2], c=colors[name],
                        s=10, label=f"{name} edge points", depthshade=False)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("3D perspective (recast ridges should stand out in Z)")
    ax2.view_init(elev=35, azim=-60)
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    """
    Command-line entry point. 
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xyz_file", help="Path to input .xyz point cloud")
    ap.add_argument("--skip-header", type=int, default=None,
                     help="Number of header lines to skip before the data "
                          "starts. If omitted, auto-detected.")
    ap.add_argument("--n-slices", type=int, default=200,
                     help="Number of slabs per slicing direction (default 200)")
    ap.add_argument("--profile-bins", type=int, default=300,
                     help="Number of cross-axis bins per slice profile (default 300)")
    ap.add_argument("--prominence", type=float, default=None,
                     help="Min peak prominence in Z units. If omitted, "
                          "defaults to 5x the robust noise estimate (MAD) "
                          "of the detrended Z residual.")
    ap.add_argument("--min-peak-distance", type=int, default=3,
                     help="Min separation between peaks, in profile bins (default 3)")
    ap.add_argument("--min-pts", type=int, default=15,
                     help="Minimum points required in a slab to process it (default 15)")
    ap.add_argument("--z-percentile", type=float, default=90,
                     help="Percentile of Z used to seed the rough center (default 90)")
    ap.add_argument("--plot", type=str, default=None,
                     help="If given, save a diagnostic PNG to this path")
    ap.add_argument("--surface-order", type=int, default=2, choices=[1, 2],
                     help="1=plane (tilt only), 2=quadratic (tilt+bow). Default 2.")
    ap.add_argument("--baseline-margin", type=float, default=0.08,
                     help="Fraction beyond the outer rim's radius required "
                          "for a point to count as baseline for the surface "
                          "fit (default 0.08, i.e. 8%% clearance)")
    ap.add_argument("--no-flatten", action="store_true",
                     help="Skip baseline flattening/curvature correction")
    ap.add_argument("--flatten-output", type=str, default=None,
                     help="Path to save the flattened point cloud as xyz "
                          "(x y z_flattened)")
    ap.add_argument("--subsample", type=int, default=15000,
                     help="Max number of original points drawn as plot "
                          "background (default 15000; the full cloud is "
                          "still used for all detection/fitting)")
    ap.add_argument("--no-outlier-rejection", action="store_true",
                     help="Skip rejecting points that deviate from the "
                          "already-fitted ellipse before computing stats")
    ap.add_argument("--outlier-k", type=float, default=3.0,
                     help="Robust-sigma threshold for ellipse outlier "
                          "rejection (default 3.0)")
    ap.add_argument("--keep-negative-z", action="store_true",
                     help="Keep points with negative Z when computing "
                          "stats/width (default: they are dropped)")
    ap.add_argument("--pool-both-directions", action="store_true",
                     help="Use every XZ and YZ candidate everywhere instead "
                          "of splitting by angle (default: each angular "
                          "region only uses whichever slicing direction is "
                          "closer to perpendicular there, to avoid "
                          "double-counting and near-tangent slices)")
    args = ap.parse_args(argv)

    return run_pipeline(
        args.xyz_file,
        skip_header=args.skip_header,
        n_slices=args.n_slices,
        profile_bins=args.profile_bins,
        prominence=args.prominence,
        min_peak_distance=args.min_peak_distance,
        min_pts=args.min_pts,
        z_percentile=args.z_percentile,
        plot=args.plot,
        surface_order=args.surface_order,
        baseline_margin=args.baseline_margin,
        no_flatten=args.no_flatten,
        flatten_output=args.flatten_output,
        subsample=args.subsample,
        reject_outliers=not args.no_outlier_rejection,
        outlier_k=args.outlier_k,
        drop_negative_z=not args.keep_negative_z,
        directional_split=not args.pool_both_directions,
    )


if __name__ == "__main__":
    main()