import math
import time
import numpy as np

from qcf_core import (
    xp,
    GPU_AVAILABLE,
    GRAVITY_INERTIA_3D,
    MOVEMENT_SPEED_3D,
    compute_gradients_3d,
)

try:
    import cupy as cp
except ImportError:
    cp = None


def classify_orbit(r_start, r_end, mean_r, sigma_r, drdt):
    """
    Simple orbit classifier used by all integrators.

    r_start, r_end, mean_r, sigma_r, drdt are scalars.
    """
    rel_diff = abs(r_end - r_start) / max(r_start, 1e-6)
    if r_end > 2.0 * r_start and drdt > 0.0:
        return "Escape"
    if r_end < 0.5 * r_start and drdt < 0.0:
        return "Spiral_in"
    if rel_diff < 0.15 and abs(drdt) < 5e-4:
        return "Quasi_stable"
    return "Other"


def summarize_orbit(r_values, phi_values, r0, dt):
    """
    Legacy helper for the scalar CPU integrators.
    Kept for compatibility (not used by the new batched GPU path).
    """
    if len(r_values) == 0:
        return {
            "r_start": r0,
            "r_end": r0,
            "ratio": 1.0,
            "mean_r": r0,
            "sigma_r": 0.0,
            "drdt": 0.0,
            "orbit_type": "No_data",
            "periapsis_count": 0,
            "mean_dphi_deg": 0.0,
        }

    r_values = np.asarray(r_values, dtype=float)
    phi_values = np.asarray(phi_values, dtype=float)

    r_start_eff = float(r_values[0])
    r_end = float(r_values[-1])
    ratio = r_end / max(r_start_eff, 1e-6)
    mean_r = float(r_values.mean())
    sigma_r = float(r_values.std())
    drdt = (r_end - r_start_eff) / (len(r_values) * dt)

    orbit_type = classify_orbit(r_start_eff, r_end, mean_r, sigma_r, drdt)

    periapsis_count = 0
    last_r = None
    last_dr = None
    for r in r_values:
        if last_r is not None:
            dr = r - last_r
            if last_dr is not None and last_dr < 0.0 and dr > 0.0:
                periapsis_count += 1
            last_dr = dr
        last_r = r

    if periapsis_count > 1 and len(phi_values) > periapsis_count:
        total_dphi = float(phi_values[-1] - phi_values[0])
        mean_dphi = total_dphi / max(periapsis_count, 1)
        mean_dphi_deg = mean_dphi * 180.0 / math.pi
    else:
        mean_dphi_deg = 0.0

    return {
        "r_start": r_start_eff,
        "r_end": r_end,
        "ratio": ratio,
        "mean_r": mean_r,
        "sigma_r": sigma_r,
        "drdt": drdt,
        "orbit_type": orbit_type,
        "periapsis_count": periapsis_count,
        "mean_dphi_deg": mean_dphi_deg,
    }


def _streaming_orbit_summary(r0, dt, r_history, phi_history, periapsis_count):
    """
    Helper used by the batched integrators.

    r_history, phi_history are dicts with scalar entries.
    """
    r_start_eff = float(r_history["r_start"])
    r_end = float(r_history["r_end"])
    count = int(r_history["count"])
    if count == 0:
        return {
            "r_start": r0,
            "r_end": r0,
            "ratio": 1.0,
            "mean_r": r0,
            "sigma_r": 0.0,
            "drdt": 0.0,
            "orbit_type": "No_data",
            "periapsis_count": 0,
            "mean_dphi_deg": 0.0,
        }

    sum_r = float(r_history["sum_r"])
    sum_r2 = float(r_history["sum_r2"])

    mean_r = sum_r / max(count, 1)
    if count > 1:
        var_r = max(sum_r2 / count - mean_r * mean_r, 0.0)
        sigma_r = math.sqrt(var_r)
    else:
        sigma_r = 0.0

    drdt = (r_end - r_start_eff) / (count * dt)
    ratio = r_end / max(r_start_eff, 1e-6)
    orbit_type = classify_orbit(r_start_eff, r_end, mean_r, sigma_r, drdt)

    peri_count = int(periapsis_count)

    phi0 = float(phi_history["phi_first"])
    phi_last = float(phi_history["phi_last"])
    if peri_count > 1:
        total_dphi = phi_last - phi0
        mean_dphi = total_dphi / max(peri_count, 1)
        mean_dphi_deg = mean_dphi * 180.0 / math.pi
    else:
        mean_dphi_deg = 0.0

    return {
        "r_start": r_start_eff,
        "r_end": r_end,
        "ratio": ratio,
        "mean_r": mean_r,
        "sigma_r": sigma_r,
        "drdt": drdt,
        "orbit_type": orbit_type,
        "periapsis_count": peri_count,
        "mean_dphi_deg": mean_dphi_deg,
    }


def _prepare_batch(values, batch_size):
    """
    Convert scalar or 1D array-like to xp array with shape (batch_size,).
    If a scalar is given and batch_size > 1, it is broadcast.
    """
    if hasattr(values, "shape"):
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            arr = np.full((batch_size,), float(arr), dtype=float)
        elif arr.ndim == 1:
            if arr.shape[0] != batch_size:
                raise ValueError("Array length does not match batch_size")
        else:
            raise ValueError("Only 1D arrays are supported for orbit batches")
    else:
        arr = np.full((batch_size,), float(values), dtype=float)

    if GPU_AVAILABLE and xp is not np:
        return xp.asarray(arr, dtype=xp.float64)
    else:
        return xp.asarray(arr, dtype=float)




def _compute_local_acceleration_from_phi(phi_grid, x, y, z, cx, cy, cz, xp_local=None):
    """
    CPU/Cupy-agnostic trilinear interpolation of phi -> analytic gradient.
    phi_grid assumed shape (N,N,N). x,y,z are index-space positions (float arrays/scalars).
    xp_local: numpy or cupy module (optional); default = numpy
    Returns: ax, ay, az, r, inside_mask  (same dtype/shape as x)
    """
    import numpy as _np
    xp = xp_local if xp_local is not None else _np

    # handle scalar inputs uniformly
    scalar = False
    if xp is _np and (not hasattr(x, "__len__")):
        scalar = True
    # convert to arrays
    x_a = xp.array(x) if not hasattr(x, "__len__") else x
    y_a = xp.array(y) if not hasattr(y, "__len__") else y
    z_a = xp.array(z) if not hasattr(z, "__len__") else z

    N = int(phi_grid.shape[0])

    # compute cell lower corner indices and local coords
    fx = xp.floor(x_a)
    fy = xp.floor(y_a)
    fz = xp.floor(z_a)
    ix0 = fx.astype(xp.int64)
    iy0 = fy.astype(xp.int64)
    iz0 = fz.astype(xp.int64)

    inside = (
        (ix0 >= 0) & (ix0 < N-1)
        & (iy0 >= 0) & (iy0 < N-1)
        & (iz0 >= 0) & (iz0 < N-1)
    )

    # clip to valid
    ix0 = xp.clip(ix0, 0, N-2)
    iy0 = xp.clip(iy0, 0, N-2)
    iz0 = xp.clip(iz0, 0, N-2)
    ix1 = ix0 + 1; iy1 = iy0 + 1; iz1 = iz0 + 1

    tx = x_a - fx; ty = y_a - fy; tz = z_a - fz
    tx = xp.clip(tx, 0.0, 1.0); ty = xp.clip(ty, 0.0, 1.0); tz = xp.clip(tz, 0.0, 1.0)

    # gather corners
    c000 = phi_grid[ix0, iy0, iz0]
    c100 = phi_grid[ix1, iy0, iz0]
    c010 = phi_grid[ix0, iy1, iz0]
    c110 = phi_grid[ix1, iy1, iz0]
    c001 = phi_grid[ix0, iy0, iz1]
    c101 = phi_grid[ix1, iy0, iz1]
    c011 = phi_grid[ix0, iy1, iz1]
    c111 = phi_grid[ix1, iy1, iz1]

    # trilinear interp
    c00 = c000 * (1.0 - tx) + c100 * tx
    c10 = c010 * (1.0 - tx) + c110 * tx
    c01 = c001 * (1.0 - tx) + c101 * tx
    c11 = c011 * (1.0 - tx) + c111 * tx
    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    phi_val = c0 * (1.0 - tz) + c1 * tz

    # analytic derivatives in normalized coords
    dc00_dtx = (c100 - c000)
    dc10_dtx = (c110 - c010)
    dc01_dtx = (c101 - c001)
    dc11_dtx = (c111 - c011)
    dc0_dtx = dc00_dtx * (1.0 - ty) + dc10_dtx * ty
    dc1_dtx = dc01_dtx * (1.0 - ty) + dc11_dtx * ty
    dphi_dtx = dc0_dtx * (1.0 - tz) + dc1_dtx * tz

    dc00_dty = (c010 - c000)
    dc10_dty = (c110 - c100)
    dc01_dty = (c011 - c001)
    dc11_dty = (c111 - c101)
    dc0_dty = dc00_dty * (1.0 - tx) + dc10_dty * tx
    dc1_dty = dc01_dty * (1.0 - tx) + dc11_dty * tx
    dphi_dty = dc0_dty * (1.0 - tz) + dc1_dty * tz

    dphi_dtz = (c1 - c0)

    # chain rule (grid spacing assumed 1 index unit)
    dphidx = dphi_dtx
    dphidy = dphi_dty
    dphidz = dphi_dtz

    ax = -dphidx; ay = -dphidy; az = -dphidz

    dx = x_a - cx; dy = y_a - cy; dz = z_a - cz
    r = xp.sqrt(dx*dx + dy*dy + dz*dz)

    # mask outside
    ax = xp.where(inside, ax, 0.0)
    ay = xp.where(inside, ay, 0.0)
    az = xp.where(inside, az, 0.0)

    if scalar:
        return float(ax), float(ay), float(az), float(r), bool(inside)
    return ax, ay, az, r, inside
def _compute_local_acceleration(Gx, Gy, Gz, x, y, z, cx, cy, cz):
    """
    Compute acceleration from a 3D potential gradient (Gx,Gy,Gz)
    at points (x,y,z) using trilinear interpolation.

    x, y, z are xp arrays with the same shape.
    Returns ax, ay, az, r, inside_mask as xp arrays.
    """
    N = Gx.shape[0]

    fx = xp.floor(x)
    fy = xp.floor(y)
    fz = xp.floor(z)

    ix0 = fx.astype(np.int32)
    iy0 = fy.astype(np.int32)
    iz0 = fz.astype(np.int32)

    inside = (
        (ix0 >= 0) & (ix0 < N - 1)
        & (iy0 >= 0) & (iy0 < N - 1)
        & (iz0 >= 0) & (iz0 < N - 1)
    )

    ix0 = xp.clip(ix0, 0, N - 2)
    iy0 = xp.clip(iy0, 0, N - 2)
    iz0 = xp.clip(iz0, 0, N - 2)

    ix1 = ix0 + 1
    iy1 = iy0 + 1
    iz1 = iz0 + 1

    tx = x - fx
    ty = y - fy
    tz = z - fz

    tx = xp.clip(tx, 0.0, 1.0)
    ty = xp.clip(ty, 0.0, 1.0)
    tz = xp.clip(tz, 0.0, 1.0)

    ax = xp.zeros_like(x, dtype=xp.float64)
    ay = xp.zeros_like(y, dtype=xp.float64)
    az = xp.zeros_like(z, dtype=xp.float64)

    dx = x - cx
    dy = y - cy
    dz = z - cz
    r = xp.sqrt(dx * dx + dy * dy + dz * dz)

    if not bool(inside.any()):
        return ax, ay, az, r, inside

    m = inside

    ix0m = ix0[m]
    iy0m = iy0[m]
    iz0m = iz0[m]
    ix1m = ix1[m]
    iy1m = iy1[m]
    iz1m = iz1[m]

    txm = tx[m].astype(xp.float64)
    tym = ty[m].astype(xp.float64)
    tzm = tz[m].astype(xp.float64)

    def trilinear_component(F):
        F000 = F[ix0m, iy0m, iz0m]
        F100 = F[ix1m, iy0m, iz0m]
        F010 = F[ix0m, iy1m, iz0m]
        F110 = F[ix1m, iy1m, iz0m]
        F001 = F[ix0m, iy0m, iz1m]
        F101 = F[ix1m, iy0m, iz1m]
        F011 = F[ix0m, iy1m, iz1m]
        F111 = F[ix1m, iy1m, iz1m]

        F00 = F000 * (1.0 - txm) + F100 * txm
        F10 = F010 * (1.0 - txm) + F110 * txm
        F01 = F001 * (1.0 - txm) + F101 * txm
        F11 = F011 * (1.0 - txm) + F111 * txm

        F0 = F00 * (1.0 - tym) + F10 * tym
        F1 = F01 * (1.0 - tym) + F11 * tym

        return F0 * (1.0 - tzm) + F1 * tzm

    gx_loc = trilinear_component(Gx).astype(xp.float64)
    gy_loc = trilinear_component(Gy).astype(xp.float64)
    gz_loc = trilinear_component(Gz).astype(xp.float64)

    ax[m] = -gx_loc * GRAVITY_INERTIA_3D
    ay[m] = -gy_loc * GRAVITY_INERTIA_3D
    az[m] = -gz_loc * GRAVITY_INERTIA_3D

    return ax, ay, az, r, inside


def _compute_spherical_acceleration(force_dGdr, x, y, z, cx, cy, cz):
    """
    Compute acceleration from a spherical force law dG/dr.
    x, y, z are xp arrays.
    """
    dx = x - cx
    dy = y - cy
    dz = z - cz
    r = xp.sqrt(dx * dx + dy * dy + dz * dz)

    safe_r = xp.where(r < 1e-6, 1e-6, r)

    if GPU_AVAILABLE and xp is not np:
        r_cpu = np.asarray(cp.asnumpy(safe_r))
    else:
        r_cpu = np.asarray(safe_r)
    dGdr_cpu = np.array([force_dGdr(float(rv)) for rv in r_cpu], dtype=float)

    if GPU_AVAILABLE and xp is not np:
        dGdr = xp.asarray(dGdr_cpu, dtype=xp.float64)
    else:
        dGdr = xp.asarray(dGdr_cpu, dtype=float)

    a_r = -dGdr
    factor = a_r / safe_r * GRAVITY_INERTIA_3D

    ax = factor * dx
    ay = factor * dy
    az = factor * dz

    return ax, ay, az, r, xp.ones_like(x, dtype=bool)


# RawKernel for local 3D orbits (one orbit per thread, full time loop inside CUDA)
_local_orbit_kernel_src = r"""
__device__ __forceinline__ int idx3(int i, int j, int k, int N) {
    return (i * N + j) * N + k;
}

extern "C" __global__
void integrate_local_orbits(
    const float* __restrict__ Gx,
    const float* __restrict__ Gy,
    const float* __restrict__ Gz,
    const int N,
    const double cx,
    const double cy,
    const double cz,
    double* x,
    double* y,
    double* z,
    double* vx,
    double* vy,
    double* vz,
    const double dt,
    const int steps,
    const double gravity_inertia,
    const double movement_speed,
    double* r_start,
    double* r_end,
    double* sum_r,
    double* sum_r2,
    long long* count,
    long long* periapsis_count,
    double* phi_first,
    double* phi_last,
    const int batch_size
)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= batch_size) {
        return;
    }

    double xi = x[i];
    double yi = y[i];
    double zi = z[i];
    double vxi = vx[i];
    double vyi = vy[i];
    double vzi = vz[i];

    double r_start_local = 0.0;
    double r_end_local = 0.0;
    double sum_r_local = 0.0;
    double sum_r2_local = 0.0;
    long long count_local = 0;
    long long peri_local = 0;
    double last_r_local = 0.0;
    double last_dr_local = 0.0;
    int have_r_start = 0;
    int have_last_r = 0;
    double phi_first_local = 0.0;
    double phi_last_local = 0.0;
    int have_phi_first = 0;

    for (int step = 0; step < steps; ++step) {
        xi += vxi * dt;
        yi += vyi * dt;
        zi += vzi * dt;

        double dx = xi - cx;
        double dy = yi - cy;
        double dz = zi - cz;

        if (fabs(dx) > (double)N || fabs(dy) > (double)N || fabs(dz) > (double)N) {
            break;
        }

        double gx = xi;
        double gy = yi;
        double gz = zi;

        int ix0 = (int)floor(gx);
        int iy0 = (int)floor(gy);
        int iz0 = (int)floor(gz);

        if (ix0 < 0 || ix0 >= N - 1 ||
            iy0 < 0 || iy0 >= N - 1 ||
            iz0 < 0 || iz0 >= N - 1) {
            break;
        }

        int ix1 = ix0 + 1;
        int iy1 = iy0 + 1;
        int iz1 = iz0 + 1;

        double tx = gx - (double)ix0;
        double ty = gy - (double)iy0;
        double tz = gz - (double)iz0;

        if (tx < 0.0) tx = 0.0;
        if (tx > 1.0) tx = 1.0;
        if (ty < 0.0) ty = 0.0;
        if (ty > 1.0) ty = 1.0;
        if (tz < 0.0) tz = 0.0;
        if (tz > 1.0) tz = 1.0;

        double c000 = (1.0 - tx) * (1.0 - ty) * (1.0 - tz);
        double c100 = tx * (1.0 - ty) * (1.0 - tz);
        double c010 = (1.0 - tx) * ty * (1.0 - tz);
        double c110 = tx * ty * (1.0 - tz);
        double c001 = (1.0 - tx) * (1.0 - ty) * tz;
        double c101 = tx * (1.0 - ty) * tz;
        double c011 = (1.0 - tx) * ty * tz;
        double c111 = tx * ty * tz;

        int i000 = idx3(ix0, iy0, iz0, N);
        int i100 = idx3(ix1, iy0, iz0, N);
        int i010 = idx3(ix0, iy1, iz0, N);
        int i110 = idx3(ix1, iy1, iz0, N);
        int i001 = idx3(ix0, iy0, iz1, N);
        int i101 = idx3(ix1, iy0, iz1, N);
        int i011 = idx3(ix0, iy1, iz1, N);
        int i111 = idx3(ix1, iy1, iz1, N);

        double gx_val =
            c000 * (double)Gx[i000] +
            c100 * (double)Gx[i100] +
            c010 * (double)Gx[i010] +
            c110 * (double)Gx[i110] +
            c001 * (double)Gx[i001] +
            c101 * (double)Gx[i101] +
            c011 * (double)Gx[i011] +
            c111 * (double)Gx[i111];

        double gy_val =
            c000 * (double)Gy[i000] +
            c100 * (double)Gy[i100] +
            c010 * (double)Gy[i010] +
            c110 * (double)Gy[i110] +
            c001 * (double)Gy[i001] +
            c101 * (double)Gy[i101] +
            c011 * (double)Gy[i011] +
            c111 * (double)Gy[i111];

        double gz_val =
            c000 * (double)Gz[i000] +
            c100 * (double)Gz[i100] +
            c010 * (double)Gz[i010] +
            c110 * (double)Gz[i110] +
            c001 * (double)Gz[i001] +
            c101 * (double)Gz[i101] +
            c011 * (double)Gz[i011] +
            c111 * (double)Gz[i111];

        double ax = -gx_val * gravity_inertia;
        double ay = -gy_val * gravity_inertia;
        double az = -gz_val * gravity_inertia;

        vxi += dt * ax * movement_speed;
        vyi += dt * ay * movement_speed;
        vzi += dt * az * movement_speed;

        double r = sqrt(dx * dx + dy * dy + dz * dz);
        double phi = atan2(dy, dx);

        if (!have_r_start) {
            r_start_local = r;
            have_r_start = 1;
        }

        r_end_local = r;
        sum_r_local += r;
        sum_r2_local += r * r;
        count_local += 1;

        if (have_last_r) {
            double dr = r - last_r_local;
            if (last_dr_local < 0.0 && dr > 0.0) {
                peri_local += 1;
            }
            last_dr_local = dr;
        } else {
            have_last_r = 1;
            last_dr_local = 0.0;
        }

        last_r_local = r;

        if (!have_phi_first) {
            phi_first_local = phi;
            have_phi_first = 1;
        }
        phi_last_local = phi;
    }

    x[i] = xi;
    y[i] = yi;
    z[i] = zi;
    vx[i] = vxi;
    vy[i] = vyi;
    vz[i] = vzi;

    r_start[i] = r_start_local;
    r_end[i] = r_end_local;
    sum_r[i] = sum_r_local;
    sum_r2[i] = sum_r2_local;
    count[i] = count_local;
    periapsis_count[i] = peri_local;
    phi_first[i] = phi_first_local;
    phi_last[i] = phi_last_local;
}
"""
_spherical_orbit_kernel_src = r"""
__device__ __forceinline__ double interp_dGdr(
    double r,
    const double* __restrict__ r_table,
    const double* __restrict__ dGdr_table,
    const int n_r
) {
    if (n_r <= 1) {
        return dGdr_table[0];
    }
    if (r <= r_table[0]) {
        return dGdr_table[0];
    }
    if (r >= r_table[n_r - 1]) {
        return dGdr_table[n_r - 1];
    }
    int idx = 0;
    while (idx < n_r - 1 && r_table[idx + 1] < r) {
        idx++;
    }
    double r0 = r_table[idx];
    double r1 = r_table[idx + 1];
    double f0 = dGdr_table[idx];
    double f1 = dGdr_table[idx + 1];
    double t = (r - r0) / (r1 - r0);
    return f0 + t * (f1 - f0);
}

extern "C" __global__
void integrate_spherical_orbits(
    const double* __restrict__ r_table,
    const double* __restrict__ dGdr_table,
    const int n_r,
    const int N,
    const double cx,
    const double cy,
    const double cz,
    double* x,
    double* y,
    double* z,
    double* vx,
    double* vy,
    double* vz,
    const double dt,
    const int steps,
    const double gravity_inertia,
    const double movement_speed,
    double* r_start,
    double* r_end,
    double* sum_r,
    double* sum_r2,
    long long* count,
    long long* periapsis_count,
    double* phi_first,
    double* phi_last,
    const int batch_size
)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= batch_size) {
        return;
    }

    double xi = x[i];
    double yi = y[i];
    double zi = z[i];
    double vxi = vx[i];
    double vyi = vy[i];
    double vzi = vz[i];

    double r_start_local = 0.0;
    double r_end_local = 0.0;
    double sum_r_local = 0.0;
    double sum_r2_local = 0.0;
    long long count_local = 0;
    long long peri_local = 0;
    double last_r_local = 0.0;
    double last_dr_local = 0.0;
    int have_r_start = 0;
    int have_last_r = 0;
    double phi_first_local = 0.0;
    double phi_last_local = 0.0;
    int have_phi_first = 0;

    for (int step = 0; step < steps; ++step) {
        xi += vxi * dt;
        yi += vyi * dt;
        zi += vzi * dt;

        double dx = xi - cx;
        double dy = yi - cy;
        double dz = zi - cz;

        if (fabs(dx) > (double)N || fabs(dy) > (double)N || fabs(dz) > (double)N) {
            break;
        }

        double r = sqrt(dx * dx + dy * dy + dz * dz);
        double phi = atan2(dy, dx);

        double safe_r = r;
        if (safe_r < 1e-6) {
            safe_r = 1e-6;
        }

        double dGdr = interp_dGdr(safe_r, r_table, dGdr_table, n_r);
        double a_r = -dGdr;
        double factor = a_r / safe_r * gravity_inertia;

        double ax = factor * dx;
        double ay = factor * dy;
        double az = factor * dz;

        vxi += dt * ax * movement_speed;
        vyi += dt * ay * movement_speed;
        vzi += dt * az * movement_speed;

        if (!have_r_start) {
            r_start_local = r;
            have_r_start = 1;
        }

        r_end_local = r;
        sum_r_local += r;
        sum_r2_local += r * r;
        count_local += 1;

        if (have_last_r) {
            double dr = r - last_r_local;
            if (last_dr_local < 0.0 && dr > 0.0) {
                peri_local += 1;
            }
            last_dr_local = dr;
        } else {
            have_last_r = 1;
            last_dr_local = 0.0;
        }

        last_r_local = r;

        if (!have_phi_first) {
            phi_first_local = phi;
            have_phi_first = 1;
        }
        phi_last_local = phi;
    }

    x[i] = xi;
    y[i] = yi;
    z[i] = zi;
    vx[i] = vxi;
    vy[i] = vyi;
    vz[i] = vzi;

    r_start[i] = r_start_local;
    r_end[i] = r_end_local;
    sum_r[i] = sum_r_local;
    sum_r2[i] = sum_r2_local;
    count[i] = count_local;
    periapsis_count[i] = peri_local;
    phi_first[i] = phi_first_local;
    phi_last[i] = phi_last_local;
}
"""


_local_orbit_kernel = None


def _get_local_orbit_kernel():
    global _local_orbit_kernel
    if _local_orbit_kernel is None:
        if not GPU_AVAILABLE or cp is None:
            raise RuntimeError("CuPy RawKernel is only available on GPU.")
        _local_orbit_kernel = cp.RawKernel(
            _local_orbit_kernel_src, "integrate_local_orbits"
        )
    return _local_orbit_kernel

_spherical_orbit_kernel = None

def _get_spherical_orbit_kernel():
    global _spherical_orbit_kernel
    if _spherical_orbit_kernel is None:
        if not GPU_AVAILABLE or cp is None:
            raise RuntimeError("CuPy RawKernel is only available on GPU.")
        _spherical_orbit_kernel = cp.RawKernel(
            _spherical_orbit_kernel_src, "integrate_spherical_orbits"
        )
    return _spherical_orbit_kernel


def _integrate_orbits_core_local_rawkernel(G, vx, vy, vz, r0, z0, steps, dt):
    """
    Fast path for local 3D orbits using a single CuPy RawKernel.
    Each CUDA thread integrates one orbit over all steps.
    """
    if not GPU_AVAILABLE or cp is None:
        raise RuntimeError("RawKernel path requested but GPU is not available.")

    candidates = []
    for v in (vx, vy, vz, r0, z0):
        if hasattr(v, "shape"):
            arr = np.asarray(v)
            if arr.ndim == 1:
                candidates.append(arr.shape[0])
    batch_size = max(candidates) if candidates else 1

    vx_arr = _prepare_batch(vx, batch_size)
    vy_arr = _prepare_batch(vy, batch_size)
    vz_arr = _prepare_batch(vz, batch_size)
    r0_arr = _prepare_batch(r0, batch_size)
    z0_arr = _prepare_batch(z0, batch_size)

    N = int(G.shape[0])
    cx = 0.5 * (N - 1)
    cy = 0.5 * (N - 1)
    cz = 0.5 * (N - 1)

    Gx, Gy, Gz = compute_gradients_3d(G)

    cx_arr = xp.asarray(cx, dtype=xp.float64)
    cy_arr = xp.asarray(cy, dtype=xp.float64)
    cz_arr = xp.asarray(cz, dtype=xp.float64)

    x = cx_arr + r0_arr
    y = cy_arr
    z = cz_arr + z0_arr

    vx3 = vx_arr.astype(xp.float64, copy=True)
    vy3 = vy_arr.astype(xp.float64, copy=True)
    vz3 = vz_arr.astype(xp.float64, copy=True)

    x = cp.asarray(x, dtype=cp.float64)
    y = cp.asarray(y, dtype=cp.float64)
    z = cp.asarray(z, dtype=cp.float64)
    vx3 = cp.asarray(vx3, dtype=cp.float64)
    vy3 = cp.asarray(vy3, dtype=cp.float64)
    vz3 = cp.asarray(vz3, dtype=cp.float64)

    Gx = cp.asarray(Gx, dtype=cp.float32)
    Gy = cp.asarray(Gy, dtype=cp.float32)
    Gz = cp.asarray(Gz, dtype=cp.float32)

    r_start = cp.zeros(batch_size, dtype=cp.float64)
    r_end = cp.zeros(batch_size, dtype=cp.float64)
    sum_r = cp.zeros(batch_size, dtype=cp.float64)
    sum_r2 = cp.zeros(batch_size, dtype=cp.float64)
    count = cp.zeros(batch_size, dtype=cp.int64)
    periapsis_count = cp.zeros(batch_size, dtype=cp.int64)
    phi_first = cp.zeros(batch_size, dtype=cp.float64)
    phi_last = cp.zeros(batch_size, dtype=cp.float64)

    kernel = _get_local_orbit_kernel()

    threads = 256
    blocks = (batch_size + threads - 1) // threads

    start_time = time.time()

    kernel(
        (blocks,),
        (threads,),
        (
            Gx,
            Gy,
            Gz,
            np.int32(N),
            float(cx),
            float(cy),
            float(cz),
            x,
            y,
            z,
            vx3,
            vy3,
            vz3,
            float(dt),
            np.int32(int(steps)),
            float(GRAVITY_INERTIA_3D),
            float(MOVEMENT_SPEED_3D),
            r_start,
            r_end,
            sum_r,
            sum_r2,
            count,
            periapsis_count,
            phi_first,
            phi_last,
            np.int32(batch_size),
        ),
    )

    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - start_time

    r_start_h = cp.asnumpy(r_start)
    r_end_h = cp.asnumpy(r_end)
    sum_r_h = cp.asnumpy(sum_r)
    sum_r2_h = cp.asnumpy(sum_r2)
    count_h = cp.asnumpy(count)
    periapsis_h = cp.asnumpy(periapsis_count)
    phi_first_h = cp.asnumpy(phi_first)
    phi_last_h = cp.asnumpy(phi_last)
    r0_h = np.asarray(cp.asnumpy(r0_arr))

    idx0 = 0
    r_hist_first = {
        "r_start": r_start_h[idx0],
        "r_end": r_end_h[idx0],
        "sum_r": sum_r_h[idx0],
        "sum_r2": sum_r2_h[idx0],
        "count": count_h[idx0],
    }
    phi_hist_first = {
        "phi_first": phi_first_h[idx0],
        "phi_last": phi_last_h[idx0],
    }
    summary_first = _streaming_orbit_summary(
        float(r0_h[idx0]),
        dt,
        r_hist_first,
        phi_hist_first,
        int(periapsis_h[idx0]),
    )

    total_active = int((count_h > 0).sum())
    if total_active <= 1:
        summary = summary_first.copy()
    else:
        mask_used = count_h > 0
        r_start_mean = float(r_start_h[mask_used].mean())
        r_end_mean = float(r_end_h[mask_used].mean())
        ratio_mean = r_end_mean / max(r_start_mean, 1e-6)
        mean_r_mean = float((sum_r_h[mask_used] / count_h[mask_used]).mean())
        sigma_mean = float(
            np.sqrt(
                np.maximum(
                    sum_r2_h[mask_used] / count_h[mask_used]
                    - (sum_r_h[mask_used] / count_h[mask_used]) ** 2,
                    0.0,
                )
            ).mean()
        )
        drdt_mean = float(
            (
                (r_end_h[mask_used] - r_start_h[mask_used])
                / (count_h[mask_used] * dt)
            ).mean()
        )
        periapsis_mean = float(periapsis_h[mask_used].mean())

        summary = {
            "r_start": r_start_mean,
            "r_end": r_end_mean,
            "ratio": ratio_mean,
            "mean_r": mean_r_mean,
            "sigma_r": sigma_mean,
            "drdt": drdt_mean,
            "orbit_type": summary_first["orbit_type"],
            "periapsis_count": periapsis_mean,
            "mean_dphi_deg": summary_first["mean_dphi_deg"],
        }

    summary["elapsed"] = elapsed
    summary["batch_size"] = batch_size
    summary["first_orbit"] = summary_first
    return summary

def _integrate_orbits_core_spherical_rawkernel(force_dGdr, vx, vy, vz, r0, z0, steps, dt):
    """
    Fast path for spherical 3D orbits using a CuPy RawKernel.
    force_dGdr musi byc zbudowane przez make_force_dGdr_from_profile,
    wtedy ma atrybuty r_arr, dGdr_arr.
    """
    if not GPU_AVAILABLE or cp is None:
        raise RuntimeError("RawKernel path requested but GPU is not available.")

    if not hasattr(force_dGdr, "r_arr") or not hasattr(force_dGdr, "dGdr_arr"):
        raise RuntimeError("force_dGdr has no r_arr/dGdr_arr; cannot use spherical RawKernel path.")

    r_table = np.asarray(force_dGdr.r_arr, dtype=float)
    dGdr_table = np.asarray(force_dGdr.dGdr_arr, dtype=float)
    n_r = int(r_table.shape[0])

    candidates = []
    for v in (vx, vy, vz, r0, z0):
        if hasattr(v, "shape"):
            arr = np.asarray(v)
            if arr.ndim == 1:
                candidates.append(arr.shape[0])
    batch_size = max(candidates) if candidates else 1

    vx_arr = _prepare_batch(vx, batch_size)
    vy_arr = _prepare_batch(vy, batch_size)
    vz_arr = _prepare_batch(vz, batch_size)
    r0_arr = _prepare_batch(r0, batch_size)
    z0_arr = _prepare_batch(z0, batch_size)

    # w wersji sferycznej nie ma prawdziwej siatki G,
    # ale logika w CPU path zaklada N=256
    N = 256
    cx = 0.5 * (N - 1)
    cy = 0.5 * (N - 1)
    cz = 0.5 * (N - 1)

    cx_arr = xp.asarray(cx, dtype=xp.float64)
    cy_arr = xp.asarray(cy, dtype=xp.float64)
    cz_arr = xp.asarray(cz, dtype=xp.float64)

    x = cx_arr + r0_arr
    y = cy_arr
    z = cz_arr + z0_arr

    vx3 = vx_arr.astype(xp.float64, copy=True)
    vy3 = vy_arr.astype(xp.float64, copy=True)
    vz3 = vz_arr.astype(xp.float64, copy=True)

    # na GPU
    x = cp.asarray(x, dtype=cp.float64)
    y = cp.asarray(y, dtype=cp.float64)
    z = cp.asarray(z, dtype=cp.float64)
    vx3 = cp.asarray(vx3, dtype=cp.float64)
    vy3 = cp.asarray(vy3, dtype=cp.float64)
    vz3 = cp.asarray(vz3, dtype=cp.float64)

    r_start = cp.zeros(batch_size, dtype=cp.float64)
    r_end = cp.zeros(batch_size, dtype=cp.float64)
    sum_r = cp.zeros(batch_size, dtype=cp.float64)
    sum_r2 = cp.zeros(batch_size, dtype=cp.float64)
    count = cp.zeros(batch_size, dtype=cp.int64)
    periapsis_count = cp.zeros(batch_size, dtype=cp.int64)
    phi_first = cp.zeros(batch_size, dtype=cp.float64)
    phi_last = cp.zeros(batch_size, dtype=cp.float64)

    r_table_gpu = cp.asarray(r_table, dtype=cp.float64)
    dGdr_table_gpu = cp.asarray(dGdr_table, dtype=cp.float64)

    kernel = _get_spherical_orbit_kernel()

    threads = 256
    blocks = (batch_size + threads - 1) // threads

    start_time = time.time()

    kernel(
        (blocks,),
        (threads,),
        (
            r_table_gpu,
            dGdr_table_gpu,
            np.int32(n_r),
            np.int32(N),
            float(cx),
            float(cy),
            float(cz),
            x,
            y,
            z,
            vx3,
            vy3,
            vz3,
            float(dt),
            np.int32(int(steps)),
            float(GRAVITY_INERTIA_3D),
            float(MOVEMENT_SPEED_3D),
            r_start,
            r_end,
            sum_r,
            sum_r2,
            count,
            periapsis_count,
            phi_first,
            phi_last,
            np.int32(batch_size),
        ),
    )

    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - start_time

    r_start_h = cp.asnumpy(r_start)
    r_end_h = cp.asnumpy(r_end)
    sum_r_h = cp.asnumpy(sum_r)
    sum_r2_h = cp.asnumpy(sum_r2)
    count_h = cp.asnumpy(count)
    periapsis_h = cp.asnumpy(periapsis_count)
    phi_first_h = cp.asnumpy(phi_first)
    phi_last_h = cp.asnumpy(phi_last)
    r0_h = np.asarray(cp.asnumpy(r0_arr))

    idx0 = 0
    r_hist_first = {
        "r_start": r_start_h[idx0],
        "r_end": r_end_h[idx0],
        "sum_r": sum_r_h[idx0],
        "sum_r2": sum_r2_h[idx0],
        "count": count_h[idx0],
    }
    phi_hist_first = {
        "phi_first": phi_first_h[idx0],
        "phi_last": phi_last_h[idx0],
    }
    summary_first = _streaming_orbit_summary(
        float(r0_h[idx0]),
        dt,
        r_hist_first,
        phi_hist_first,
        int(periapsis_h[idx0]),
    )

    total_active = int((count_h > 0).sum())
    if total_active <= 1:
        summary = summary_first.copy()
    else:
        mask_used = count_h > 0
        r_start_mean = float(r_start_h[mask_used].mean())
        r_end_mean = float(r_end_h[mask_used].mean())
        ratio_mean = r_end_mean / max(r_start_mean, 1e-6)
        mean_r_mean = float((sum_r_h[mask_used] / count_h[mask_used]).mean())
        sigma_mean = float(
            np.sqrt(
                np.maximum(
                    sum_r2_h[mask_used] / count_h[mask_used]
                    - (sum_r_h[mask_used] / count_h[mask_used]) ** 2,
                    0.0,
                )
            ).mean()
        )
        drdt_mean = float(
            (
                (r_end_h[mask_used] - r_start_h[mask_used])
                / (count_h[mask_used] * dt)
            ).mean()
        )
        periapsis_mean = float(periapsis_h[mask_used].mean())

        summary = {
            "r_start": r_start_mean,
            "r_end": r_end_mean,
            "ratio": ratio_mean,
            "mean_r": mean_r_mean,
            "sigma_r": sigma_mean,
            "drdt": drdt_mean,
            "orbit_type": summary_first["orbit_type"],
            "periapsis_count": periapsis_mean,
            "mean_dphi_deg": summary_first["mean_dphi_deg"],
        }

    summary["elapsed"] = elapsed
    summary["batch_size"] = batch_size
    summary["first_orbit"] = summary_first
    return summary


def _integrate_orbits_core(acc_func, G_or_force, vx, vy, vz, r0, z0, steps, dt, leapfrog=False):
    """
    Core batched integrator used by all public entry points.

    acc_func(x, y, z) must return a 5-tuple:
        (ax, ay, az, r, inside_mask)
    where all entries are xp arrays and inside_mask is a boolean mask
    indicating which orbits are still in a valid region.

    vx, vy, vz, r0, z0 can be scalars or 1D arrays (NumPy or xp).
    Returns a dictionary with the same fields as summarize_orbit, plus "elapsed".
    For a batch, the dictionary fields are batch-averaged; the first orbit
    statistics are also returned under the key "first_orbit".
    """
    candidates = []
    for v in (vx, vy, vz, r0, z0):
        if hasattr(v, "shape"):
            arr = np.asarray(v)
            if arr.ndim == 1:
                candidates.append(arr.shape[0])
    batch_size = max(candidates) if candidates else 1

    vx_arr = _prepare_batch(vx, batch_size)
    vy_arr = _prepare_batch(vy, batch_size)
    vz_arr = _prepare_batch(vz, batch_size)
    r0_arr = _prepare_batch(r0, batch_size)
    z0_arr = _prepare_batch(z0, batch_size)

    if hasattr(G_or_force, "shape"):
        N = int(G_or_force.shape[0])
    else:
        N = 256
    cx = 0.5 * (N - 1)
    cy = 0.5 * (N - 1)
    cz = 0.5 * (N - 1)

    cx_arr = xp.asarray(cx, dtype=xp.float64)
    cy_arr = xp.asarray(cy, dtype=xp.float64)
    cz_arr = xp.asarray(cz, dtype=xp.float64)

    x = cx_arr + r0_arr
    y = cy_arr
    z = cz_arr + z0_arr

    vx3 = vx_arr.copy()
    vy3 = vy_arr.copy()
    vz3 = vz_arr.copy()

    sum_r = xp.zeros(batch_size, dtype=xp.float64)
    sum_r2 = xp.zeros(batch_size, dtype=xp.float64)
    count = xp.zeros(batch_size, dtype=xp.int64)

    r_start = xp.zeros(batch_size, dtype=xp.float64)
    r_end = xp.zeros(batch_size, dtype=xp.float64)

    last_r = xp.zeros(batch_size, dtype=xp.float64)
    last_dr = xp.zeros(batch_size, dtype=xp.float64)
    last_dr_defined = xp.zeros(batch_size, dtype=bool)

    periapsis_count = xp.zeros(batch_size, dtype=xp.int64)

    phi_first = xp.zeros(batch_size, dtype=xp.float64)
    phi_last = xp.zeros(batch_size, dtype=xp.float64)
    phi_has_first = xp.zeros(batch_size, dtype=bool)

    active = xp.ones(batch_size, dtype=bool)

    start_time = time.time()

    if leapfrog:
        ax0, ay0, az0, r_init, inside_init = acc_func(x, y, z)
        vx3 = vx3 + 0.5 * dt * ax0
        vy3 = vy3 + 0.5 * dt * ay0
        vz3 = vz3 + 0.5 * dt * az0

    for _ in range(int(steps)):
        if not bool(active.any()):
            break

        x = x + vx3 * dt
        y = y + vy3 * dt
        z = z + vz3 * dt

        out_of_box = (
            (xp.abs(x - cx_arr) > N)
            | (xp.abs(y - cy_arr) > N)
            | (xp.abs(z - cz_arr) > N)
        )

        ax, ay, az, r, inside_mask = acc_func(x, y, z)

        still_ok = (~out_of_box) & inside_mask
        active = active & still_ok

        if not bool(active.any()):
            break

        if leapfrog:
            vx3 = vx3 + dt * ax
            vy3 = vy3 + dt * ay
            vz3 = vz3 + dt * az
        else:
            vx3 = vx3 + dt * ax * MOVEMENT_SPEED_3D
            vy3 = vy3 + dt * ay * MOVEMENT_SPEED_3D
            vz3 = vz3 + dt * az * MOVEMENT_SPEED_3D

        mask = active
        if not bool(mask.any()):
            break

        r_active = r[mask]
        dx = x[mask] - cx_arr
        dy = y[mask] - cy_arr
        phi = xp.arctan2(dy, dx)

        is_first_step = (count[mask] == 0)
        r_start_masked = r_start[mask]
        r_start_masked = xp.where(is_first_step, r_active, r_start_masked)
        r_start[mask] = r_start_masked
        r_end[mask] = r_active

        sum_r[mask] = sum_r[mask] + r_active
        sum_r2[mask] = sum_r2[mask] + r_active * r_active
        count[mask] = count[mask] + 1

        last_r_masked = last_r[mask]
        last_dr_masked = last_dr[mask]
        last_dr_def_masked = last_dr_defined[mask]

        dr = r_active - last_r_masked
        has_last_r = last_r_masked != 0.0
        has_last_dr = last_dr_def_masked

        periapsis = has_last_r & has_last_dr & (last_dr_masked < 0.0) & (dr > 0.0)
        periapsis_count[mask] = periapsis_count[mask] + periapsis.astype(xp.int64)

        last_r[mask] = r_active
        last_dr[mask] = dr
        last_dr_defined[mask] = True

        need_first = (~phi_has_first[mask])
        phi_first_vals = phi_first[mask]
        phi_first_vals = xp.where(need_first, phi, phi_first_vals)
        phi_first[mask] = phi_first_vals
        phi_has_first[mask] = True

        phi_last[mask] = phi

    elapsed = time.time() - start_time

    idx0 = 0
    r_hist_first = {
        "r_start": r_start[idx0],
        "r_end": r_end[idx0],
        "sum_r": sum_r[idx0],
        "sum_r2": sum_r2[idx0],
        "count": count[idx0],
    }
    phi_hist_first = {
        "phi_first": phi_first[idx0],
        "phi_last": phi_last[idx0],
    }
    summary_first = _streaming_orbit_summary(
        float(r0_arr[idx0]),
        dt,
        r_hist_first,
        phi_hist_first,
        int(periapsis_count[idx0]),
    )

    total_active = int((count > 0).sum())
    if total_active <= 1:
        summary = summary_first.copy()
    else:
        mask_used = count > 0
        r_start_mean = float(r_start[mask_used].mean())
        r_end_mean = float(r_end[mask_used].mean())
        ratio_mean = r_end_mean / max(r_start_mean, 1e-6)
        mean_r_mean = float((sum_r[mask_used] / count[mask_used]).mean())
        sigma_mean = float(
            xp.sqrt(
                xp.maximum(
                    sum_r2[mask_used] / count[mask_used]
                    - (sum_r[mask_used] / count[mask_used]) ** 2,
                    0.0,
                )
            ).mean()
        )
        drdt_mean = float(
            ((r_end[mask_used] - r_start[mask_used]) / (count[mask_used] * dt)).mean()
        )
        periapsis_mean = float(periapsis_count[mask_used].mean())

        summary = {
            "r_start": r_start_mean,
            "r_end": r_end_mean,
            "ratio": ratio_mean,
            "mean_r": mean_r_mean,
            "sigma_r": sigma_mean,
            "drdt": drdt_mean,
            "orbit_type": summary_first["orbit_type"],
            "periapsis_count": periapsis_mean,
            "mean_dphi_deg": summary_first["mean_dphi_deg"],
        }

    summary["elapsed"] = elapsed
    summary["batch_size"] = batch_size
    summary["first_orbit"] = summary_first
    return summary


def long_orbit_3d_local(G, vx, vy, vz, r0, z0, steps, dt):
    """
    Public entry point: orbit in a 3D potential grid G.

    GPU:
        If CuPy RawKernel is available, uses a single fused kernel that
        integrates each orbit over all steps on the GPU.

    CPU:
        Falls back to the generic batched integrator using xp backend.
    """
    N = G.shape[0]
    cx = 0.5 * (N - 1)
    cy = 0.5 * (N - 1)
    cz = 0.5 * (N - 1)

    if GPU_AVAILABLE and cp is not None:
        try:
            return _integrate_orbits_core_local_rawkernel(
                G, vx, vy, vz, r0, z0, steps, dt
            )
        except Exception as e:
            print("RawKernel path failed, falling back to generic integrator:", e)

    Gx, Gy, Gz = compute_gradients_3d(G)

    def acc_func(x, y, z):
        # If user requests CPU-trilinear gradient, use conservative phi->grad helper
        import os as _os
        if _os.environ.get("QCF_FORCE_CPU_GRAD","0") == "1":
            return _compute_local_acceleration_from_phi(G, x, y, z, cx, cy, cz)
        return _compute_local_acceleration(Gx, Gy, Gz, x, y, z, cx, cy, cz)

    return _integrate_orbits_core(
        acc_func, G, vx, vy, vz, r0, z0, steps, dt, leapfrog=False
    )


def long_orbit_3d_spherical(force_dGdr, vx, vy, vz, r0, z0, steps, dt):
    """
    Orbit in a spherical potential defined by dG/dr.
    Najpierw probuje GPU RawKernel (jesli force_dGdr ma r_arr/dGdr_arr),
    w przeciwnym wypadku spada do ogolnego integratora XP.
    """
    N = 256
    cx = 0.5 * (N - 1)
    cy = 0.5 * (N - 1)
    cz = 0.5 * (N - 1)

    # fast path: jedna petla czasowa w RawKernelu, bez callbackow do Pythona
    if GPU_AVAILABLE and cp is not None:
        try:
            return _integrate_orbits_core_spherical_rawkernel(
                force_dGdr, vx, vy, vz, r0, z0, steps, dt
            )
        except Exception as e:
            print("Spherical RawKernel path failed, falling back to generic integrator:", e)

    # wolniejsza sciezka, ale funkcjonalnie identyczna
    def acc_func(x, y, z):
        return _compute_spherical_acceleration(force_dGdr, x, y, z, cx, cy, cz)

    return _integrate_orbits_core(
        acc_func, force_dGdr, vx, vy, vz, r0, z0, steps, dt, leapfrog=False
    )



def long_orbit_3d_spherical_leapfrog(force_dGdr, vx, vy, vz, r0, z0, steps, dt):
    """
    Leapfrog variant for the spherical case.
    """
    N = 256
    cx = 0.5 * (N - 1)
    cy = 0.5 * (N - 1)
    cz = 0.5 * (N - 1)

    def acc_func(x, y, z):
        return _compute_spherical_acceleration(force_dGdr, x, y, z, cx, cy, cz)

    return _integrate_orbits_core(
        acc_func, force_dGdr, vx, vy, vz, r0, z0, steps, dt, leapfrog=True
    )
