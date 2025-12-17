import math
import numpy as np

try:
    import cupy as cp

    xp = cp
    GPU_AVAILABLE = True
except ImportError:
    xp = np
    GPU_AVAILABLE = True
WAVE_STEP_RAW_KERNEL = None
if GPU_AVAILABLE:
    try:
        _wave_step_kernel_src = r'''
        extern "C" __global__
        void wave_step_kernel(const float* m, const float* v, float* m_out, float* v_out, const int N, const float dt, const float c2, const float gamma, const float mass2) {

            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            int k = blockIdx.z * blockDim.z + threadIdx.z;
            if (i >= N || j >= N || k >= N) return;

            int idx = (i * N + j) * N + k;

            // periodic indices
            int ip = (i + 1 == N) ? 0 : i + 1;
            int im = (i == 0) ? (N - 1) : i - 1;
            int jp = (j + 1 == N) ? 0 : j + 1;
            int jm = (j == 0) ? (N - 1) : j - 1;
            int kp = (k + 1 == N) ? 0 : k + 1;
            int km = (k == 0) ? (N - 1) : k - 1;

            int idx_ip = (ip * N + j) * N + k;
            int idx_im = (im * N + j) * N + k;
            int idx_jp = (i * N + jp) * N + k;
            int idx_jm = (i * N + jm) * N + k;
            int idx_kp = (i * N + j) * N + kp;
            int idx_km = (i * N + j) * N + km;

            float m_c = m[idx];
            float lap = m[idx_ip] + m[idx_im] + m[idx_jp] + m[idx_jm] + m[idx_kp] + m[idx_km] - 6.0f * m_c;

            float v_old = v[idx];
            float v_new = v_old + dt * (c2 * lap - gamma * v_old - mass2 * m_c);
            float m_new = m_c + dt * v_new;

            v_out[idx] = v_new;
            m_out[idx] = m_new;
        }
        '''
        WAVE_STEP_RAW_KERNEL = cp.RawKernel(_wave_step_kernel_src, 'wave_step_kernel')
    except Exception:
        # If RawKernel compilation fails, leave as None and fallback to roll-based implementation.
        WAVE_STEP_RAW_KERNEL = None
# ---  END  ---



PURE_MASS_MODE = False #True
GRAVITY_INERTIA_3D = 1.0
MOVEMENT_SPEED_3D = 1.0


class QCFConfig:
    def __init__(
        self,
        mass1_ampl=1.0,
        mass1_sigma=2.0,
        mass3_ampl=1.0,
        mass3_sigma=3.0,
        e3_coupling_qcf=0.5,
        e3_noise_amp=0.2,
        relax_steps=16000,
        w_layers=1,
        w_coupling=0.0,
        iso_damp=0.0,      # juz masz
        grad_penalty=0.0,  # NEW: kara za gradienty halo
        u1_coupling=0.0,
        u1_self_lambda=0.0
    ):
        self.mass1_ampl = mass1_ampl
        self.mass1_sigma = mass1_sigma
        self.mass3_ampl = mass3_ampl
        self.mass3_sigma = mass3_sigma
        self.e3_coupling_qcf = e3_coupling_qcf
        self.e3_noise_amp = e3_noise_amp
        self.relax_steps = relax_steps

        self.w_layers = int(w_layers)
        if self.w_layers < 1:
            self.w_layers = 1
        self.w_coupling = float(w_coupling)

        self.iso_damp = float(iso_damp)
        self.u1_coupling = float(u1_coupling)
        self.u1_self_lambda = float(u1_self_lambda)



class QCFField3D:
    def __init__(self, N, cfg: QCFConfig):
        self.N = N
        self.cfg = cfg
        self.xp = xp

        # dynamic U(1) field
        self.mag_x = xp.zeros((N, N, N), dtype=xp.float32)
        self.mag_y = xp.zeros((N, N, N), dtype=xp.float32)

        self.vel_x = xp.zeros((N, N, N), dtype=xp.float32)
        self.vel_y = xp.zeros((N, N, N), dtype=xp.float32)

        # static sources
        self.mass1 = xp.zeros((N, N, N), dtype=xp.float32)
        self.mass3 = xp.zeros((N, N, N), dtype=xp.float32)

        self._build_static_masses()

    def apply_global_phase(self, alpha):
        # rotate entire complex order parameter
        phase = np.exp(1j * alpha)
        self.mag_x[...] = np.real((self.mag_x + 1j * self.mag_y) * phase)
        self.mag_y[...] = np.imag((self.mag_x + 1j * self.mag_y) * phase)

    def compute_local_energy_density(self, zslice=None):
        """
        Local energy density ~ |grad Phi|^2 + V(|Phi|)
        Minimal classical EFT definition.
        """
        xp = self.xp
        if zslice is None:
            zslice = self.N // 2

        mx = self.mag_x[:, :, zslice]
        my = self.mag_y[:, :, zslice]

        # gradients
        dx_mx = xp.roll(mx, -1, axis=0) - mx
        dy_mx = xp.roll(mx, -1, axis=1) - mx
        dx_my = xp.roll(my, -1, axis=0) - my
        dy_my = xp.roll(my, -1, axis=1) - my

        grad2 = dx_mx**2 + dy_mx**2 + dx_my**2 + dy_my**2
        pot = (mx**2 + my**2 - 1.0)**2

        return grad2 + pot


    def seed_vortex(self, amp=0.2, r0=6.0):
        # Seed a single U(1) vortex in the x-y plane, centered at grid center
        from qcf_core import xp
        N = self.N
        cx = cy = cz = N // 2

        x = xp.arange(N) - cx
        y = xp.arange(N) - cy
        X, Y = xp.meshgrid(x, y, indexing="ij")
        R = xp.sqrt(X*X + Y*Y) + 1e-6
        Phi = xp.arctan2(Y, X)

        # radial profile (Gaussian core)
        A = amp * xp.exp(-(R*R) / (2.0 * r0 * r0))

        # initialize vortex: mag_x + i mag_y
        for z in range(N):
            self.mag_x[:, :, z] = A * xp.cos(Phi)
            self.mag_y[:, :, z] = A * xp.sin(Phi)

        # zero velocities
        self.vel_x[...] = 0.0
        self.vel_y[...] = 0.0

    def seed_double_vortex(self, amp=0.2, r0=6.0, separation=8.0, orientation=-1):
        xp = self.xp
        N = self.N
        cx = cy = N // 2

        x = xp.arange(N) - cx
        y = xp.arange(N) - cy
        X, Y = xp.meshgrid(x, y, indexing="ij")

        dx = separation / 2.0

        # vortex A
        R1 = xp.sqrt((X - dx)**2 + Y**2) + 1e-6
        Phi1 = xp.arctan2(Y, X - dx)
        A1 = amp * xp.exp(-(R1*R1)/(2*r0*r0))

        # vortex B (opposite orientation)
        R2 = xp.sqrt((X + dx)**2 + Y**2) + 1e-6
        Phi2 = xp.arctan2(Y, X + dx)
        A2 = amp * xp.exp(-(R2*R2)/(2*r0*r0))

        mx = A1 * xp.cos(Phi1) + orientation * A2 * xp.cos(Phi2)
        my = A1 * xp.sin(Phi1) + orientation * A2 * xp.sin(Phi2)

        for z in range(N):
            self.mag_x[:, :, z] = mx
            self.mag_y[:, :, z] = my

        self.vel_x[...] = 0.0
        self.vel_y[...] = 0.0



    def _build_static_masses(self):
        N = self.N
        cfg = self.cfg
        x = xp.arange(N, dtype=xp.float32)
        y = xp.arange(N, dtype=xp.float32)
        z = xp.arange(N, dtype=xp.float32)
        X, Y, Z = xp.meshgrid(x, y, z, indexing="ij")

        cx = 0.5 * (N - 1)
        cy = 0.5 * (N - 1)
        cz = 0.5 * (N - 1)

        r2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2

        if cfg.mass1_sigma > 0.0:
            self.mass1 = cfg.mass1_ampl * xp.exp(-0.5 * r2 / (cfg.mass1_sigma ** 2))
        else:
            self.mass1[...] = 0.0

        if cfg.mass3_sigma > 0.0:
            self.mass3 = cfg.mass3_ampl * xp.exp(-0.5 * r2 / (cfg.mass3_sigma ** 2))
        else:
            self.mass3[...] = 0.0

        self.mag_x[...] = 0.0
        self.mag_y[...] = 0.0


    def run_coherence3D(self):
        cfg = self.cfg
        N = self.N

        # base QCF noise injection
        noise = cfg.e3_noise_amp * (
            xp.random.random((N, N, N)).astype(xp.float32) - 0.5
        )
        self.mag_x += noise
        self.mag_y += noise


        # NEW: optional isotropic damping in (x,y,z) 
        k = getattr(cfg, "iso_damp", 0.0)
        if k != 0.0:
            # neighbor average in 6 directions (von Neumann neighborhood) ala ma kota
            for comp in ("mag_x", "mag_y"):
                m = getattr(self, comp)
                neigh_sum = (
                    xp.roll(m, 1, axis=0) + xp.roll(m, -1, axis=0) +
                    xp.roll(m, 1, axis=1) + xp.roll(m, -1, axis=1) +
                    xp.roll(m, 1, axis=2) + xp.roll(m, -1, axis=2)
                )
                neigh_avg = neigh_sum / 6.0
                setattr(self, comp, (1.0 - k) * m + k * neigh_avg)

        # global damping, as before
        self.mag_x *= 1.0 - cfg.e3_coupling_qcf
        self.mag_y *= 1.0 - cfg.e3_coupling_qcf


    def run_wave_step3D(self, dt=1.0, c=0.1, gamma=0.0, mass=0.0):
        xp = self.xp

        dt = float(dt)
        c = 0.0 if c is None else float(c)
        gamma = 0.0 if gamma is None else float(gamma)
        mass2 = 0.0 if mass is None else float(mass) * float(mass)

        mx = self.mag_x
        my = self.mag_y
        vx = self.vel_x
        vy = self.vel_y

        # =========================
        # GPU FAST PATH
        # =========================
        if GPU_AVAILABLE and WAVE_STEP_RAW_KERNEL is not None:
            N = int(self.N)

            m_out = xp.empty_like(mx)
            v_out = xp.empty_like(vx)

            tpx = tpy = tpz = 8
            bx = (N + tpx - 1) // tpx
            by = (N + tpy - 1) // tpy
            bz = (N + tpz - 1) // tpz

            WAVE_STEP_RAW_KERNEL(
                (bx, by, bz),
                (tpx, tpy, tpz),
                (
                    mx,
                    vx,
                    m_out,
                    v_out,
                    xp.int32(N),
                    xp.float32(dt),
                    xp.float32(c * c),
                    xp.float32(gamma),
                    xp.float32(mass2),
                ),
            )

            self.mag_x = m_out
            self.vel_x = v_out
            return

        # =========================
        # CPU FALLBACK
        # =========================
        def lap(f):
            return (
                xp.roll(f, 1, 0) + xp.roll(f, -1, 0) +
                xp.roll(f, 1, 1) + xp.roll(f, -1, 1) +
                xp.roll(f, 1, 2) + xp.roll(f, -1, 2) -
                6.0 * f
            )

        mx_new = mx + dt * vx
        my_new = my + dt * vy

        vx_new = vx + dt * ((c * c) * lap(mx_new) - gamma * vx - mass2 * mx_new)
        vy_new = vy + dt * ((c * c) * lap(my_new) - gamma * vy - mass2 * my_new)

        self.mag_x = mx_new
        self.mag_y = my_new
        self.vel_x = vx_new
        self.vel_y = vy_new


    def get_magnitude3D(self):
        return xp.sqrt(self.mag_x**2 + self.mag_y**2)

    def compute_phase_xy(self, zslice=None):
        xp = self.xp
        if zslice is None:
            zslice = self.N // 2
        mx = self.mag_x[:, :, zslice]
        my = self.mag_y[:, :, zslice]

        phase = xp.arctan2(my, mx)
        return phase

    def probe_phase_time_series(self, center, radius=2, zslice=None):
        """
        Measure averaged U(1) phase around a vortex core
        as a function of time. This observable oscillates
        if the vortex has a mass (internal mode).
        """
        xp = self.xp
        if zslice is None:
            zslice = self.N // 2

        cx, cy = center
        cx = int(cx)
        cy = int(cy)

        phase = self.compute_phase_xy(zslice)

        vals = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x = cx + dx
                y = cy + dy
                if 0 <= x < self.N and 0 <= y < self.N:
                    vals.append(phase[x, y])

        if not vals:
            return 0.0

        # average phase (unwrap-safe)
        vals = xp.array(vals)
        return float(xp.angle(xp.mean(xp.exp(1j * vals))))

    def probe_amplitude_time_series(self, center, r_inner=2, r_outer=4, zslice=None):
        """
        Measure averaged field amplitude |Phi| on a RING
        around a vortex core (massive Higgs-like mode).
        """
        xp = self.xp
        if zslice is None:
            zslice = self.N // 2

        cx, cy = center
        cx = int(cx)
        cy = int(cy)

        mx = self.mag_x[:, :, zslice]
        my = self.mag_y[:, :, zslice]

        vals = []
        for dx in range(-r_outer, r_outer + 1):
            for dy in range(-r_outer, r_outer + 1):
                r2 = dx*dx + dy*dy
                if r_inner*r_inner <= r2 <= r_outer*r_outer:
                    x = cx + dx
                    y = cy + dy
                    if 0 <= x < self.N and 0 <= y < self.N:
                        vals.append(xp.sqrt(mx[x, y]**2 + my[x, y]**2))

        if not vals:
            return 0.0

        vals = xp.array(vals)
        return float(vals.mean())


    def compute_winding_xy(self):
        xp = self.xp
        phase = self.compute_phase_xy()

        dpx = xp.diff(phase, axis=0)
        dpy = xp.diff(phase, axis=1)

        dpx = (dpx + xp.pi) % (2 * xp.pi) - xp.pi
        dpy = (dpy + xp.pi) % (2 * xp.pi) - xp.pi

        circulation = xp.sum(dpx[:, :-1] + dpy[:-1, :])
        winding = circulation / (2 * xp.pi)
        return float(winding)

    def compute_local_winding_xy(self, x0=None, y0=None, zslice=None):
        xp = self.xp
        if zslice is None:
            zslice = self.N // 2

        phase = self.compute_phase_xy(zslice)

        if x0 is None:
            x0 = self.N // 2
        if y0 is None:
            y0 = self.N // 2

        p00 = phase[x0, y0]
        p10 = phase[x0 + 1, y0]
        p11 = phase[x0 + 1, y0 + 1]
        p01 = phase[x0, y0 + 1]

        def dphi(a, b):
            d = b - a
            return (d + xp.pi) % (2 * xp.pi) - xp.pi

        circulation = (
            dphi(p00, p10) +
            dphi(p10, p11) +
            dphi(p11, p01) +
            dphi(p01, p00)
        )

        return float(circulation / (2 * xp.pi))
    
    def compute_total_winding_xy(self, zslice=None):
        """
        Compute TOTAL U(1) winding number in XY plane
        by summing ONLY quantized plaquette windings
        (vortex cores). This is the true topological charge.
        """
        xp = self.xp
        if zslice is None:
            zslice = self.N // 2

        phase = self.compute_phase_xy(zslice)

        total = 0
        for i in range(self.N - 1):
            for j in range(self.N - 1):
                p00 = phase[i, j]
                p10 = phase[i + 1, j]
                p11 = phase[i + 1, j + 1]
                p01 = phase[i, j + 1]

                def dphi(a, b):
                    d = b - a
                    return (d + xp.pi) % (2 * xp.pi) - xp.pi

                circulation = (
                    dphi(p00, p10) +
                    dphi(p10, p11) +
                    dphi(p11, p01) +
                    dphi(p01, p00)
                ) / (2 * xp.pi)

                # convert GPU/NumPy scalar -> Python float
                c = float(circulation)

                if abs(c) > 0.5:
                    total += int(round(c))

        return int(total)

    def compute_vortex_winding_xy(self, centers, zslice=None, radius=3):
        """
        Compute total U(1) winding by summing LOCAL windings
        around known vortex cores only.
        This is the correct topological charge.
        """
        xp = self.xp
        if zslice is None:
            zslice = self.N // 2

        phase = self.compute_phase_xy(zslice)

        total = 0
        for (cx, cy) in centers:
            cx = int(cx)
            cy = int(cy)

            wloc = 0.0
            for dx in range(-radius, radius):
                for dy in range(-radius, radius):
                    x = cx + dx
                    y = cy + dy
                    if x < 0 or y < 0 or x + 1 >= self.N or y + 1 >= self.N:
                        continue

                    p00 = phase[x, y]
                    p10 = phase[x + 1, y]
                    p11 = phase[x + 1, y + 1]
                    p01 = phase[x, y + 1]

                    def dphi(a, b):
                        d = b - a
                        return (d + xp.pi) % (2 * xp.pi) - xp.pi

                    circulation = (
                        dphi(p00, p10) +
                        dphi(p10, p11) +
                        dphi(p11, p01) +
                        dphi(p01, p00)
                    ) / (2 * xp.pi)

                    wloc += float(circulation)

            total += int(round(wloc))

        return int(total)
    

    def compute_u1_energy(self):
        xp = self.xp
        mx = self.mag[..., 0]
        my = self.mag[..., 1]

        dx = xp.roll(mx, -1, axis=0) - mx
        dy = xp.roll(mx, -1, axis=1) - mx
        dz = xp.roll(mx, -1, axis=2) - mx

        ex = dx*dx + dy*dy + dz*dz

        dx = xp.roll(my, -1, axis=0) - my
        dy = xp.roll(my, -1, axis=1) - my
        dz = xp.roll(my, -1, axis=2) - my

        ey = dx*dx + dy*dy + dz*dz

        return float(xp.mean(ex + ey))
    
    def find_energy_peak_xy(self):
        energy = self.compute_local_energy_density()

        # FIX: handle both 2D and 3D energy arrays
        if energy.ndim == 3:
            energy_xy = xp.mean(energy, axis=2)
        elif energy.ndim == 2:
            energy_xy = energy
        else:
            raise RuntimeError(f"Unexpected energy ndim = {energy.ndim}")

        idx = xp.argmax(energy_xy)
        ex, ey = xp.unravel_index(idx, energy_xy.shape)

        if GPU_AVAILABLE:
            ex = int(ex.get())
            ey = int(ey.get())

        return ex, ey




class QCFField4D:
    """
    Simple 4D QCF field: extra discrete dimension w.

    mag has shape (L, N, N, N), where L = w_layers.
    Static masses (mass1, mass3) are copied from 3D field
    into each w-slice.
    """

    def __init__(self, N, cfg: QCFConfig):
        self.N = N
        self.cfg = cfg
        self.w_layers = max(1, int(getattr(cfg, "w_layers", 1)))
        self.w_coupling = float(getattr(cfg, "w_coupling", 0.1))

        # Allocate 4D magnitude
        self.mag = xp.zeros((self.w_layers, N, N, N), dtype=xp.float32)

        # Build 3D static masses once, then broadcast along w
        base3d = QCFField3D(N, cfg)
        m1_3d = base3d.mass1
        m3_3d = base3d.mass3

        self.mass1 = xp.broadcast_to(m1_3d, self.mag.shape)
        self.mass3 = xp.broadcast_to(m3_3d, self.mag.shape)

    def run_coherence4D(self):
        """
        4D coherence step:
          - noise in all (w,x,y,z)
          - diffusion along w (coupling between slices)
          - optional isotropic diffusion in (x,y,z)
          - damping like in 3D via e3_coupling_qcf
        """
        cfg = self.cfg
        L = self.w_layers
        N = self.N

        # noise in 4D
        noise = cfg.e3_noise_amp * (
            xp.random.random((L, N, N, N)).astype(xp.float32) - 0.5
        )
        self.mag += noise

        # diffusion along w dimension
        if L > 1 and self.w_coupling != 0.0:
            lap_w = xp.zeros_like(self.mag)

            # interior layers
            lap_w[1:-1, :, :, :] = (
                self.mag[0:-2, :, :, :]
                + self.mag[2:, :, :, :]
                - 2.0 * self.mag[1:-1, :, :, :]
            )

            # simple one-sided boundaries
            lap_w[0, :, :, :] = self.mag[1, :, :, :] - self.mag[0, :, :, :]
            lap_w[-1, :, :, :] = self.mag[-2, :, :, :] - self.mag[-1, :, :, :]

            self.mag += self.w_coupling * lap_w

        # NEW: isotropic diffusion in (x,y,z) within each w-slice
        if self.w_coupling != 0.0:
            lam_xyz = 0.1 * self.w_coupling
            lap_xyz = (
                xp.roll(self.mag, 1, axis=1) + xp.roll(self.mag, -1, axis=1) +
                xp.roll(self.mag, 1, axis=2) + xp.roll(self.mag, -1, axis=2) +
                xp.roll(self.mag, 1, axis=3) + xp.roll(self.mag, -1, axis=3) -
                6.0 * self.mag
            )
            self.mag += lam_xyz * lap_xyz

        # global damping, same as 3D
        self.mag *= (1.0 - cfg.e3_coupling_qcf)


    def get_magnitude4D(self):
        return xp.abs(self.mag)

    def project_to_3d(self):
        """
        Effective 3D magnitude: average over w dimension.
        """
        return xp.mean(self.get_magnitude4D(), axis=0)

def solve_poisson_3d(rho, dx=1.0):
    N = rho.shape[0]
    assert rho.shape == (N, N, N)

    rho_hat = xp.fft.fftn(rho)
    kx = xp.fft.fftfreq(N, d=dx) * 2.0 * math.pi
    ky = xp.fft.fftfreq(N, d=dx) * 2.0 * math.pi
    kz = xp.fft.fftfreq(N, d=dx) * 2.0 * math.pi
    KX, KY, KZ = xp.meshgrid(kx, ky, kz, indexing="ij")
    k2 = KX ** 2 + KY ** 2 + KZ ** 2

    green = xp.zeros_like(k2, dtype=xp.complex64)
    mask = k2 > 0
    green[mask] = -1.0 / k2[mask]

    G_hat = green * rho_hat
    G = xp.fft.ifftn(G_hat).real
    return G.astype(xp.float32)


def build_rho_3d(N, cfg: QCFConfig, pure_mass: bool):
    qcf = QCFField3D(N, cfg)
    if pure_mass:
        return qcf.mass3
    else:
        for _ in range(max(1, cfg.relax_steps // 800)):
            qcf.run_coherence3D()
        mag = qcf.get_magnitude3D()
        mag_mean = mag.mean()
        halo = cfg.e3_coupling_qcf * (mag - mag_mean)

        # NEW: local gradient penalty smoothing on halo
        gp = float(getattr(cfg, "grad_penalty", 0.0))
        if gp != 0.0:
            h = halo
            lap = (
                xp.roll(h, 1, axis=0) + xp.roll(h, -1, axis=0) +
                xp.roll(h, 1, axis=1) + xp.roll(h, -1, axis=1) +
                xp.roll(h, 1, axis=2) + xp.roll(h, -1, axis=2) -
                6.0 * h
            )
            halo = h + gp * lap

        rho = qcf.mass3 + halo
        return rho


def build_rho_3d_from_4d(N, cfg: QCFConfig, pure_mass: bool):
    """
    Build effective 3D density from a 4D QCF field.
    Uses w-averaged |mag| as the halo contribution.
    """
    qcf4 = QCFField4D(N, cfg)

    # static mass: average mass3 over w (identical slices)
    mass3_3d = xp.mean(qcf4.mass3, axis=0)

    if pure_mass:
        return mass3_3d

    # number of coherence steps in 4D
    steps = max(1, cfg.relax_steps // 800)
    for _ in range(steps):
        qcf4.run_coherence4D()

    mag3 = qcf4.project_to_3d()
    mag_mean = float(xp.mean(mag3))
    halo = cfg.e3_coupling_qcf * (mag3 - mag_mean)

    rho = mass3_3d + halo
    return rho


def build_G_field_3d_from_4d(N, cfg: QCFConfig, pure_mass: bool):
    """
    Build 3D potential G from a 4D QCF field projected to 3D.
    """
    rho = build_rho_3d_from_4d(N, cfg, pure_mass)
    rho = rho - xp.mean(rho)
    G = solve_poisson_3d(rho, dx=1.0)
    return G

def build_G_dispatch(N, cfg: QCFConfig, pure_mass: bool, use_4d: bool):
    """
    Helper: choose between 3D QCF and 4D-projected-to-3D.
    """
    if use_4d and getattr(cfg, "w_layers", 1) > 1:
        return build_G_field_3d_from_4d(N, cfg, pure_mass)
    else:
        return build_G_field_3d(N, cfg, pure_mass)

def build_G_field_3d(N, cfg: QCFConfig, pure_mass: bool):
    rho = build_rho_3d(N, cfg, pure_mass)
    rho = rho - rho.mean()
    G = solve_poisson_3d(rho, dx=1.0)
    return G


def compute_gradients_3d(G):
    """
    Compute 3D gradients of the potential G on the grid using central differences.
    Returns three arrays Gx, Gy, Gz with the same shape as G.
    """
    N = G.shape[0]
    Gx = xp.zeros_like(G, dtype=xp.float32)
    Gy = xp.zeros_like(G, dtype=xp.float32)
    Gz = xp.zeros_like(G, dtype=xp.float32)

    # Central differences in x
    Gx[1:-1, :, :] = 0.5 * (G[2:, :, :] - G[:-2, :, :])
    Gx[0, :, :] = Gx[1, :, :]
    Gx[-1, :, :] = Gx[-2, :, :]

    # Central differences in y
    Gy[:, 1:-1, :] = 0.5 * (G[:, 2:, :] - G[:, :-2, :])
    Gy[:, 0, :] = Gy[:, 1, :]
    Gy[:, -1, :] = Gy[:, -2, :]

    # Central differences in z
    Gz[:, :, 1:-1] = 0.5 * (G[:, :, 2:] - G[:, :, :-2])
    Gz[:, :, 0] = Gz[:, :, 1]
    Gz[:, :, -1] = Gz[:, :, -2]

    return Gx, Gy, Gz


def spherical_radial_profile(G, r_min=4.5, r_max=31.5, dr=2.0):
    N = G.shape[0]
    x = xp.arange(N, dtype=xp.float32)
    y = xp.arange(N, dtype=xp.float32)
    z = xp.arange(N, dtype=xp.float32)
    X, Y, Z = xp.meshgrid(x, y, z, indexing="ij")

    cx = 0.5 * (N - 1)
    cy = 0.5 * (N - 1)
    cz = 0.5 * (N - 1)

    r = xp.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)

    nbins = int((r_max - r_min) / dr)
    radii = []
    means = []

    for i in range(nbins):
        r_lo = r_min + i * dr
        r_hi = r_lo + dr
        shell = xp.logical_and(r >= r_lo, r < r_hi)
        count = int(shell.sum())
        if count > 0:
            mean_val = float(G[shell].mean())
            radii.append(0.5 * (r_lo + r_hi))
            means.append(mean_val)

    return xp.array(radii, dtype=xp.float32), xp.array(means, dtype=xp.float32)


def fit_kepler_like(radii, G_r):
    x = xp.array(1.0 / radii, dtype=xp.float64)
    y = xp.array(G_r, dtype=xp.float64)
    A = xp.vstack([x, xp.ones_like(x)]).T
    sol, _, _, _ = xp.linalg.lstsq(A, y, rcond=None)
    a, b = sol

    y_fit = a * x + b
    ss_res = float(((y - y_fit) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(a), float(b), r2


def compute_radial_derivative_profile(G, r_min=5.0, r_max=31.0, dr=1.0):
    radii, G_r = spherical_radial_profile(G, r_min=r_min, r_max=r_max, dr=dr)
    if GPU_AVAILABLE:
        r = np.asarray(cp.asnumpy(radii))
        g = np.asarray(cp.asnumpy(G_r))
    else:
        r = np.asarray(radii)
        g = np.asarray(G_r)

    n = len(r)
    if n < 2:
        raise RuntimeError("Not enough radial bins to compute derivative.")

    dGdr = np.zeros_like(g)
    for i in range(1, n - 1):
        dGdr[i] = (g[i + 1] - g[i - 1]) / (r[i + 1] - r[i - 1])
    dGdr[0] = (g[1] - g[0]) / (r[1] - r[0])
    dGdr[-1] = (g[-1] - g[-2]) / (r[-1] - r[-2])

    return r, g, dGdr


def make_force_dGdr_from_profile(r_arr, dGdr_arr):
    r_arr = np.asarray(r_arr, dtype=float)
    dGdr_arr = np.asarray(dGdr_arr, dtype=float)

    def force_dGdr(r):
        if r <= r_arr[0]:
            return dGdr_arr[0]
        if r >= r_arr[-1]:
            return dGdr_arr[-1]
        idx = np.searchsorted(r_arr, r) - 1
        if idx < 0:
            idx = 0
        if idx >= len(r_arr) - 1:
            idx = len(r_arr) - 2
        r0 = r_arr[idx]
        r1 = r_arr[idx + 1]
        f0 = dGdr_arr[idx]
        f1 = dGdr_arr[idx + 1]
        t = (r - r0) / (r1 - r0)
        return f0 + t * (f1 - f0)

    # dla GPU-owego integratora sferycznego
    force_dGdr.r_arr = r_arr
    force_dGdr.dGdr_arr = dGdr_arr

    return force_dGdr

def make_force_dGdr_kepler(a_fit):
    """
    Analytic dG/dr for a pure Kepler potential V(r) = a/r.
    Returns dG/dr, not the radial acceleration.
    """
    def force_dGdr(r):
        r_safe = float(r)
        if r_safe < 1e-6:
            r_safe = 1e-6
        inv_r = 1.0 / r_safe
        inv_r2 = inv_r * inv_r
        # dV/dr = -a / r^2
        return -a_fit * inv_r2

    return force_dGdr


def make_force_dGdr_geom(a_fit, c2, c3):
    """
    Analytic dG/dr for V_eff(r) = a/r + c2/r^2 + c3/r^3.
    Returns dG/dr, not the radial acceleration.
    """
    def force_dGdr(r):
        r_safe = float(r)
        if r_safe < 1e-6:
            r_safe = 1e-6
        inv_r = 1.0 / r_safe
        inv_r2 = inv_r * inv_r
        inv_r3 = inv_r2 * inv_r
        inv_r4 = inv_r3 * inv_r
        # dV/dr = -a/r^2 - 2 c2 / r^3 - 3 c3 / r^4
        return -a_fit * inv_r2 - 2.0 * c2 * inv_r3 - 3.0 * c3 * inv_r4

    return force_dGdr


def compute_vc_curve_3d(G, r_list):
    N = G.shape[0]
    cx = 0.5 * (N - 1)
    cy = 0.5 * (N - 1)
    cz = 0.5 * (N - 1)

    x = xp.arange(N, dtype=xp.float32)
    y = xp.arange(N, dtype=xp.float32)
    z = xp.arange(N, dtype=xp.float32)
    X, Y, Z = xp.meshgrid(x, y, z, indexing="ij")

    dx = X - cx
    dy = Y - cy
    dz = Z - cz
    r = xp.sqrt(dx ** 2 + dy ** 2 + dz ** 2) + 1e-6

    dGdr = xp.zeros_like(G, dtype=xp.float32)

    for axis in range(3):
        if axis == 0:
            grad = (xp.roll(G, -1, axis=0) - xp.roll(G, 1, axis=0)) * 0.5
            dGdr += grad * (dx / r)
        elif axis == 1:
            grad = (xp.roll(G, -1, axis=1) - xp.roll(G, 1, axis=1)) * 0.5
            dGdr += grad * (dy / r)
        else:
            grad = (xp.roll(G, -1, axis=2) - xp.roll(G, 1, axis=2)) * 0.5
            dGdr += grad * (dz / r)

    results = []
    for r0 in r_list:
        shell = xp.logical_and(r >= (r0 - 0.5), r < (r0 + 0.5))
        count = int(shell.sum())
        if count == 0:
            results.append((r0, None, None, None))
            continue
        G_mean = float(G[shell].mean())
        dGdr_mean = float(dGdr[shell].mean())

        # v_c^2 / r = -a_r = dG/dr
        if dGdr_mean > 0.0:
            vc = math.sqrt(dGdr_mean * r0)
        else:
            vc = 0.0

        results.append((r0, G_mean, dGdr_mean, vc))

    return results


def anisotropy_3d(G, radii, dr=0.5):
    N = G.shape[0]
    cx = 0.5 * (N - 1)
    cy = 0.5 * (N - 1)
    cz = 0.5 * (N - 1)

    x = xp.arange(N, dtype=xp.float32)
    y = xp.arange(N, dtype=xp.float32)
    z = xp.arange(N, dtype=xp.float32)
    X, Y, Z = xp.meshgrid(x, y, z, indexing="ij")
    r = xp.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)

    print("============================================================")
    print(" 3D angular anisotropy diagnostics for G(r)")
    print(" shell half-width dr = %.1f" % dr)
    print(" radii:", list(radii))
    print(" PURE_MASS_MODE =", PURE_MASS_MODE)
    print("============================================================")
    for r0 in radii:
        shell = xp.logical_and(r >= (r0 - dr), r < (r0 + dr))
        count = int(shell.sum())
        if count == 0:
            continue
        vals = G[shell]
        mean_val = float(vals.mean())
        std_val = float(vals.std())
        rel_std = std_val / abs(mean_val) if abs(mean_val) > 1e-8 else 0.0
        print(
            " r=%.2f | mean G=%+e, std=%+e, rel_std=%.3f, n=%d"
            % (r0, mean_val, std_val, rel_std, count)
        )
    print("============================================================")


# ============================================================
# CP1 FIELD (SAFE ADD-ON, DOES NOT TOUCH QCF)
# ============================================================

class CP1Field:
    def __init__(self, N):
        import numpy as np
        self.N = N
        self.z1x = np.ones((N,N,N), dtype=float)
        self.z1y = np.zeros((N,N,N), dtype=float)
        self.z2x = np.zeros((N,N,N), dtype=float)
        self.z2y = np.zeros((N,N,N), dtype=float)
        self.normalize()

    def normalize(self):
        import numpy as np
        rho = np.sqrt(
            self.z1x**2 + self.z1y**2 +
            self.z2x**2 + self.z2y**2 + 1e-12
        )
        self.z1x /= rho; self.z1y /= rho
        self.z2x /= rho; self.z2y /= rho

    def step(self, omega=1e-3):
        import numpy as np
        # simple internal SU(2) precession
        th = omega
        c = np.cos(th)
        s = np.sin(th)

        z1x = c*self.z1x - s*self.z1y
        z1y = s*self.z1x + c*self.z1y

        self.z1x = z1x
        self.z1y = z1y
        self.normalize()

    def global_phase(self):
        import numpy as np
        return np.arctan2(self.z1y.mean(), self.z1x.mean())

def spectral_evolution(eigs, steps=10, forward=True):
    """
    Simple spectral time evolution for testing arrow of time.
    """
    amps = [1.0]
    lam = np.max(np.abs(eigs))

    for _ in range(steps):
        if forward:
            amps.append(amps[-1] * lam)
        else:
            amps.append(amps[-1] / lam)

    return amps