# simulation.py
# Version ajustée pour Pyodide + progress bar
import numpy as np
import asyncio
import time

# JS callbacks (set_progress, update) — import if available
try:
    from js import set_progress
except Exception:
    def set_progress(_):
        pass

try:
    from js import update
except Exception:
    def update(_):
        pass

# ----- paramètres (tu peux les ajuster) -----
sigma = 1.0
p0 = np.array([1.0, 0.0])
T_final = 20.0
n_image = 80
dt = T_final / max(1, (n_image - 1))

grid_nx = 5
grid_ny = 5
grid_x_min, grid_x_max = -2.0, 2.0
grid_y_min, grid_y_max = -2.0, 2.0

orientation_mode = 'random'
seed = 42
rng = np.random.RandomState(seed)

K_noise = 6
noise_amp = 0.25
L_domain = 10.0
kvecs = []
for kk in range(K_noise):
    angle = 2 * np.pi * kk / max(1, K_noise)
    kvecs.append(np.array([np.cos(angle), np.sin(angle)]) * (2 * np.pi / L_domain))
kvecs = np.array(kvecs)
amps = noise_amp * (1.0 + 0.5 * rng.randn(K_noise))

eps = 1e-10

# Kernels & integrator (kept compact)
def Green_K(pts, sigma=1.0):
    r = np.asarray(pts, dtype=float)
    if r.ndim == 1:
        r = r[None, :]
    r2 = np.sum(r * r, axis=1)
    A = 1.0 / (4.0 * np.pi * sigma**2)
    B = np.exp(-r2 / (4.0 * sigma**2))
    C = 2.0 * sigma**2 / (r2 + eps)
    alpha = B - C * (1.0 - B)
    beta  = (2.0 * C * (1.0 - B) - B) / (r2 + eps)
    I = np.eye(2)
    outer = r[:, :, None] * r[:, None, :]
    G = A * (alpha[:, None, None] * I + beta[:, None, None] * outer)
    return G

def u_au_point(x, qs_list, ps_list, sigma=sigma):
    u = np.zeros(2)
    for q, p in zip(qs_list, ps_list):
        G = Green_K(np.array([x - q]), sigma=sigma)[0]
        u += G.dot(p)
    return u

def jacobian_u_au_point(x, qs_list, ps_list, sigma=sigma, h=1e-5):
    J = np.zeros((2,2))
    for j in range(2):
        e = np.zeros(2); e[j] = 1.0
        u_plus  = u_au_point(x + h*e, qs_list, ps_list, sigma=sigma)
        u_minus = u_au_point(x - h*e, qs_list, ps_list, sigma=sigma)
        J[:, j] = (u_plus - u_minus) / (2.0 * h)
    return J

def deriv(state, N, sigma=sigma):
    qs = [state[2*i:2*i+2] for i in range(N)]
    ps = [state[2*N + 2*i:2*N + 2*i+2] for i in range(N)]
    dq = np.zeros((N,2))
    dp = np.zeros((N,2))
    for a in range(N):
        q_a = qs[a].copy()
        p_a = ps[a].copy()
        u_q = u_au_point(q_a, qs, ps, sigma=sigma)
        J = jacobian_u_au_point(q_a, qs, ps, sigma=sigma, h=1e-5)
        dq[a] = u_q
        dp[a] = - J.T.dot(p_a)
    deriv_vec = np.zeros_like(state)
    for i in range(N):
        deriv_vec[2*i:2*i+2] = dq[i]
    for i in range(N):
        deriv_vec[2*N + 2*i:2*N + 2*i+2] = dp[i]
    return deriv_vec

def rk4_step(state, dt, N, sigma=sigma):
    k1 = deriv(state, N, sigma=sigma)
    k2 = deriv(state + 0.5*dt*k1, N, sigma=sigma)
    k3 = deriv(state + 0.5*dt*k2, N, sigma=sigma)
    k4 = deriv(state + dt*k3, N, sigma=sigma)
    return state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

def sigma_k_at(kidx, qs_array):
    k = kvecs[kidx]
    phase = qs_array.dot(k)
    a = amps[kidx]
    return a * np.stack([-np.sin(phase), np.cos(phase)], axis=-1)

def grad_sigma_k_at(kidx, qs_array):
    k = kvecs[kidx]
    phase = qs_array.dot(k)
    a = amps[kidx]
    M = qs_array.shape[0]
    G = np.zeros((M,2,2))
    G[:,0,0] = -np.cos(phase) * k[0] * a
    G[:,0,1] = -np.cos(phase) * k[1] * a
    G[:,1,0] = -np.sin(phase) * k[0] * a
    G[:,1,1] = -np.sin(phase) * k[1] * a
    return G

def gk_on_state(kidx, state, N):
    qs = np.array([state[2*i:2*i+2] for i in range(N)])
    ps = np.array([state[2*N + 2*i:2*N + 2*i+2] for i in range(N)])
    sig = sigma_k_at(kidx, qs)
    grad_sig = grad_sigma_k_at(kidx, qs)
    dq = sig
    dp = np.zeros_like(ps)
    for i in range(N):
        dp[i] = - grad_sig[i].T.dot(ps[i])
    return np.concatenate([dq.ravel(), dp.ravel()])

# Initialisation
grid_x = np.linspace(grid_x_min, grid_x_max, grid_nx)
grid_y = np.linspace(grid_y_min, grid_y_max, grid_ny)
qs_init = []
ps_init = []

p_magnitude = np.linalg.norm(p0) if np.linalg.norm(p0) > 0 else 1.0

for yi in grid_y:
    for xi in grid_x:
        qs_init.append(np.array([xi, yi]))
        if orientation_mode == 'random':
            angle = rng.uniform(0.0, 2*np.pi)
        elif orientation_mode == 'radial':
            angle = np.arctan2(yi, xi)
        elif orientation_mode == 'inward':
            angle = np.arctan2(yi, xi) + np.pi
        elif orientation_mode == 'tangential':
            angle = np.arctan2(yi, xi) + 0.5*np.pi
        elif orientation_mode == 'vortex':
            angle = np.arctan2(yi, xi) - 0.5*np.pi
        else:
            angle = rng.uniform(0.0, 2*np.pi)

        p_vec = p_magnitude * np.array([np.cos(angle), np.sin(angle)])
        p_vec += 0.0 * rng.randn(2)
        ps_init.append(p_vec)

N = len(qs_init)
state0 = np.zeros(4 * N)
for i in range(N):
    state0[2*i:2*i+2] = qs_init[i]
for i in range(N):
    state0[2*N + 2*i:2*N + 2*i+2] = ps_init[i]

qs_hist = np.zeros((n_image, N, 2))
ps_hist = np.zeros((n_image, N, 2))
u_selfs = np.zeros((n_image, N, 2))

# Control & playback
_computed = False
_computing = None
_playing = False
_play_interval = 0.04

_xmin, _xmax = grid_x_min, grid_x_max
_ymin, _ymax = grid_y_min, grid_y_max
_xrange = _xmax - _xmin if (_xmax - _xmin) != 0 else 1.0
_yrange = _ymax - _ymin if (_ymax - _ymin) != 0 else 1.0

def _make_frame_state(frame_idx, r_rel=0.03):
    frame_idx = int(np.clip(frame_idx, 0, n_image - 1))
    qs_frame = qs_hist[frame_idx]
    out = []
    for a in range(N):
        x = float(qs_frame[a,0])
        y = float(qs_frame[a,1])
        xnorm = (x - _xmin) / _xrange
        ynorm = (y - _ymin) / _yrange
        out.append({'x': xnorm, 'y': ynorm, 'r': r_rel})
    return out

async def _compute_trajectories():
    global _computed, qs_hist, ps_hist, u_selfs
    if _computed:
        return
    state = state0.copy()
    sqrt_dt = np.sqrt(dt)
    t0 = time.time()
    for i in range(n_image):
        for a in range(N):
            qs_hist[i,a] = state[2*a:2*a+2]
            ps_hist[i,a] = state[2*N + 2*a:2*N + 2*a+2]
        for a in range(N):
            u_selfs[i,a] = u_au_point(qs_hist[i,a], list(qs_hist[i]), list(ps_hist[i]), sigma=sigma)
        if i < n_image - 1:
            state = rk4_step(state, dt, N, sigma=sigma)
        # update progress every few frames
        if (i % 2) == 0 or i == n_image-1:
            pct = int(100.0 * (i+1) / n_image)
            try:
                set_progress(pct)
            except Exception:
                pass
        if (i % 6) == 0:
            await asyncio.sleep(0)  # yield to event loop
    _computed = True
    try:
        set_progress(100)
    except Exception:
        pass

async def _playback_loop():
    global _playing
    while _playing:
        if not _computed:
            await asyncio.sleep(0.02)
            continue
        for i in range(n_image):
            if not _playing:
                break
            state = _make_frame_state(i)
            try:
                update(state)
            except Exception:
                pass
            await asyncio.sleep(_play_interval)

def start(interval_seconds=None):
    global _playing, _play_interval
    if interval_seconds is not None:
        _play_interval = float(interval_seconds)
    if _playing:
        return
    _playing = True
    # schedule compute & playback
    try:
        asyncio.ensure_future(_compute_trajectories())
        asyncio.ensure_future(_playback_loop())
    except Exception:
        # fallback synchronous compute (not ideal)
        pass

def stop():
    global _playing
    _playing = False
    try:
        set_progress(0)
    except Exception:
        pass

def set_interval(seconds):
    global _play_interval
    _play_interval = float(seconds)
