# simulation.py
# Version adaptée pour Pyodide (navigateur)
# Modifications limitées : imports sous try/except, suppression du backend TkAgg,
# suppression de plt.show(), exposition de start()/stop() et playback via window.update()

import numpy as np
import time
import sys
import warnings

# imports optionnels (fallback si non disponibles dans Pyodide)
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

try:
    from scipy.stats import binned_statistic, gaussian_kde
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False
    binned_statistic = None
    gaussian_kde = None

try:
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

# IMPORTANT: on ne change pas le calcul ; paramètres initiaux (inchangés)
sigma = 1.0                # largeur du noyau
p0 = np.array([1.0, .0])  # vecteur de référence
T_final = 100.0
n_image = 100
dt = T_final / (n_image - 1)

# grille pour tracer le champ (utilisée pour normalisation ici)
nx = 100
ny = 100
xs = np.linspace(-5, 5, nx)
ys = np.linspace(-5, 5, ny)
X, Y = np.meshgrid(xs, ys)
grid_pts = np.stack([X.ravel(), Y.ravel()], axis=-1)

eps = 1e-10

# paramètres particules initiales (idem)
grid_nx = 5
grid_ny = 5
grid_x_min, grid_x_max = -2.0, 2.0
grid_y_min, grid_y_max = -2.0, 2.0
orientation_mode = 'random'
fixed_angle = np.pi/6.0
p_magnitude = None
p_perturb = 0.00
seed = 42
rng = np.random.RandomState(seed)

# bruit
K_noise = 6
noise_amp = 0.25
L_domain = 10.0
kvecs = []
for kk in range(K_noise):
    angle = 2 * np.pi * kk / max(1, K_noise)
    kvecs.append(np.array([np.cos(angle), np.sin(angle)]) * (2 * np.pi / L_domain))
kvecs = np.array(kvecs)
amps = noise_amp * (1.0 + 0.5 * rng.randn(K_noise))

# Green kernel
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
    outer = r[:, :, None] * r[:, None, :]  # (N,2,2)
    G = A * (alpha[:, None, None] * I + beta[:, None, None] * outer)  # (N,2,2)
    return G  # (N,2,2)

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

def Evaluation_K_frame(qs_frame, ps_frame, sigma=sigma):
    Ngrid = grid_pts.shape[0]
    U_flat = np.zeros(Ngrid)
    V_flat = np.zeros(Ngrid)
    for q, p in zip(qs_frame, ps_frame):
        pts = grid_pts - q
        G = Green_K(pts, sigma=sigma)
        u = np.einsum('nij,j->ni', G, p)
        U_flat += u[:, 0]
        V_flat += u[:, 1]
    U = U_flat.reshape(X.shape)
    V = V_flat.reshape(Y.shape)
    return U, V

def deriv(state, N, sigma=sigma):
    qs = [state[2*i:2*i+2] for i in range(N)]
    ps = [state[2*N + 2*i:2*N + 2*i+2] for i in range(N)]
    dq = np.zeros((N,2))
    dp = np.zeros((N,2))
    for a in range(N):
        q_a = qs[a]
        p_a = ps[a]
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
    new_state = state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return new_state

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

# --- initialisation des particules (idem) ---
grid_x = np.linspace(grid_x_min, grid_x_max, grid_nx)
grid_y = np.linspace(grid_y_min, grid_y_max, grid_ny)
qs_init = []
ps_init = []

if p_magnitude is None:
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
        elif orientation_mode == 'fixed_angle':
            angle = fixed_angle
        else:
            angle = rng.uniform(0.0, 2*np.pi)

        p_vec = p_magnitude * np.array([np.cos(angle), np.sin(angle)])
        p_vec += p_perturb * rng.randn(2)
        ps_init.append(p_vec)

N = len(qs_init)
print(f"Initialisation: {N} particules ({grid_nx} x {grid_ny})'")

qs_hist = np.zeros((n_image, N, 2))
ps_hist = np.zeros((n_image, N, 2))
u_selfs = np.zeros((n_image, N, 2))

state = np.zeros(4 * N)
for i in range(N):
    state[2*i:2*i+2] = qs_init[i]
for i in range(N):
    state[2*N + 2*i:2*N + 2*i+2] = ps_init[i]

use_tqdm = HAS_TQDM

def format_time(seconds):
    if seconds is None:
        return "--:--:--"
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

start_time = time.time()
sqrt_dt = np.sqrt(dt)

# Intégration temporelle (principalement inchangée)
if use_tqdm:
    iterator = tqdm(range(n_image), desc="Integration", ncols=80)
    for i in iterator:
        for a in range(N):
            qs_hist[i,a] = state[2*a:2*a+2]
            ps_hist[i,a] = state[2*N + 2*a:2*N + 2*a+2]
        for a in range(N):
            u_selfs[i,a] = u_au_point(qs_hist[i,a], list(qs_hist[i]), list(ps_hist[i]), sigma=sigma)

        if i < n_image - 1:
            state_det = rk4_step(state, dt, N, sigma=sigma)
            g0 = np.zeros((K_noise, 4*N))
            for kidx in range(K_noise):
                g0[kidx] = gk_on_state(kidx, state, N)
            dW = rng.normal(0.0, sqrt_dt, size=K_noise)
            state_tilde = state_det.copy()
            state_tilde += np.tensordot(dW, g0, axes=(0,0))
            g1 = np.zeros_like(g0)
            for kidx in range(K_noise):
                g1[kidx] = gk_on_state(kidx, state_tilde, N)
            stochastic_increment = 0.5 * np.tensordot(dW, (g0 + g1), axes=(0,0))
            state = state_det + stochastic_increment
    elapsed = time.time() - start_time
    print(f"Intégration terminée en {format_time(elapsed)}")
else:
    last_print_len = 0
    for i in range(n_image):
        iter_start = time.time()
        for a in range(N):
            qs_hist[i,a] = state[2*a:2*a+2]
            ps_hist[i,a] = state[2*N + 2*a:2*N + 2*a+2]
        for a in range(N):
            u_selfs[i,a] = u_au_point(qs_hist[i,a], list(qs_hist[i]), list(ps_hist[i]), sigma=sigma)

        if i < n_image - 1:
            state_det = rk4_step(state, dt, N, sigma=sigma)
            g0 = np.zeros((K_noise, 4*N))
            for kidx in range(K_noise):
                g0[kidx] = gk_on_state(kidx, state, N)
            dW = rng.normal(0.0, sqrt_dt, size=K_noise)
            state_tilde = state_det.copy()
            state_tilde += np.tensordot(dW, g0, axes=(0,0))
            g1 = np.zeros_like(g0)
            for kidx in range(K_noise):
                g1[kidx] = gk_on_state(kidx, state_tilde, N)
            stochastic_increment = 0.5 * np.tensordot(dW, (g0 + g1), axes=(0,0))
            state = state_det + stochastic_increment

        elapsed = time.time() - start_time
        completed = i + 1
        frac = completed / n_image
        pct = int(frac * 100)
        if completed > 0:
            avg_time_per_step = elapsed / completed
            remaining = avg_time_per_step * (n_image - completed)
        else:
            remaining = None
        bar_len = 40
        filled = int(np.round(bar_len * frac))
        bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
        eta_str = format_time(remaining) if remaining is not None else "--:--:--"
        msg = f"Integration {bar} {pct:3d}% ETA {eta_str}"
        sys.stdout.write("\r" + " " * last_print_len + "\r")
        sys.stdout.write(msg)
        sys.stdout.flush()
        last_print_len = len(msg)

    total_elapsed = time.time() - start_time
    sys.stdout.write("\n")
    print(f"Intégration terminée en {format_time(total_elapsed)}")

# calcul du vmax (peut rester pour info)
vmax = 0.0
for i in range(n_image):
    Utmp, Vtmp = Evaluation_K_frame(qs_hist[i], ps_hist[i], sigma=sigma)
    vmax = max(vmax, np.sqrt(Utmp**2 + Vtmp**2).max())
if vmax == 0:
    vmax = 1e-6

# ----------------------------------------------------
# Playback pour le navigateur
# expose start() and stop() for JS
# envoie des frames via window.update(state), avec x,y normalisés [0,1], r relatif
# ----------------------------------------------------
try:
    from js import update  # window.update sera appelé côté JS
except Exception:
    # si on exécute localement en dehors du navigateur, fallback print
    def update(state):
        pass

import asyncio
_play_task = None
_playing = False
_play_interval = 0.1  # secondes entre frames (modifiable via set_interval)

# normalisation helpers
_xmin, _xmax = xs.min(), xs.max()
_ymin, _ymax = ys.min(), ys.max()
_xrange = _xmax - _xmin if (_xmax - _xmin) != 0 else 1.0
_yrange = _ymax - _ymin if (_ymax - _ymin) != 0 else 1.0

def _make_frame_state(frame_idx, r_rel=0.02):
    frame_idx = int(np.clip(frame_idx, 0, n_image - 1))
    qs_frame = qs_hist[frame_idx]
    state_list = []
    for a in range(N):
        x = float(qs_frame[a,0])
        y = float(qs_frame[a,1])
        xnorm = (x - _xmin) / _xrange
        ynorm = (y - _ymin) / _yrange
        state_list.append({'x': xnorm, 'y': ynorm, 'r': r_rel})
    return state_list

async def _playback_loop():
    global _playing
    try:
        while _playing:
            for i in range(n_image):
                if not _playing:
                    break
                state = _make_frame_state(i)
                try:
                    update(state)  # envoie au JS
                except Exception:
                    # silent fail si pas dans navigateur
                    pass
                await asyncio.sleep(_play_interval)
            # loop again
    except asyncio.CancelledError:
        # arrêt propre
        pass
    finally:
        _playing = False

def start(interval_seconds=None):
    """Démarre la lecture des frames ; non-bloquant (idempotent)."""
    global _play_task, _playing, _play_interval
    if interval_seconds is not None:
        _play_interval = float(interval_seconds)
    if _playing:
        return
    _playing = True
    # lance la coroutine playback
    try:
        # Pyodide: ensure_future fonctionne pour lancer la coroutine
        _play_task = asyncio.ensure_future(_playback_loop())
    except Exception:
        # fallback sync loop (exécution bloquante, peu probable en Pyodide)
        import threading
        def _worker():
            global _playing
            while _playing:
                for i in range(n_image):
                    if not _playing:
                        break
                    state = _make_frame_state(i)
                    try:
                        update(state)
                    except Exception:
                        pass
                    time.sleep(_play_interval)
        t = threading.Thread(target=_worker, daemon=True)
        t.start()

def stop():
    """Arrête la lecture (non bloquant)."""
    global _play_task, _playing
    _playing = False
    if _play_task is not None:
        try:
            _play_task.cancel()
        except Exception:
            pass
        _play_task = None

# Petit utilitaire : exposer une fonction non-async pour régler la vitesse si appelé depuis JS
def set_interval(seconds):
    global _play_interval
    _play_interval = float(seconds)

# Fin du fichier
