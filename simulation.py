# simulation.py
# Version adaptée pour Pyodide : calcule trajectoires + champs et renvoie
# uniquement des listes Python sérialisables pour affichage côté JS.
#
# Basée sur ton code — j'ai retiré matplotlib / seaborn / sklearn / tkinter,
# j'ai vectorisé certaines étapes et converti les résultats en listes.

import math
import time
import numpy as np

# ----------- paramètres (tu peux ajuster) -------------
sigma = 1.0
p0 = np.array([1.0, 0.0])
T_final = 30.0
n_image = 60            # nombre d'images (frames)
dt_default = T_final / max(1, (n_image - 1))

# grille pour tracer le champ (par défaut raisonnable : 40x40)
grid_nx_default = 40
grid_ny_default = 40
grid_extent = 5.0  # domaine [-grid_extent, grid_extent] dans x,y

# paramètres particules initiales (grid)
grid_nx = 5
grid_ny = 5
grid_x_min, grid_x_max = -2.0, 2.0
grid_y_min, grid_y_max = -2.0, 2.0

# orientation des vitesses initiales
orientation_mode = 'random'
fixed_angle = np.pi / 6.0
p_magnitude = None
p_perturb = 0.00
seed = 42
rng_global = np.random.RandomState(seed)

# bruit (modes)
K_noise = 6
noise_amp = 0.25
L_domain = 10.0
kvecs = []
for kk in range(K_noise):
    angle = 2 * np.pi * kk / max(1, K_noise)
    kvecs.append(np.array([np.cos(angle), np.sin(angle)]) * (2 * np.pi / L_domain))
kvecs = np.array(kvecs)
amps = noise_amp * (1.0 + 0.5 * rng_global.randn(K_noise))

eps = 1e-10

# ---------------- core math functions (vectorized where appropriate) ----------------
def Green_K(pts, sigma=1.0):
    # pts: (M,2) array of displacement vectors r = x - q
    r = np.asarray(pts, dtype=float)
    if r.ndim == 1:
        r = r[None, :]
    r2 = np.sum(r * r, axis=1)  # (M,)
    A = 1.0 / (4.0 * np.pi * sigma ** 2)
    B = np.exp(-r2 / (4.0 * sigma ** 2))
    C = 2.0 * sigma ** 2 / (r2 + eps)
    alpha = B - C * (1.0 - B)
    beta = (2.0 * C * (1.0 - B) - B) / (r2 + eps)
    I = np.eye(2)
    # build G for each point
    M = r.shape[0]
    # outer products r_i * r_j
    outer = r[:, :, None] * r[:, None, :]  # (M,2,2)
    # broadcast alpha and beta
    G = A * (alpha[:, None, None] * I + beta[:, None, None] * outer)
    return G  # shape (M,2,2)

def u_au_point_vectorized(xs, qs_array, ps_array, sigma=1.0):
    # xs: (M,2) points where to evaluate
    # qs_array: (N,2) particle positions
    # ps_array: (N,2) particle momenta
    # returns u at each xs shape (M,2)
    xs = np.asarray(xs)
    qs = np.asarray(qs_array)
    ps = np.asarray(ps_array)
    M = xs.shape[0]
    N = qs.shape[0]
    U = np.zeros((M,2))
    # vectorized loop over sources (N moderate)
    for a in range(N):
        r = xs - qs[a]  # (M,2)
        G = Green_K(r, sigma=sigma)  # (M,2,2)
        # contract G with p_a
        U += np.einsum('nij,j->ni', G, ps[a])
    return U  # (M,2)

def u_at_particle(q_a, qs_array, ps_array, sigma=sigma):
    # compute u at position q_a due to all sources (including self)
    qs = np.asarray(qs_array)
    ps = np.asarray(ps_array)
    r = qs - q_a  # (N,2)
    G = Green_K(r, sigma=sigma)  # (N,2,2)
    # for each source j: G_j dot p_j -> (N,2) then sum
    u_j = np.einsum('nij,j->ni', G, ps)  # (N,2)
    u = u_j.sum(axis=0)
    return u

def jacobian_u_at_point_fd(x, qs_list, ps_list, sigma=sigma, h=1e-6):
    # finite difference Jacobian (2x2)
    J = np.zeros((2,2))
    for j in range(2):
        e = np.zeros(2); e[j] = 1.0
        u_plus = u_au_point_vectorized(x + h*e, qs_list, ps_list, sigma)[0]
        u_minus = u_au_point_vectorized(x - h*e, qs_list, ps_list, sigma)[0]
        J[:, j] = (u_plus - u_minus) / (2.0 * h)
    return J

def gk_on_state(kidx, state, N):
    # returns 4N vector action of sigma_k on flattened state
    qs = np.array([state[2*i:2*i+2] for i in range(N)])
    ps = np.array([state[2*N + 2*i:2*N + 2*i+2] for i in range(N)])
    sig = sigma_k_at(kidx, qs)            # (N,2)
    grad_sig = grad_sigma_k_at(kidx, qs)  # (N,2,2)
    dq = sig
    dp = np.zeros_like(ps)
    for i in range(N):
        dp[i] = - grad_sig[i].T.dot(ps[i])
    return np.concatenate([dq.ravel(), dp.ravel()])

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

# deriv, rk4 as in your code (vectorized loops)
def deriv(state, N, sigma=sigma):
    qs = [state[2*i:2*i+2] for i in range(N)]
    ps = [state[2*N + 2*i:2*N + 2*i+2] for i in range(N)]
    dq = np.zeros((N,2))
    dp = np.zeros((N,2))
    for a in range(N):
        q_a = qs[a]
        p_a = ps[a]
        u_q = u_au_point_vectorized(q_a[None,:], qs, ps, sigma=sigma)[0]
        J = jacobian_u_at_point_fd(q_a, qs, ps, sigma=sigma, h=1e-5)
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

# ---------------- initial condition construction ----------------
def build_initial_state(grid_nx_local=grid_nx, grid_ny_local=grid_ny, mode=orientation_mode, rng_seed=seed):
    rng = np.random.RandomState(rng_seed)
    grid_x = np.linspace(grid_x_min, grid_x_max, grid_nx_local)
    grid_y = np.linspace(grid_y_min, grid_y_max, grid_ny_local)
    qs_init = []
    ps_init = []
    mag = p_magnitude if p_magnitude is not None else (np.linalg.norm(p0) if np.linalg.norm(p0) > 0 else 1.0)
    for yi in grid_y:
        for xi in grid_x:
            qs_init.append(np.array([xi, yi]))
            if mode == 'random':
                angle = rng.uniform(0.0, 2*np.pi)
            elif mode == 'radial':
                angle = np.arctan2(yi, xi)
            elif mode == 'inward':
                angle = np.arctan2(yi, xi) + np.pi
            elif mode == 'tangential':
                angle = np.arctan2(yi, xi) + 0.5*np.pi
            elif mode == 'vortex':
                angle = np.arctan2(yi, xi) - 0.5*np.pi
            elif mode == 'fixed_angle':
                angle = fixed_angle
            else:
                angle = rng.uniform(0.0, 2*np.pi)
            p_vec = mag * np.array([np.cos(angle), np.sin(angle)])
            p_vec += p_perturb * rng.randn(2)
            ps_init.append(p_vec)
    return np.array(qs_init), np.array(ps_init)

# ---------------- function exposed to Stage.html ----------------
def simulate(steps=None, dt=None, n_image_local=None, grid_nx_field=None, grid_ny_field=None,
             grid_extent_local=None, rng_seed=None):
    """
    simulate(...) computes the particle trajectories and (optionally) the field on a grid.
    Returns a dict containing serializable lists:
      - 't' : list of times (n_image_local)
      - 'qs_hist' : list of frames; each frame is flattened positions [x1,y1,x2,y2,...] (length 2N)
      - 'u_selfs' : list of frames; each frame flattened self velocities for each particle [vx1,vy1,...]
      - 'Ufields' : list of frames; each frame flattened magnitude on grid (grid_nx_field*grid_ny_field) (optional)
      - 'grid_x', 'grid_y' : lists for grid coordinates
      - 'N', 'n_image' : ints
    """
    # defaults
    if n_image_local is None:
        n_image_local = n_image
    if steps is None:
        steps = n_image_local
    if dt is None:
        dt = dt_default
    if grid_nx_field is None:
        grid_nx_field = grid_nx_default
    if grid_ny_field is None:
        grid_ny_field = grid_ny_default
    if grid_extent_local is None:
        grid_extent_local = grid_extent
    if rng_seed is None:
        rng_seed = seed

    # build initial particles
    qs_init, ps_init = build_initial_state(grid_nx_local=grid_nx, grid_ny_local=grid_ny, mode=orientation_mode, rng_seed=rng_seed)
    N = qs_init.shape[0]

    # prepare state vector
    state = np.zeros(4 * N)
    for i in range(N):
        state[2*i:2*i+2] = qs_init[i]
    for i in range(N):
        state[2*N + 2*i:2*N + 2*i+2] = ps_init[i]

    # allocate history
    qs_hist = np.zeros((n_image_local, N, 2))
    ps_hist = np.zeros((n_image_local, N, 2))
    u_selfs = np.zeros((n_image_local, N, 2))

    # grid for field
    xs = np.linspace(-grid_extent_local, grid_extent_local, grid_nx_field)
    ys = np.linspace(-grid_extent_local, grid_extent_local, grid_ny_field)
    X, Y = np.meshgrid(xs, ys)
    grid_pts = np.stack([X.ravel(), Y.ravel()], axis=-1)  # (grid_nx_field*grid_ny_field, 2)

    rng = np.random.RandomState(rng_seed)
    sqrt_dt = np.sqrt(dt)

    for i in range(n_image_local):
        # store
        for a in range(N):
            qs_hist[i, a] = state[2*a:2*a+2]
            ps_hist[i, a] = state[2*N + 2*a:2*N + 2*a+2]
        # compute u_selfs for each particle
        qs_now = np.array([state[2*a:2*a+2] for a in range(N)])
        ps_now = np.array([state[2*N + 2*a:2*N + 2*a+2] for a in range(N)])
        # compute per-particle velocities
        for a in range(N):
            u_selfs[i, a] = u_at_particle(qs_now[a], qs_now, ps_now, sigma=sigma)

        if i < n_image_local - 1:
            # deterministic RK4
            state_det = rk4_step(state, dt, N, sigma=sigma)

            # stochastic part (Stratonovich Heun) using precomputed gk_on_state
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

    # optionally compute field magnitudes on grid for each frame (may be heavy)
    Ufields = np.zeros((n_image_local, grid_nx_field * grid_ny_field))
    for i in range(n_image_local):
        U, V = u_au_point_vectorized(grid_pts, qs_hist[i], ps_hist[i], sigma=sigma).T  # returns (2, M)? ensure shape
        # u_au_point_vectorized returns (M,2) ; we want magnitude
        UV = np.sqrt(U**2 + V**2) if isinstance(U, np.ndarray) else np.sqrt(U*U + V*V)
        # In case returned shape: ensure 1D
        if UV.ndim > 1:
            UV = UV.ravel()
        Ufields[i] = UV.ravel()

    # build serializable output (convert numpy arrays to lists)
    out = {
        't': (np.arange(n_image_local) * dt).tolist(),
        'qs_hist': [frame.ravel().tolist() for frame in qs_hist],  # each frame flattened [x1,y1,x2,y2,...]
        'u_selfs': [frame.ravel().tolist() for frame in u_selfs],
        'Ufields': [frame.tolist() for frame in Ufields],  # each is length grid_nx_field*grid_ny_field
        'grid_x': xs.tolist(),
        'grid_y': ys.tolist(),
        'N': int(N),
        'n_image': int(n_image_local),
        'grid_nx': int(grid_nx_field),
        'grid_ny': int(grid_ny_field)
    }
    return out

# If executed locally for testing:
if __name__ == "__main__":
    res = simulate()
    print("simulate produced keys:", list(res.keys()))
    print("t len:", len(res['t']), "N:", res['N'], "frames:", res['n_image'])
