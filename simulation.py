# simulation.py
# Version adaptée pour Pyodide, jetlet niveau 0 (25 particules), renvoie Ux_fields & Uy_fields si compute_field True.
import math
import numpy as np

# ---------- paramètres principaux ----------
sigma = 1.0
p0 = np.array([1.0, 0.0])
T_final = 10.0
n_image = 30
dt_default = T_final / max(1, (n_image - 1))

# grille pour tracer le champ (optionnel)
grid_nx_default = 32
grid_ny_default = 32
grid_extent = 5.0

# --- initial particle grid (5x5 = 25 particules) ---
grid_nx = 5
grid_ny = 5
grid_x_min, grid_x_max = -2.0, 2.0
grid_y_min, grid_y_max = -2.0, 2.0

# orientation aléatoire (par défaut)
orientation_mode = 'random'
p_magnitude = None
p_perturb = 0.00
seed = 42
rng_global = np.random.RandomState(seed)

# Bruit optionnel (désactivé par défaut)
K_noise = 0
noise_amp = 0.25
L_domain = 10.0

eps = 1e-10

# ---------------- utilitaires -----------------
def Green_K(pts, sigma=1.0):
    r = np.asarray(pts, dtype=float)
    if r.ndim == 1:
        r = r[None, :]
    r2 = np.sum(r * r, axis=1)
    A = 1.0 / (4.0 * np.pi * sigma ** 2)
    B = np.exp(-r2 / (4.0 * sigma ** 2))
    C = 2.0 * sigma ** 2 / (r2 + eps)
    alpha = B - C * (1.0 - B)
    beta = (2.0 * C * (1.0 - B) - B) / (r2 + eps)
    I = np.eye(2)
    outer = r[:, :, None] * r[:, None, :]
    G = A * (alpha[:, None, None] * I + beta[:, None, None] * outer)
    return G  # (M,2,2)

def u_au_point_vectorized(xs, qs_array, ps_array, sigma=1.0):
    xs = np.asarray(xs)
    qs = np.asarray(qs_array)
    ps = np.asarray(ps_array)
    M = xs.shape[0]
    N = qs.shape[0]
    U = np.zeros((M,2))
    for a in range(N):
        r = xs - qs[a]
        G = Green_K(r, sigma=sigma)
        U += np.einsum('nij,j->ni', G, ps[a])
    return U  # (M,2)

def u_at_particle(q_a, qs_array, ps_array, sigma=sigma):
    qs = np.asarray(qs_array)
    ps = np.asarray(ps_array)
    r = qs - q_a
    G = Green_K(r, sigma=sigma)
    # corrected einsum: source index n and vector index j
    u_j = np.einsum('nij,nj->ni', G, ps)  # (N,2)
    u = u_j.sum(axis=0)
    return u

def jacobian_u_at_point_fd(x, qs_list, ps_list, sigma=sigma, h=1e-6):
    J = np.zeros((2,2))
    for j in range(2):
        e = np.zeros(2); e[j] = 1.0
        u_plus = u_au_point_vectorized(x + h*e, qs_list, ps_list, sigma)[0]
        u_minus = u_au_point_vectorized(x - h*e, qs_list, ps_list, sigma)[0]
        J[:, j] = (u_plus - u_minus) / (2.0 * h)
    return J

def sigma_k_at(kidx, qs_array):
    return np.zeros((qs_array.shape[0], 2))

def grad_sigma_k_at(kidx, qs_array):
    return np.zeros((qs_array.shape[0], 2, 2))

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

# deriv & rk4
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

# initial conditions: 5x5 grid, random orientations
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
            else:
                angle = 0.0
            p_vec = mag * np.array([np.cos(angle), np.sin(angle)])
            p_vec += p_perturb * rng.randn(2)
            ps_init.append(p_vec)
    return np.array(qs_init), np.array(ps_init)

# ---------------- simulate exposée (unique, propre) ----------------
def simulate(steps=None, dt=None, n_image_local=None, grid_nx_field=None, grid_ny_field=None,
             grid_extent_local=None, rng_seed=None, compute_field=False):
    """
    Retourne dict sérialisable :
      't', 'qs_hist', 'u_selfs', 'Ux_fields', 'Uy_fields', 'grid_x', 'grid_y', 'N', 'n_image', 'grid_nx', 'grid_ny'
    compute_field: bool -> si False Ux_fields/Uy_fields seront des listes vides par frame.
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

    # initial particles (5x5)
    qs_init, ps_init = build_initial_state(grid_nx_local=grid_nx, grid_ny_local=grid_ny, mode=orientation_mode, rng_seed=rng_seed)
    N = qs_init.shape[0]

    # state vector (positions then momenta)
    state = np.zeros(4 * N)
    for i in range(N):
        state[2*i:2*i+2] = qs_init[i]
    for i in range(N):
        state[2*N + 2*i:2*N + 2*i+2] = ps_init[i]

    qs_hist = np.zeros((n_image_local, N, 2))
    ps_hist = np.zeros((n_image_local, N, 2))
    u_selfs = np.zeros((n_image_local, N, 2))

    # prepare grid if needed
    if compute_field:
        xs = np.linspace(-grid_extent_local, grid_extent_local, grid_nx_field)
        ys = np.linspace(-grid_extent_local, grid_extent_local, grid_ny_field)
        X, Y = np.meshgrid(xs, ys)
        grid_pts = np.stack([X.ravel(), Y.ravel()], axis=-1)
        Ux_fields = np.zeros((n_image_local, grid_nx_field * grid_ny_field))
        Uy_fields = np.zeros((n_image_local, grid_nx_field * grid_ny_field))
    else:
        xs = np.array([])
        ys = np.array([])
        grid_pts = np.zeros((0,2))
        Ux_fields = [[] for _ in range(n_image_local)]
        Uy_fields = [[] for _ in range(n_image_local)]

    rng = np.random.RandomState(rng_seed)

    for i in range(n_image_local):
        # store positions and momenta
        for a in range(N):
            qs_hist[i, a] = state[2*a:2*a+2]
            ps_hist[i, a] = state[2*N + 2*a:2*N + 2*a+2]
        qs_now = np.array([state[2*a:2*a+2] for a in range(N)])
        ps_now = np.array([state[2*N + 2*a:2*N + 2*a+2] for a in range(N)])

        # compute per-particle self velocities
        for a in range(N):
            u_selfs[i, a] = u_at_particle(qs_now[a], qs_now, ps_now, sigma=sigma)

        # step forward
        if i < n_image_local - 1:
            state_det = rk4_step(state, dt, N, sigma=sigma)
            if K_noise > 0:
                g0 = np.zeros((K_noise, 4*N))
                for kidx in range(K_noise):
                    g0[kidx] = gk_on_state(kidx, state, N)
                dW = rng.normal(0.0, np.sqrt(dt), size=K_noise)
                state_tilde = state_det.copy()
                state_tilde += np.tensordot(dW, g0, axes=(0,0))
                g1 = np.zeros_like(g0)
                for kidx in range(K_noise):
                    g1[kidx] = gk_on_state(kidx, state_tilde, N)
                stochastic_increment = 0.5 * np.tensordot(dW, (g0 + g1), axes=(0,0))
                state = state_det + stochastic_increment
            else:
                state = state_det

    # compute Ux, Uy per frame if requested (uses qs_hist and ps_hist)
    if compute_field and grid_pts.shape[0] > 0:
        for i in range(n_image_local):
            uv = u_au_point_vectorized(grid_pts, qs_hist[i], ps_hist[i], sigma=sigma)  # (M,2)
            Ux_fields[i] = uv[:,0].ravel()
            Uy_fields[i] = uv[:,1].ravel()

    # prepare output: convert everything to plain python lists safely
    out = {
        't': (np.arange(n_image_local) * dt).tolist(),
        'qs_hist': [frame.ravel().tolist() for frame in qs_hist],
        'u_selfs': [frame.ravel().tolist() for frame in u_selfs],
        # use np.asarray(...).tolist() so it works whether frame is list or ndarray
        'Ux_fields': [np.asarray(frame).tolist() for frame in Ux_fields],
        'Uy_fields': [np.asarray(frame).tolist() for frame in Uy_fields],
        'grid_x': np.asarray(xs).tolist(),
        'grid_y': np.asarray(ys).tolist(),
        'N': int(N),
        'n_image': int(n_image_local),
        'grid_nx': int(grid_nx_field) if compute_field else 0,
        'grid_ny': int(grid_ny_field) if compute_field else 0
    }
    return out

# Quick local test when run as script
if __name__ == "__main__":
    res = simulate(n_image_local=10, grid_nx_field=16, grid_ny_field=16, compute_field=False)
    print("simulate keys:", list(res.keys()))
    print("t len:", len(res['t']), "N:", res['N'])

