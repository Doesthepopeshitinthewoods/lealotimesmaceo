import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from matplotlib.widgets import Slider, Button
import matplotlib as mpl
import time
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from sklearn.linear_model import LinearRegression
############### paramètres ##################
sigma = 1.0                # largeur du noyau
p0 = np.array([1.0, .0])  # vecteur de référence
T_final = 100.0
n_image = 100
dt = T_final / (n_image - 1)

# grille pour tracer le champ
nx = 100
ny = 100
xs = np.linspace(-5, 5, nx)
ys = np.linspace(-5, 5, ny)
X, Y = np.meshgrid(xs, ys)
grid_pts = np.stack([X.ravel(), Y.ravel()], axis=-1)

eps = 1e-10  # Pour éviter les divisions par zéro.

############ paramètres du cadrillage initial pour les particules ############
grid_nx = 5   # nombre de particules en x
grid_ny = 5   # nombre de particules en y
grid_x_min, grid_x_max = -2.0, 2.0
grid_y_min, grid_y_max = -2.0, 2.0

# orientation des vitesses initiales : choisissez dans
# 'random', 'radial', 'inward', 'tangential', 'vortex', 'fixed_angle'
orientation_mode = 'random'

# si fixed_angle choisi, utiliser cette valeur (en radians)
fixed_angle = np.pi/6.0

# intensité/magnitude des p (si None on prend ||p0||)
p_magnitude = None

# optionnel : petite perturbation aléatoire sur p pour éviter symétries parfaites
p_perturb = 0.00
seed = 42
rng = np.random.RandomState(seed)

############# --- paramètres bruit --- ################
K_noise = 6           # nombre de modes de bruit sigma_k
noise_amp = 0.25      # amplitude globale du bruit
L_domain = 10.0       # échelle pour vecteurs d'onde
kvecs = []
for kk in range(K_noise):
    angle = 2 * np.pi * kk / max(1, K_noise)
    kvecs.append(np.array([np.cos(angle), np.sin(angle)]) * (2 * np.pi / L_domain))
kvecs = np.array(kvecs)
amps = noise_amp * (1.0 + 0.5 * rng.randn(K_noise))  # amplitudes aléatoires par mode

############# noyau Green K  ##################
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

######## Fonctions pour dynamique à N particules #########
def u_au_point(x, qs_list, ps_list, sigma=sigma):
    u = np.zeros(2)
    for q, p in zip(qs_list, ps_list):
        G = Green_K(np.array([x - q]), sigma=sigma)[0]  # (2,2)
        u += G.dot(p)
    return u

def jacobian_u_au_point(x, qs_list, ps_list, sigma=sigma, h=1e-5):
    J = np.zeros((2,2))
    for j in range(2):
        e = np.zeros(2); e[j] = 1.0
        u_plus  = u_au_point(x + h*e, qs_list, ps_list, sigma)
        u_minus = u_au_point(x - h*e, qs_list, ps_list, sigma)
        J[:, j] = (u_plus - u_minus) / (2.0 * h)
    return J

def Evaluation_K_frame(qs_frame, ps_frame, sigma=sigma):
    Ngrid = grid_pts.shape[0]
    U_flat = np.zeros(Ngrid)
    V_flat = np.zeros(Ngrid)
    for q, p in zip(qs_frame, ps_frame):
        pts = grid_pts - q                     # (Ngrid, 2)
        G = Green_K(pts, sigma=sigma)         # (Ngrid,2,2)
        u = np.einsum('nij,j->ni', G, p)      # (Ngrid,2)
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
        J = jacobian_u_au_point(q_a, qs, ps, sigma=sigma, h=1e-5)  # (2,2)
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

######### --- champs sigma_k et gradients analytiques simples --- ##########
def sigma_k_at(kidx, qs_array):
    # qs_array: (N,2) -> renvoie (N,2)
    k = kvecs[kidx]
    phase = qs_array.dot(k)
    a = amps[kidx]
    return a * np.stack([-np.sin(phase), np.cos(phase)], axis=-1)

def grad_sigma_k_at(kidx, qs_array):
    # renvoie (N,2,2) avec dérivée de sigma_i par rapport à x_j
    k = kvecs[kidx]
    phase = qs_array.dot(k)
    a = amps[kidx]
    M = qs_array.shape[0]
    G = np.zeros((M,2,2))
    # d/dx_j [-sin(k·x)] = -cos(k·x) * k_j ; d/dx_j [cos(k·x)] = -sin(k·x) * k_j
    G[:,0,0] = -np.cos(phase) * k[0] * a
    G[:,0,1] = -np.cos(phase) * k[1] * a
    G[:,1,0] = -np.sin(phase) * k[0] * a
    G[:,1,1] = -np.sin(phase) * k[1] * a
    return G

def gk_on_state(kidx, state, N):
    # renvoie vecteur shape (4N,) correspondant à l'action du champ sigma_k sur l'état
    qs = np.array([state[2*i:2*i+2] for i in range(N)])
    ps = np.array([state[2*N + 2*i:2*N + 2*i+2] for i in range(N)])
    sig = sigma_k_at(kidx, qs)            # (N,2)
    grad_sig = grad_sigma_k_at(kidx, qs)  # (N,2,2)
    dq = sig
    dp = np.zeros_like(ps)
    # dp_i = - (grad_sig[i].T @ p_i)
    for i in range(N):
        dp[i] = - grad_sig[i].T.dot(ps[i])
    return np.concatenate([dq.ravel(), dp.ravel()])

############# Pré-calcul de la trajectoire (N particules) ###########
# construction du cadrillage initial pour les particules
grid_x = np.linspace(grid_x_min, grid_x_max, grid_nx)
grid_y = np.linspace(grid_y_min, grid_y_max, grid_ny)
qs_init = []
ps_init = []

# magnitude par défaut
if p_magnitude is None:
    p_magnitude = np.linalg.norm(p0) if np.linalg.norm(p0) > 0 else 1.0

for yi in grid_y:
    for xi in grid_x:
        qs_init.append(np.array([xi, yi]))
        # calcul de l'angle selon le mode choisi
        if orientation_mode == 'random':
            angle = rng.uniform(0.0, 2*np.pi)
        elif orientation_mode == 'radial':
            angle = np.arctan2(yi, xi)        # sortant du centre
        elif orientation_mode == 'inward':
            angle = np.arctan2(yi, xi) + np.pi  # vers le centre
        elif orientation_mode == 'tangential':
            angle = np.arctan2(yi, xi) + 0.5*np.pi  # ccw tangentiel
        elif orientation_mode == 'vortex':
            angle = np.arctan2(yi, xi) - 0.5*np.pi  # cw tangentiel
        elif orientation_mode == 'fixed_angle':
            angle = fixed_angle
        else:
            # fallback : random
            angle = rng.uniform(0.0, 2*np.pi)

        # vecteur p avec magnitude fixée + perturbation aléatoire
        p_vec = p_magnitude * np.array([np.cos(angle), np.sin(angle)])
        p_vec += p_perturb * rng.randn(2)
        ps_init.append(p_vec)

N = len(qs_init)
print(f"Initialisation: {N} particules ({grid_nx} x {grid_ny})'")

# vecteurs pour stocker l'historique
qs_hist = np.zeros((n_image, N, 2))
ps_hist = np.zeros((n_image, N, 2))
u_selfs = np.zeros((n_image, N, 2))

# état initial aplati
state = np.zeros(4 * N)
for i in range(N):
    state[2*i:2*i+2] = qs_init[i]
for i in range(N):
    state[2*N + 2*i:2*N + 2*i+2] = ps_init[i]

# --- Barre de progression / estimation du temps restant ---
# On utilise tqdm si disponible, sinon on affiche une barre simple avec ETA.
use_tqdm = False
try:
    from tqdm import tqdm
    use_tqdm = True
except Exception:
    use_tqdm = False

def format_time(seconds):
    if seconds is None:
        return "--:--:--"
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

start_time = time.time()
sqrt_dt = np.sqrt(dt)
if use_tqdm:
    iterator = tqdm(range(n_image), desc="Integration", ncols=80)
    for i in iterator:
        # déplier et stocker
        for a in range(N):
            qs_hist[i,a] = state[2*a:2*a+2]
            ps_hist[i,a] = state[2*N + 2*a:2*N + 2*a+2]
        for a in range(N):
            u_selfs[i,a] = u_au_point(qs_hist[i,a], list(qs_hist[i]), list(ps_hist[i]), sigma=sigma)

        if i < n_image - 1:
            # 1) pas déterministe RK4
            state_det = rk4_step(state, dt, N, sigma=sigma)

            # 2) pas stochastique Stratonovich via Heun explicite
            # calcul g_k au temps t
            g0 = np.zeros((K_noise, 4*N))
            for kidx in range(K_noise):
                g0[kidx] = gk_on_state(kidx, state, N)

            # tirages Gaussien indépendants ~ N(0, dt)
            dW = rng.normal(0.0, sqrt_dt, size=K_noise)

            # predictor : state_tilde = state_det + sum_k g0_k * dW_k
            state_tilde = state_det.copy()
            state_tilde += np.tensordot(dW, g0, axes=(0,0))

            # eval g at tilde
            g1 = np.zeros_like(g0)
            for kidx in range(K_noise):
                g1[kidx] = gk_on_state(kidx, state_tilde, N)

            # Heun Stratonovich correction
            stochastic_increment = 0.5 * np.tensordot(dW, (g0 + g1), axes=(0,0))
            state = state_det + stochastic_increment
    elapsed = time.time() - start_time
    print(f"Intégration terminée en {format_time(elapsed)}")
else:
    # barre simple avec ETA
    last_print_len = 0
    for i in range(n_image):
        iter_start = time.time()
        # déplier et stocker
        for a in range(N):
            qs_hist[i,a] = state[2*a:2*a+2]
            ps_hist[i,a] = state[2*N + 2*a:2*N + 2*a+2]
        for a in range(N):
            u_selfs[i,a] = u_au_point(qs_hist[i,a], list(qs_hist[i]), list(ps_hist[i]), sigma=sigma)

        if i < n_image - 1:
            # 1) pas déterministe RK4
            state_det = rk4_step(state, dt, N, sigma=sigma)

            # 2) pas stochastique Stratonovich via Heun explicite
            g0 = np.zeros((K_noise, 4*N))
            for kidx in range(K_noise):
                g0[kidx] = gk_on_state(kidx, state, N)

            # tirages Gaussien indépendants ~ N(0, dt)
            dW = rng.normal(0.0, sqrt_dt, size=K_noise)

            # predictor : state_tilde = state_det + sum_k g0_k * dW_k
            state_tilde = state_det.copy()
            state_tilde += np.tensordot(dW, g0, axes=(0,0))

            # eval g at tilde
            g1 = np.zeros_like(g0)
            for kidx in range(K_noise):
                g1[kidx] = gk_on_state(kidx, state_tilde, N)

            # Heun Stratonovich correction
            stochastic_increment = 0.5 * np.tensordot(dW, (g0 + g1), axes=(0,0))
            state = state_det + stochastic_increment

        # mise à jour de la barre
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

# calcul du vmax pour l'échelle du plot
vmax = 0.0
for i in range(n_image):
    Utmp, Vtmp = Evaluation_K_frame(qs_hist[i], ps_hist[i], sigma=sigma)
    vmax = max(vmax, np.sqrt(Utmp**2 + Vtmp**2).max())
if vmax == 0:
    vmax = 1e-6

cmap_name = 'plasma'
norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)
sm = mpl.cm.ScalarMappable(cmap=cmap_name, norm=norm)
sm.set_array([])

fig = plt.figure(figsize=(8, 9))
ax_field = fig.add_axes([0.08, 0.25, 0.88, 0.70])
ax_field.set_xlim(xs.min(), xs.max())
ax_field.set_ylim(ys.min(), ys.max())
ax_field.set_aspect('equal')

cbar = fig.colorbar(sm, ax=ax_field, orientation='vertical', pad=0.02)
cbar.set_label('|u| module de la vitesse')

ax_bar = fig.add_axes([0.08, 0.18, 0.88, 0.04])
ax_bar.set_xlim(0, T_final)
ax_bar.set_ylim(0, 1)
ax_bar.axis('off')
bar_bg = patches.Rectangle((0, 0.15), T_final, 0.7, edgecolor='black', facecolor='none')
ax_bar.add_patch(bar_bg)
progress_rect = patches.Rectangle((0, 0.15), 0.0, 0.7, edgecolor='none', facecolor='tab:blue')
ax_bar.add_patch(progress_rect)
time_text = ax_bar.text(0.5 * T_final, 0.55, "", ha='center', va='center', color='white', fontsize=9)

ax_slider_t = fig.add_axes([0.08, 0.09, 0.60, 0.03])
slider_t = Slider(ax_slider_t, 'Temps', 0.0, T_final, valinit=0.0)

ax_slider_speed = fig.add_axes([0.70, 0.09, 0.16, 0.03])
slider_speed = Slider(ax_slider_speed, 'Vitesse', 0.1, 5.0, valinit=1.0)

ax_button = fig.add_axes([0.88, 0.09, 0.06, 0.03])
button = Button(ax_button, 'Pause')

current_frame = 0
is_playing = True
base_interval = 100  # ms à vitesse = 1

vel_scale = 1.0
min_arrow_len = 0.4
back_offset = 0.12
head_extra = 0.4

def draw_frame(frame_idx):
    frame_idx = int(np.clip(frame_idx, 0, n_image - 1))
    qs_frame = qs_hist[frame_idx]
    ps_frame = ps_hist[frame_idx]

    U, V = Evaluation_K_frame(qs_frame, ps_frame, sigma=sigma)

    ax_field.clear()
    ax_field.set_xlim(xs.min(), xs.max())
    ax_field.set_ylim(ys.min(), ys.max())
    ax_field.set_aspect('equal')

    speed = np.sqrt(U ** 2 + V ** 2)
    strm = ax_field.streamplot(X, Y, U, V, color=speed, cmap=cmap_name, norm=norm,
                               density=1.3, linewidth=1)

    for a in range(N):
        q = qs_frame[a]
        u_self = u_selfs[frame_idx, a]
        ax_field.plot([q[0]], [q[1]], 'ro', ms=6, zorder=5)

        vnorm = np.linalg.norm(u_self)
        if vnorm < 1e-12:
            direction = np.array([1.0, 0.0])
        else:
            direction = u_self / vnorm
        length = max(min_arrow_len, vel_scale * vnorm) + head_extra
        tail = q - back_offset * direction
        head = q + length * direction
        ax_field.annotate(
            "",
            xy=(head[0], head[1]),
            xytext=(tail[0], tail[1]),
            arrowprops=dict(arrowstyle='-|>', linewidth=2.0, mutation_scale=18, color='r'),
            zorder=6
        )
        text_dx = 0.08
        text_dy = 0.12 + 0.03 * (a % 6)
        ax_field.text(q[0] + text_dx, q[1] + text_dy, f"{vnorm:.3f}", fontsize=7, color='k', zorder=7)

    t = frame_idx * dt
    ax_field.set_title(f"jetlets 0-niveau, t = {t:.2f} / {T_final:.2f} s — orientation des vitesses='{orientation_mode}'")
    progress_rect.set_x(0.0)
    progress_rect.set_width(t)
    time_text.set_text(f"{t:.2f} s")
    fig.canvas.draw_idle()

def update_anim(_):
    global current_frame
    speed = slider_speed.val
    step = int(np.clip(np.round(speed), 1, n_image))
    current_frame = (current_frame + step) % n_image
    draw_frame(current_frame)
    slider_t.eventson = False
    slider_t.set_val(current_frame * dt)
    slider_t.eventson = True
    return []

def on_slider_t(val):
    global current_frame, is_playing
    frame_idx = int(round(val / dt))
    frame_idx = np.clip(frame_idx, 0, n_image - 1)
    current_frame = frame_idx
    if is_playing:
        anim.event_source.stop()
        is_playing = False
        button.label.set_text('Play')
    draw_frame(current_frame)

slider_t.on_changed(on_slider_t)

def on_speed(val):
    speed = val
    new_interval = base_interval / speed
    anim.event_source.interval = max(1, new_interval)

slider_speed.on_changed(on_speed)

def on_button(event):
    global is_playing
    if is_playing:
        anim.event_source.stop()
        is_playing = False
        button.label.set_text('Play')
    else:
        anim.event_source.start()
        is_playing = True
        button.label.set_text('Pause')

button.on_clicked(on_button)

anim = animation.FuncAnimation(fig, update_anim, frames=n_image, interval=base_interval, blit=False, repeat=True)

draw_frame(0)
plt.show()








# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import warnings

# check imports
try:
    from scipy.stats import gaussian_kde
except Exception:
    gaussian_kde = None
    warnings.warn("scipy.stats.gaussian_kde non disponible — fallback histogrammes pour les densités.")

# ----------------------------
# Paramètres Monte-Carlo
# ----------------------------
M = 200               # nombre de réalisations (ajuste selon ressources)
seed0 = 1000          # graine de base pour dériver seeds
particle_index = 0    # index de la particule que l'on suit (0..N-1)
# si tu veux sélectionner par position initiale, utilitaire plus bas
save_results = False  # si True, sauvegarde .npz à la fin

# On assume que qs_init, ps_init, N, n_image, dt, sigma, K_noise, kvecs, amps existent
sqrt_dt = np.sqrt(dt)

# Reconstruction de l'état initial aplati (pour une simulation indépendante)
def build_state_from_inits(qs_init_list, ps_init_list):
    Nloc = len(qs_init_list)
    state0 = np.zeros(4 * Nloc)
    for i in range(Nloc):
        state0[2*i:2*i+2] = qs_init_list[i]
    for i in range(Nloc):
        state0[2*Nloc + 2*i:2*Nloc + 2*i+2] = ps_init_list[i]
    return state0

# wrapper qui lance une simulation (même intégrateur stochastique que précédemment)
def run_simulation(seed):
    rng_local = np.random.RandomState(seed)
    state = build_state_from_inits(qs_init, ps_init)  # qs_init, ps_init doivent exister
    Nloc = N
    qs_hist_run = np.zeros((n_image, Nloc, 2))
    us_hist_run = np.zeros((n_image, Nloc, 2))

    for i in range(n_image):
        # stocker
        for a in range(Nloc):
            qs_hist_run[i,a] = state[2*a:2*a+2]
            us_hist_run[i,a] = u_au_point(qs_hist_run[i,a], list(qs_hist_run[i]), list(state[2*Nloc:2*Nloc+2*Nloc].reshape(Nloc,2)), sigma=sigma) if False else None
            # note: above line is left as None to avoid expensive self-evaluation. We'll compute us below properly.

        # Compute and store u_selfs (efficient way)
        # we can use the existing Evaluation_K_frame to compute global field and then call u_au_point per particle,
        # but cheaper: call u_au_point for each particle using current qs and ps.
        qs_now = np.array([state[2*a:2*a+2] for a in range(Nloc)])
        ps_now = np.array([state[2*Nloc + 2*a:2*Nloc + 2*a+2] for a in range(Nloc)])
        # compute u_self for every particle
        for a in range(Nloc):
            us_hist_run[i,a] = u_au_point(qs_now[a], list(qs_now), list(ps_now), sigma=sigma)

        if i < n_image - 1:
            # deterministic RK4 step
            state_det = rk4_step(state, dt, Nloc, sigma=sigma)

            # stochastic increments g_k at time t
            g0 = np.zeros((K_noise, 4*Nloc))
            for kidx in range(K_noise):
                g0[kidx] = gk_on_state(kidx, state, Nloc)

            # draw dW ~ N(0, dt)
            dW = rng_local.normal(0.0, sqrt_dt, size=K_noise)

            # predictor
            state_tilde = state_det.copy()
            state_tilde += np.tensordot(dW, g0, axes=(0,0))

            # eval g at tilde
            g1 = np.zeros_like(g0)
            for kidx in range(K_noise):
                g1[kidx] = gk_on_state(kidx, state_tilde, Nloc)

            stochastic_increment = 0.5 * np.tensordot(dW, (g0 + g1), axes=(0,0))
            state = state_det + stochastic_increment

    return qs_hist_run, us_hist_run

# ---- lancer M simulations et stocker trajectoires pour la particule d'intérêt ----
# allocations
qs_samples = np.zeros((M, n_image, 2))   # position of tracked particle for each run/time
u_samples = np.zeros((M, n_image, 2))    # velocity (u_self) of tracked particle for each run/time

for m in range(M):
    seed_m = seed0 + m
    qs_run, us_run = run_simulation(seed_m)
    qs_samples[m] = qs_run[:, particle_index, :]
    u_samples[m] = us_run[:, particle_index, :]

# Optionnel : sauvegarde
if save_results:
    np.savez("mc_particle_samples.npz", qs_samples=qs_samples, u_samples=u_samples, particle_index=particle_index)

# ----------------------------
# Fonctions d'analyse / probabilités
# ----------------------------
def empirical_probability_in_region(time_index, pos_region, vel_region):
    """
    pos_region: tuple (center, radius) or axis-aligned box ((x_min,x_max),(y_min,y_max))
    vel_region: same structure for velocity (vx,vy) or (center, radius)
    returns fraction of runs where particle at time_index is in pos_region AND vel_region
    """
    # positions and velocities across runs at that time
    posM = qs_samples[:, time_index, :]   # (M,2)
    velM = u_samples[:, time_index, :]   # (M,2)

    def in_pos(pt):
        if len(pos_region) == 2 and np.isscalar(pos_region[1]):
            c, r = pos_region
            return np.sum((pt - c)**2) <= r**2
        else:
            # box ((xmin,xmax),(ymin,ymax))
            (xmin,xmax),(ymin,ymax) = pos_region
            return (xmin <= pt[0] <= xmax) and (ymin <= pt[1] <= ymax)

    def in_vel(v):
        if len(vel_region) == 2 and np.isscalar(vel_region[1]):
            c, r = vel_region
            return np.sum((v - c)**2) <= r**2
        else:
            (vxmin,vxmax),(vymin,vymax) = vel_region
            return (vxmin <= v[0] <= vxmax) and (vymin <= v[1] <= vymax)

    # vectorize checks
    count = 0
    for m in range(M):
        if in_pos(posM[m]) and in_vel(velM[m]):
            count += 1
    prob = count / float(M)
    return prob

def empirical_marginal_kde(time_index, which='pos', bw_method='scott'):
    """
    Returns a KDE object if scipy available, otherwise returns histogram (bin centers, density).
    which: 'pos' or 'vel' (2D each). For 2D KDE gaussian_kde can estimate joint density in 2D.
    """
    data = qs_samples[:, time_index, :].T if which=='pos' else u_samples[:, time_index, :].T  # shape (2,M)
    if gaussian_kde is not None:
        try:
            kde = gaussian_kde(data, bw_method=bw_method)
            return kde
        except Exception as e:
            warnings.warn("gaussian_kde failed: " + str(e))
    # fallback histogram 2D
    xs = data[0,:]; ys = data[1,:]
    nb = 30
    H, xedges, yedges = np.histogram2d(xs, ys, bins=nb, density=True)
    xcent = 0.5*(xedges[:-1]+xedges[1:])
    ycent = 0.5*(yedges[:-1]+yedges[1:])
    return ('hist2d', H, xcent, ycent)

# ----------------------------
# Exemples d'utilisation
# ----------------------------
# 1) Probabilité empirique : exemple, probabilité que la particule soit dans un disque center=(0,0), r=0.5
t_idx = int(n_image//2)   # instant intermédiaire
pos_center = np.array([0.0, 0.0])
pos_radius = 0.5
# pour la vitesse centre (0,0) et rayon 0.3
vel_center = np.array([0.0, 0.0])
vel_radius = 0.3
prob_example = empirical_probability_in_region(t_idx, (pos_center,pos_radius), (vel_center,vel_radius))
print(f"Probabilité empirique (pos dans disque r={pos_radius}, vel dans disque r={vel_radius}) à t={t_idx*dt:.3f}s ≈ {prob_example:.3f} (M={M})")

# 2) tracer KDE marginale joint pos (2D) et vel (2D) pour l'instant t_idx (si scipy gaussian_kde dispo)
kde_pos = empirical_marginal_kde(t_idx, which='pos')   # 2D KDE object ou histogram fallback
if isinstance(kde_pos, tuple) and kde_pos[0]=='hist2d':
    H, xcent, ycent = kde_pos[1], kde_pos[2], kde_pos[3]
    plt.figure(figsize=(6,5))
    plt.contourf(xcent, ycent, H.T, cmap='viridis')
    plt.title(f"Histogramme 2D empirique position, t={t_idx*dt:.2f}s")
    plt.xlabel('x'); plt.ylabel('y'); plt.colorbar()
else:
    # dessiner contours de la KDE (évaluer sur grille)
    xmin = qs_samples[:,:,0].min(); xmax = qs_samples[:,:,0].max()
    ymin = qs_samples[:,:,1].min(); ymax = qs_samples[:,:,1].max()
    Xg = np.linspace(xmin, xmax, 80)
    Yg = np.linspace(ymin, ymax, 80)
    XX, YY = np.meshgrid(Xg, Yg)
    pts = np.vstack([XX.ravel(), YY.ravel()])
    Z = kde_pos(pts).reshape(XX.shape)
    plt.figure(figsize=(6,5))
    plt.contourf(XX, YY, Z, levels=20, cmap='viridis')
    plt.title(f"KDE empirique position (particle {particle_index}), t={t_idx*dt:.2f}s")
    plt.xlabel('x'); plt.ylabel('y'); plt.colorbar()

plt.show()
