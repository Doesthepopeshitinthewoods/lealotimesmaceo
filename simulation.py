# simulation_test.py
# Test simple pour vérifier l'affichage : envoie des frames simples vers window.update()
# Compatible Pyodide (utilise from js import update, set_progress si disponibles)

import math
import asyncio
import time

# try to get JS callbacks provided by the page
try:
    from js import update, set_progress
except Exception:
    # fallback si on lance en dehors du navigateur
    def update(state):
        print("update called, len=", len(state))
    def set_progress(p):
        print("progress:", p)

# Paramètres du test
n_particles = 8
n_frames = 60
_loop_duration_seconds = 4.0   # durée d'une boucle complète (s)
_interval = max(0.01, _loop_duration_seconds / n_frames)  # secondes entre frames
_radius_x = 0.35   # amplitude en x (dans [0,1])
_radius_y = 0.25   # amplitude en y
_center = (0.5, 0.5)
_particle_radius = 0.03  # rayon relatif pour le dessin

_playing = False
_play_task = None

def _make_frame(frame_idx):
    """Retourne la liste de dicts {x,y,r} pour le frame donné (coord normalisées 0..1)."""
    t = frame_idx / float(n_frames)  # [0..1)
    out = []
    for i in range(n_particles):
        base_angle = 2.0 * math.pi * (i / float(n_particles))
        # chaque particule avance: angle = base + 2pi * t
        angle = base_angle + 2.0 * math.pi * t
        # position elliptique autour du centre
        x = _center[0] + _radius_x * math.cos(angle)
        y = _center[1] + _radius_y * math.sin(angle)
        out.append({'x': float(x), 'y': float(y), 'r': float(_particle_radius)})
    return out

async def _playback_loop():
    """Coroutine qui envoie les frames en boucle tant que _playing == True."""
    frame = 0
    try:
        while _playing:
            state = _make_frame(frame % n_frames)
            try:
                update(state)
            except Exception:
                # silent fail si update absent
                pass
            # progress pour la barre (0..100) ; on affiche la progression du cycle
            try:
                pct = int(100.0 * ((frame % n_frames) + 1) / n_frames)
                set_progress(pct)
            except Exception:
                pass
            frame += 1
            await asyncio.sleep(_interval)
    except asyncio.CancelledError:
        pass

def start(interval_seconds=None):
    """Démarre la lecture non bloquante. start() est idempotent."""
    global _playing, _play_task, _interval
    if interval_seconds is not None:
        try:
            _interval = float(interval_seconds)
        except Exception:
            pass
    if _playing:
        return
    _playing = True
    try:
        # lance la coroutine de playback
        _play_task = asyncio.ensure_future(_playback_loop())
    except Exception:
        # fallback : si asyncio non dispo, faire une boucle bloquante basique (peu probable en Pyodide)
        import threading
        def _worker():
            global _playing
            frame = 0
            while _playing:
                state = _make_frame(frame % n_frames)
                try:
                    update(state)
                except Exception:
                    pass
                try:
                    pct = int(100.0 * ((frame % n_frames) + 1) / n_frames)
                    set_progress(pct)
                except Exception:
                    pass
                frame += 1
                time.sleep(_interval)
        t = threading.Thread(target=_worker, daemon=True)
        t.start()

def stop():
    """Arrête la lecture."""
    global _playing, _play_task
    _playing = False
    try:
        if _play_task is not None:
            _play_task.cancel()
    except Exception:
        pass
    try:
        set_progress(0)
    except Exception:
        pass

# Permet d'ajuster la vitesse depuis JS si besoin
def set_interval(seconds):
    global _interval
    try:
        _interval = float(seconds)
    except Exception:
        pass

# Si exécuté directement dans un REPL (hors navigateur), démonstration minimaliste:
if __name__ == "__main__":
    print("Test playback: lancement pendant ~3 secondes (CTRL+C pour stop).")
    start()
    try:
        time.sleep(3.0)
    finally:
        stop()
        print("Terminé.")
