# simulation.py
# Test ultra-simple : tracer une droite y = ax + b
# Compatible Pyodide (aucune dépendance externe)

import asyncio

# récupération de la fonction JS update()
try:
    from js import update, set_progress
except Exception:
    # fallback si exécuté hors navigateur
    def update(data):
        print(data)
    def set_progress(p):
        print("progress:", p)

# paramètres de la droite
a = 0.8   # pente
b = 0.1   # ordonnée à l'origine

n_points = 100

def compute_line():
    """
    Génère les points de la droite y = a x + b
    avec x dans [0,1]
    """
    points = []
    for i in range(n_points):
        x = i / (n_points - 1)
        y = a * x + b
        points.append({
            "x": float(x),
            "y": float(y)
        })
    return points

async def run():
    # barre de chargement (fake mais utile pour test)
    for p in range(0, 101, 20):
        try:
            set_progress(p)
        except Exception:
            pass
        await asyncio.sleep(0.05)

    # calcul + affichage
    data = compute_line()
    update(data)

# point d’entrée appelé depuis le bouton HTML
def start():
    asyncio.ensure_future(run())

def stop():
    try:
        set_progress(0)
    except Exception:
        pass
