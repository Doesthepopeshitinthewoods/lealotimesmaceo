# simulation.py
# Exemple autonome de simulation compatible avec Stage.html
# Définit simulate(steps=..., dt=...) -> dict sérialisable JSON
# Ne dépend que de la stdlib (math) pour fonctionner directement dans Pyodide.

import math

def simulate(steps=400, dt=0.05):
    """
    Génère une trajectoire r(t) et des vecteurs P, E, B au cours du temps.
    - steps : nombre de pas temporels
    - dt    : pas de temps
    Retourne un dict contenant des listes Python:
      't', 'rx','ry','rz',
      'Px','Py','Pz',
      'Ex','Ey','Ez',
      'Bx','By','Bz'
    """
    t = [i * dt for i in range(steps)]

    # paramètres de la trajectoire et champs (exemples)
    R0 = 1.2
    omega = 2.0
    decay = 0.0008

    rx = []
    ry = []
    rz = []

    Px = []; Py = []; Pz = []
    Ex = []; Ey = []; Ez = []
    Bx = []; By = []; Bz = []

    for i, ti in enumerate(t):
        amp = R0 * math.exp(-decay * i)
        x = amp * math.cos(omega * ti)
        y = amp * math.sin(omega * ti)
        z = 0.25 * math.sin(0.5 * omega * ti)

        rx.append(x); ry.append(y); rz.append(z)

        # Polarisation P : exemple simple perpendiculaire à r
        Px.append(-0.5 * y)
        Py.append( 0.5 * x)
        Pz.append( 0.05 * math.sin(1.2 * ti))

        # Champ électrique E : mélange onde temporelle + dépendance spatiale faible
        Ex.append(0.6 * math.sin(1.5 * ti) + 0.15 * x)
        Ey.append(0.45 * math.cos(1.2 * ti) + 0.10 * y)
        Ez.append(0.25 * math.sin(0.8 * ti) + 0.05 * z)

        # Champ magnétique B : champ tournant simple
        Bx.append(0.2 * math.cos(0.9 * ti) - 0.05 * y)
        By.append(0.2 * math.sin(0.9 * ti) + 0.05 * x)
        Bz.append(0.05 * math.cos(1.3 * ti))

    return {
        't': t,
        'rx': rx, 'ry': ry, 'rz': rz,
        'Px': Px, 'Py': Py, 'Pz': Pz,
        'Ex': Ex, 'Ey': Ey, 'Ez': Ez,
        'Bx': Bx, 'By': By, 'Bz': Bz
    }

# petit test si exécuté directement (utile en local)
if __name__ == "__main__":
    out = simulate(steps=10, dt=0.1)
    print("Clés produites :", list(out.keys()))
    print("Longueur t:", len(out['t']), "exemple t:", out['t'][:3])
    print("Exemple rx:", out['rx'][:3])
