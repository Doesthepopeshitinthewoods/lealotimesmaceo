# simulation.py
# Script Pyodide — robuste : attend que window.update / window.set_progress soient prêts,
# gère start/stop, et protège les appels JS contre les erreurs.

import asyncio
import math

# accès au module js (Pyodide)
import js as _js

# flag pour contrôler l'exécution
running = False

async def _wait_for_js_funcs(timeout=5.0, interval=0.05):
    """Attendre que window.update et window.set_progress soient définis.
    Renvoie (update, set_progress) ou lève RuntimeError après timeout (secondes)."""
    t = 0.0
    while t < timeout:
        if hasattr(_js, "update") and hasattr(_js, "set_progress"):
            return _js.update, _js.set_progress
        await asyncio.sleep(interval)
        t += interval
    raise RuntimeError("Fonctions JS 'update' et/ou 'set_progress' introuvables")

async def run(steps: int = 100, delay: float = 0.03):
    """Coroutine principale.
    - steps : nombre de pas (100 = 0..1 par pas de 0.01)
    - delay : pause non-bloquante entre chaque pas (s)
    """
    global running
    if running:
        # déjà en cours : ne lance pas une seconde instance
        try:
            _js.console.warn("Simulation déjà en cours — run() ignoré")
        except Exception:
            pass
        return

    # Attendre les fonctions JS (utile si Python est démarré avant que JS expose les handlers)
    try:
        update, set_progress = await _wait_for_js_funcs()
    except Exception as e:
        try:
            _js.console.error("Impossible d'obtenir update/set_progress :", str(e))
        except Exception:
            pass
        return

    running = True
    points = []

    try:
        for i in range(steps + 1):
            if not running:
                try:
                    _js.console.log("Arrêt demandé — sortie de la boucle")
                except Exception:
                    pass
                break

            x = i / steps
            # Exemple de fonction : une droite légèrement modifiée (tu peux mettre n'importe quoi)
            y = 0.2 + 0.6 * x

            points.append({"x": float(x), "y": float(y)})

            # Appels JS protégés
            try:
                # Pyodide convertit automatiquement la liste/dict en objets JS
                update(points)
                # on envoie la progression en pourcentage 0..100
                set_progress(int(round(100 * i / steps)))
            except Exception as e:
                # Log côté JS si possible
                try:
                    _js.console.error("Erreur lors de l'appel update/set_progress :", str(e))
                except Exception:
                    pass

            # pause non bloquante
            await asyncio.sleep(delay)

    finally:
        running = False
        # s'assurer que la progression affiche 100% à la fin
        try:
            set_progress(100)
        except Exception:
            pass

def start():
    """Démarre la simulation en créant une tâche asyncio (si possible).
    Si l'environnement n'a pas de boucle active, lancez plutôt pyodide.runPythonAsync('run()') depuis JS."""
    try:
        asyncio.create_task(run())
    except Exception as e:
        # s'il n'y a pas de boucle active ici, on log et on quitte (JS devrait appeler run() via runPythonAsync)
        try:
            _js.console.warn("Impossible de créer une task asyncio ici — appelez pyodide.runPythonAsync('run()') depuis JS. Détail :", str(e))
        except Exception:
            pass

def stop():
    """Demande l'arrêt de la boucle (flag)."""
    global running
    running = False
