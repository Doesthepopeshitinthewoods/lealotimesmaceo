from js import update, set_progress
import asyncio

# Flag partagé pour contrôler l'exécution
running = False

async def run():
    """Coroutine principale — ajoute des points et notifie l'UI JS.
    - Met à jour le canvas via `update(points)`
    - Met à jour la progression via `set_progress(percent)`

    Remarques d'utilisation :
    - Dans l'UI, on peut appeler `pyodide.runPythonAsync('run()')` pour lancer proprement
      la coroutine depuis JS (et récupérer une Promise JS).
    - `start()` crée une tâche si vous préférez démarrer côté Python.
    - `stop()` arrête proprement la boucle.
    """
    global running
    if running:
        # déjà en cours
        return
    running = True

    points = []

    try:
        for i in range(101):
            if not running:
                break

            x = i / 100
            y = 0.2 + 0.6 * x   # simple ligne

            # on append un dict Python — Pyodide le convertira pour JS
            points.append({"x": x, "y": y})

            # notifications vers JS
            try:
                update(points)
                set_progress(i)
            except Exception as e:
                # s'il y a un problème JS, on log en Python (console JS est disponible)
                try:
                    from js import console
                    console.error('Erreur en appelant update/set_progress:', e)
                except Exception:
                    pass

            # pause non-bloquante
            await asyncio.sleep(0.03)
    finally:
        running = False


def start():
    """Démarre la simulation en arrière-plan (création d'une task asyncio)."""
    # Si une boucle est présente, créer une tâche ; sinon, run() peut être appelé
    # via pyodide.runPythonAsync('run()') depuis JS.
    try:
        asyncio.create_task(run())
    except Exception:
        # en cas d'environnement sans boucle active, on peut fallback sur runPythonAsync
        pass


def stop():
    """Demande l'arrêt de la boucle."""
    global running
    running = False
