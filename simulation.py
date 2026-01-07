from js import update, set_progress
import asyncio

running = False

async def run():
    global running
    running = True

    points = []

    for i in range(101):
        if not running:
            break

        x = i / 100
        y = 0.2 + 0.6 * x   # droite simple

        points.append({"x": x, "y": y})

        update(points)
        set_progress(i)

        await asyncio.sleep(0.03)

def start():
    asyncio.create_task(run())

def stop():
    global running
    running = False
