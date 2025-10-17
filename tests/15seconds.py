import time

initial_time = time.time()

running = True

while running:
    if time.time() - initial_time > 15:
        running = False