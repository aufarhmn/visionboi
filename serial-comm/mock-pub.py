import zmq
import threading
import random
import time

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://127.0.0.1:5555")

serial_ports = ['COM4', 'COM10']

def generate_random_data(port, zmq_socket):
    print(f"Simulating data generation for {port}...")

    try:
        while True:
            angleValue = random.randint(0, 180)
            distanceValue = random.randint(0, 1000)

            zmq_socket.send_string(f"{port}:{angleValue},{distanceValue}")
            print(f"Published from {port}: {angleValue},{distanceValue}")

            time.sleep(random.randint(3, 10))
    except KeyboardInterrupt:
        print(f"Stopping the data generation for {port}.")

threads = []

for port in serial_ports:
    thread = threading.Thread(target=generate_random_data, args=(port, socket))
    threads.append(thread)
    thread.start()

try:
    for thread in threads:
        thread.join()
except KeyboardInterrupt:
    print("Stopping the program.")
finally:
    socket.close()
    context.term()
