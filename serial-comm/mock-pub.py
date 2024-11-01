# python script_name.py COM4 COM10
import zmq
import threading
import random
import time
import argparse

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://127.0.0.1:5555")

def generate_random_data(port, zmq_socket):
    try:
        while True:
            angleValue = random.randint(0, 180)
            distanceValue = random.randint(0, 1000)

            zmq_socket.send_string(f"{port}:{angleValue},{distanceValue}")

            time.sleep(random.randint(3, 10))
    except KeyboardInterrupt:
        print(f"Stopping the data generation for {port}.")

def main():
    parser = argparse.ArgumentParser(description="Simulate LIDAR data on specified COM ports.")
    parser.add_argument("ports", nargs="+", help="Specify COM ports (e.g., COM4 COM10)")
    args = parser.parse_args()
    
    threads = []

    for port in args.ports:
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

if __name__ == "__main__":
    main()
