import zmq
import serial
import threading

# Configuration for ZMQ and Serial Communication
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://127.0.0.1:5555")

serial_ports = ['COM4', 'COM10']
baudrate = 115200

def read_serial_data(port, baudrate, zmq_socket):
    ser = serial.Serial(
        port=port,
        baudrate=baudrate,
        timeout=1
    )

    if not ser.is_open:
        ser.open()

    print(f"Reading serial data from {port}...")

    try:
        while True:
            if ser.in_waiting > 0:
                data = ser.readline().decode('utf-8').strip()
                angleValue, distanceValue = data.split(', ')
                angleValue = int(angleValue)
                distanceValue = int(distanceValue)
                
                zmq_socket.send_string(f"{port}:{angleValue},{distanceValue}")
                print(f"Published from {port}: {angleValue},{distanceValue}")
    except KeyboardInterrupt:
        print(f"Stopping the reading from {port}.")
    finally:
        ser.close()

threads = []

for port in serial_ports:
    thread = threading.Thread(target=read_serial_data, args=(port, baudrate, socket))
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
