# python script_name.py COM4 --baudrate 9600
import zmq
import serial
import argparse

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://127.0.0.1:5555")

def read_serial_data(port, baudrate, zmq_socket):
    ser = serial.Serial(
        port=port,
        baudrate=baudrate,
        timeout=1
    )

    if not ser.is_open:
        ser.open()

    print(f"Reading serial data from {port} with baudrate {baudrate}...")

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

def main():
    parser = argparse.ArgumentParser(description="Read serial data and publish via ZMQ.")
    parser.add_argument("port", help="Specify a COM port (e.g., COM4)")
    parser.add_argument("--baudrate", type=int, default=115200, help="Set baudrate for serial communication (default: 115200)")
    args = parser.parse_args()

    read_serial_data(args.port, args.baudrate, socket)

    socket.close()
    context.term()

if __name__ == "__main__":
    main()
