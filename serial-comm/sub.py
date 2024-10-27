import zmq

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://127.0.0.1:5555")
socket.setsockopt_string(zmq.SUBSCRIBE, "")

print("Waiting for data...")

try:
    while True:
        data = socket.recv_string()
        port_and_data = data.split(':')
        
        if len(port_and_data) == 2:
            port = port_and_data[0]
            angleValue, distanceValue = port_and_data[1].split(',')

            print(f"Received: Port={port}, Angle={angleValue}, Distance={distanceValue}")
        else:
            print("Invalid data received.")
except KeyboardInterrupt:
    print("Stopping the subscriber.")
finally:
    socket.close()
    context.term()
