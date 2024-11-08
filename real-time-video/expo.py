import cv2
import numpy as np
import zmq
import threading
import math
import argparse

# Version 7
# One Module Only
# LIDAR Attached
# For Capstone Expo Showcase
dragging = False
x_start = 0
x_offset = 0

lidar_points = []

# Camera Parameters
DIM = (1920, 1080)
K = np.array([[1336.4921919893268, 0.0, 908.8692451447415], 
              [0.0, 1340.0325057841567, 466.10657810710325], 
              [0.0, 0.0, 1.0]])
D = np.array([[0.019735576742026275], 
              [0.2600139930451915], 
              [-1.1954519511846629], 
              [0.8408293625138502]])

def zmq_subscriber():
    global lidar_points
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://127.0.0.1:5555")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    try:
        while True:
            data = socket.recv_string()
            port_and_data = data.split(':')
            
            if len(port_and_data) == 2:
                lidar_points.clear()
                angleValue, distanceValue = port_and_data[1].split(',')

                angle = float(angleValue)
                distance = float(distanceValue)
                
                if angle >= 10 and angle <= 170:
                    math.radians(angle)

                    y = distance * math.sin(angle)
                    y = float(y / 1000)

                    lidar_points.append(abs(y))
                
    except KeyboardInterrupt:
        print("Stopping the subscriber.")
    finally:
        socket.close()
        context.term()

def show_video_with_distance(video_path):
    global x_offset, lidar_points

    cap1 = cv2.VideoCapture(video_path)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cv2.namedWindow("Video with Distance", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Video with Distance", mouse_drag)

    if not cap1.isOpened():
        print("Error: Couldn't open the video source.")
        cap1.release()
        cv2.destroyAllWindows()
        return

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)

    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break

        undistorted_frame1 = cv2.remap(frame1, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        window_width = cv2.getWindowImageRect("Video with Distance")[2]
        
        width = undistorted_frame1.shape[1]
        x_offset = max(0, min(x_offset, width - window_width))
        visible_frame = undistorted_frame1[:, x_offset:x_offset + window_width]

        for y in lidar_points:
            if y < 1:
                warning_message = f"Warning! Object detected at {y:.2f} meters"
                position = (50, 50)
                cv2.putText(visible_frame, warning_message, position, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Video with Distance", visible_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cv2.destroyAllWindows()

def mouse_drag(event, x, y, flags, param):
    global dragging, x_start, x_offset
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        x_start = x

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            x_offset += x - x_start
            x_start = x

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show from one module with the LIDAR attached")
    parser.add_argument("video_path", type=int, help="Path to video file or camera index.")
    args = parser.parse_args()
    
    # ZMQ Subscriber Thread
    zmq_thread = threading.Thread(target=zmq_subscriber)
    zmq_thread.daemon = True
    zmq_thread.start()

    # Display Video with Distance
    show_video_with_distance(args.video_path)
