import cv2
import numpy as np
import zmq
import threading
import math

# Version 6.3.1
# Camera Position: Not Flipped
# Frames: 1280x720
# Adding Interactive Stitching with Aligned Normal Screen Size
# Adding Delayed Frame Extraction
# Adding Homography Calculation
# Using Recorded Videos

# ISSUE: BLANK SPACE ON CANVAS COMPROMISES DOT NOTATION
# FIX 1: OVERLAP WIDTH AND NON BLANK CANVAS CALCULATED
# TODO: USE THE VALUE CALCULATED ABOVE TO REMOVE BLANK SPACE AND FIX DOT NOTATION

dragging = False
dragging = False
x_start, y_start = 0, 0
x_offset = 0

selected_match = -1
good_matches = []

lidar_points = []

def angle_to_coordinates(port, angle, distance, stitched_width, frame_height, overlap_width):
    angle_to_radians = math.radians(angle)
    x = math.cos(angle_to_radians) * distance
    
    y = frame_height // 2

    if port == 'COM4':
        x_pixel = int(x + overlap_width / 2)
    elif port == 'COM10':
        x_pixel = int(x + stitched_width // 2 - overlap_width / 2)
    else:
        print("Invalid port.")
        return None, None
    
    x_pixel = max(0, min(x_pixel, stitched_width - 1))
    return x_pixel, y

def zmq_subscriber(stitched_width, frame_height, overlap_width):
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
                port = port_and_data[0]
                angleValue, distanceValue = port_and_data[1].split(',')

                angle = float(angleValue)
                distance = float(distanceValue)
                
                x, y = angle_to_coordinates(port, angle, distance, stitched_width, frame_height, overlap_width)
                if x is not None and y is not None:
                    lidar_points.clear()
                    lidar_points.append((x, y))
            else:
                print("Invalid data received.")
    except KeyboardInterrupt:
        print("Stopping the subscriber.")
    finally:
        socket.close()
        context.term()

def extract_frame_after_delay(video_path):
    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"Error: Video {video_path} could not be opened.")
        return None
    
    print("Warming up video...")
    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not warm up the video.")
            cap.release()
            return None
        
    print("Video warmed up.")
    ret, frame = cap.read()
    cap.release()
    return frame

def calculate_homography_from_delayed_frames(video1_path, video2_path):
    img1 = extract_frame_after_delay(video1_path)
    img2 = extract_frame_after_delay(video2_path)

    if img1 is None or img2 is None:
        print("Error loading frames!")
        return None, None

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    global good_matches
    good_matches = []
    src_pts = []
    dst_pts = []
    for m, n in matches:
        if m.distance < 0.80 * n.distance:
            good_matches.append(m)
            src_pts.append(kp1[m.queryIdx].pt)
            dst_pts.append(kp2[m.trainIdx].pt)

    src_pts = np.float32(src_pts).reshape(-1, 1, 2)
    dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)

    if draw_matches(img1, kp1, img2, kp2): 
        if len(src_pts) >= 8:
            F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)

            inliers_src_pts = src_pts[mask.ravel() == 1]
            inliers_dst_pts = dst_pts[mask.ravel() == 1]

            H, _ = cv2.findHomography(inliers_dst_pts, inliers_src_pts, cv2.RANSAC, 5.0)

            calculate_overlap_width(H, img1, img2)

            return H, img1.shape
        else:
            print("Not enough matches found to compute homography.")
            return None, None
    else:
        return None, None 

def calculate_overlap_width(H, img1, img2):
    global overlap_width
    height, width = img1.shape[:2]

    img2_corners = np.float32([
        [0, 0], [width, 0], [width, height], [0, height]
    ]).reshape(-1, 1, 2)

    projected_corners = cv2.perspectiveTransform(img2_corners, H)

    overlap_x_coords = [pt[0][0] for pt in projected_corners]

    overlap_width = width - min(overlap_x_coords)

    print(f"Calculated overlap width: {overlap_width}")
    return overlap_width

def draw_matches(img1, kp1, img2, kp2):
    global matched_keypoints_img
    global good_matches

    matched_keypoints_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.namedWindow("Matched Keypoints")
    params = {
        "kp1": kp1,
        "kp2": kp2,
        "img1": img1,
        "img2": img2
    }
    cv2.setMouseCallback("Matched Keypoints", mouse_click, params)

    while True:
        cv2.imshow("Matched Keypoints", matched_keypoints_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyWindow("Matched Keypoints")
            return True 
        elif key == ord('d'):
            delete_selected_match(params)  

def delete_selected_match(params):
    global selected_match, good_matches

    if selected_match != -1 and selected_match < len(good_matches):
        good_matches.pop(selected_match)
        print(f"Deleted match {selected_match}")

        img1 = params["img1"]
        kp1 = params["kp1"]
        img2 = params["img2"]
        kp2 = params["kp2"]
        
        global matched_keypoints_img
        matched_keypoints_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        selected_match = -1

def mouse_click(event, x, y, flags, param):
    global selected_match

    kp1 = param["kp1"]
    kp2 = param["kp2"]
    img1 = param["img1"]

    if event == cv2.EVENT_LBUTTONDOWN:
        min_distance = float("inf")
        for idx, match in enumerate(good_matches):
            pt1 = np.array(kp1[match.queryIdx].pt)
            pt2 = np.array(kp2[match.trainIdx].pt) + np.array([img1.shape[1], 0])
            line_center = (pt1 + pt2) / 2

            distance = np.linalg.norm(line_center - np.array([x, y]))
            if distance < min_distance:
                min_distance = distance
                selected_match = idx

        if min_distance < 10:
            print(f"Selected match {selected_match}")

def stitch_video_frames(video1_path, video2_path, H, frame_shape):
    global x_offset, lidar_points, non_blank_width
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    height, width = frame_shape[:2]
    stitched_width = width * 2
    cv2.namedWindow("Stitched Video")

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Error: Couldn't read the video frames.")
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
        return

    warped_frame2 = cv2.warpPerspective(frame2, H, (stitched_width, height))

    non_blank_width = int(frame1.shape[1] + warped_frame2.shape[1] - overlap_width)
    print(f"Stitched width without blank canvas: {non_blank_width}")

    cv2.setMouseCallback("Stitched Video", mouse_drag)

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        canvas = np.zeros((height, stitched_width, 3), dtype=np.uint8)
        canvas[0:frame1.shape[0], 0:frame1.shape[1]] = frame1
        warped_frame2 = cv2.warpPerspective(frame2, H, (stitched_width, height))

        alpha = 0.5
        overlap_area = (canvas > 0) & (warped_frame2 > 0)
        blended_region = cv2.addWeighted(canvas, alpha, warped_frame2, 1 - alpha, 0)
        non_overlap_region = np.where(warped_frame2 > 0, warped_frame2, canvas)
        final_stitched = np.where(overlap_area, blended_region, non_overlap_region)

        for (x, y) in lidar_points:
            cv2.circle(final_stitched, (x, y), 10, (0, 0, 255), -1)

        visible_width = width
        max_offset = stitched_width - visible_width
        x_offset = max(0, min(x_offset, max_offset))
        visible_region = final_stitched[:, x_offset:x_offset + visible_width]
        
        cv2.imshow("Stitched Video", visible_region)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
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
    video1_path = "./assets/left-vid.mp4"  # LEFT CAMERA
    video2_path = "./assets/right-vid.mp4"  # RIGHT CAMERA

    H, frame_shape = calculate_homography_from_delayed_frames(video1_path, video2_path)

    print(H)
    print(frame_shape)

    if H is not None:
        stitched_width = frame_shape[1] * 2
        zmq_thread = threading.Thread(target=zmq_subscriber, args=(stitched_width, frame_shape[0], 100))
        zmq_thread.daemon = True
        zmq_thread.start()

        stitch_video_frames(video1_path, video2_path, H, frame_shape)
    else:
        print("Failed to compute homography! Video stitching aborted!")
