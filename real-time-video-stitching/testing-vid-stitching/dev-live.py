import cv2
import numpy as np

# TESTING NEEDED
# Version 6.3.2
# Camera Position: Not Flipped
# Frames: 1280x720
# Adding Interactive Stitching with Aligned Normal Screen Size
# Adding Delayed Frame Extraction
# Adding Homography Calculation
# Using Live Camera
# TODO: MARKING LOCATION

# PERFORMANCE ISSUE WHEN EVERY FRAME IS PROCESSED TO UNDISTORT

dragging = False
x_start, y_start = 0, 0
x_offset = 0

selected_match = -1
good_matches = []

# Replace with computed DIM, K, and D
DIM=(1920, 1080)
K=np.array([[1336.4921919893268, 0.0, 908.8692451447415], [0.0, 1340.0325057841567, 466.10657810710325], [0.0, 0.0, 1.0]])
D=np.array([[0.019735576742026275], [0.2600139930451915], [-1.1954519511846629], [0.8408293625138502]])

def undistort_fisheye_frame(frame):
    h, w = frame.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_frame

def extract_frame_after_delay(camera_index):
    cap = cv2.VideoCapture(camera_index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DIM[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DIM[1])

    if not cap.isOpened():
        print(f"Error: Camera {camera_index} could not be opened.")
        return None

    print("Warming up camera...")
    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not warm up the camera.")
            cap.release()
            return None

    print("Camera warmed up.")
    ret, frame = cap.read()
    cap.release()

    undistorted_frame = undistort_fisheye_frame(frame)

    return undistorted_frame

def calculate_homography_from_delayed_frames(camera1_index, camera2_index):
    img1 = extract_frame_after_delay(camera1_index)
    img2 = extract_frame_after_delay(camera2_index)

    # cv2.imwrite("img1.jpg", img1)
    # cv2.imwrite("img2.jpg", img2)

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

            return H, img1.shape
        else:
            print("Not enough matches found to compute homography.")
            return None, None
    else:
        return None, None 

def draw_matches(img1, kp1, img2, kp2):
    global matched_keypoints_img
    global good_matches

    screen_width = 1920
    screen_height = 1080

    matched_keypoints_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    def resize_and_show():
        img_height, img_width = matched_keypoints_img.shape[:2]

        scale_width = screen_width / img_width
        scale_height = screen_height / img_height
        scale = min(scale_width, scale_height)

        if scale < 1:
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            matched_keypoints_img_resized = cv2.resize(matched_keypoints_img, (new_width, new_height))
        else:
            matched_keypoints_img_resized = matched_keypoints_img

        cv2.imshow("Matched Keypoints", matched_keypoints_img_resized)

    cv2.namedWindow("Matched Keypoints", cv2.WINDOW_NORMAL)
    params = {
        "kp1": kp1,
        "kp2": kp2,
        "img1": img1,
        "img2": img2
    }
    cv2.setMouseCallback("Matched Keypoints", mouse_click, params)

    while True:
        resize_and_show()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyWindow("Matched Keypoints")
            return True 
        elif key == ord('d'):
            delete_selected_match(params)
            resize_and_show()

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


def stitch_video_frames(camera1_index, camera2_index, H, frame_shape):
    global x_offset
    x_offset = 0
    cap1 = cv2.VideoCapture(camera1_index)
    cap2 = cv2.VideoCapture(camera2_index)

    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap1.isOpened():
        print(f"Error: Camera {camera1_index} could not be opened.")
        return None
    if not cap2.isOpened():
        print(f"Error: Camera {camera2_index} could not be opened.")
        return None

    cv2.namedWindow("Stitched Video")
    cv2.setMouseCallback("Stitched Video", mouse_drag)

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        height, width = frame1.shape[:2]
        stitched_width = width * 2
        canvas = np.zeros((height, stitched_width, 3), dtype=np.uint8)

        canvas[0:frame1.shape[0], 0:frame1.shape[1]] = frame1

        warped_frame2 = cv2.warpPerspective(frame2, H, (stitched_width, height))

        overlap_mask = np.zeros_like(canvas, dtype=np.uint8)
        overlap_mask[0:frame1.shape[0], 0:frame1.shape[1]] = 1  

        alpha = 0.5
        overlap_area = (overlap_mask & (warped_frame2 > 0)).astype(np.uint8)
        blended_region = cv2.addWeighted(canvas, alpha, warped_frame2, 1 - alpha, 0)

        non_overlap_region = np.where(warped_frame2 > 0, warped_frame2, canvas)

        final_stitched = np.where(overlap_area, blended_region, non_overlap_region)

        visible_width = frame1.shape[1]
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
    camera1_index = 2 # Index of the left camera
    camera2_index = 0 # Index of the right camera

    H, frame_shape = calculate_homography_from_delayed_frames(camera1_index, camera2_index)

    print(H)

    if H is not None:
        stitch_video_frames(camera1_index, camera2_index, H, frame_shape)
    else:
        print("Failed to compute homography! Video stitching aborted!")
