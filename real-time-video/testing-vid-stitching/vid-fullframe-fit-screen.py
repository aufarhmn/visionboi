import cv2
import numpy as np

# TESTING NEEDED
# Version 6.2.1
# Camera Position: Not Flipped
# Frames: 1280x720
# Adding Interactive Stitching with Aligned Normal Screen Size
# Adding Delayed Frame Extraction
# Adding Homography Calculation
# Using Live Camera
dragging = False
x_start, y_start = 0, 0
x_offset = 0

def extract_frame_after_delay(camera_index):
    cap = cv2.VideoCapture(camera_index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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
    return frame

def calculate_homography_from_delayed_frames(camera1_index, camera2_index):
    img1 = extract_frame_after_delay(camera1_index)
    img2 = extract_frame_after_delay(camera2_index)

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

    src_pts = np.float32([kp1[m.queryIdx].pt for m, n in matches if m.distance < 0.75 * n.distance]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m, n in matches if m.distance < 0.75 * n.distance]).reshape(-1, 1, 2)

    if len(src_pts) >= 8: 
        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)

        inliers_src_pts = src_pts[mask.ravel() == 1]
        inliers_dst_pts = dst_pts[mask.ravel() == 1]

        H, _ = cv2.findHomography(inliers_dst_pts, inliers_src_pts, cv2.RANSAC, 5.0)

        return H, img1.shape
    else:
        print("Not enough matches found to compute homography.")
        return None, None

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

if __name__ == "__main__":
    camera1_index = 2 # Index of the left camera
    camera2_index = 0 # Index of the right camera

    H, frame_shape = calculate_homography_from_delayed_frames(camera1_index, camera2_index)

    if H is not None:
        stitch_video_frames(camera1_index, camera2_index, H, frame_shape)
    else:
        print("Failed to compute homography! Video stitching aborted!")
