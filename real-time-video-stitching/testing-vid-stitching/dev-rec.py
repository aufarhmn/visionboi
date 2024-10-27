import cv2
import numpy as np

# Version 6.3.1
# Camera Position: Not Flipped
# Frames: 1280x720
# Adding Interactive Stitching with Aligned Normal Screen Size
# Adding Delayed Frame Extraction
# Adding Homography Calculation
# Using Recorded Videos
dragging = False
dragging = False
x_start, y_start = 0, 0
x_offset = 0

selected_match = -1
good_matches = []

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

            return H, img1.shape
        else:
            print("Not enough matches found to compute homography.")
            return None, None
    else:
        return None, None 

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
    global x_offset
    x_offset = 0
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap1.isOpened():
        print(f"Error: Video {video1_path} could not be opened.")
        return None
    if not cap2.isOpened():
        print(f"Error: Video {video2_path} could not be opened.")
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
    video1_path = "video1.mp4"  # LEFT CAMERA
    video2_path = "video2.mp4"  # RIGHT CAMERA

    H, frame_shape = calculate_homography_from_delayed_frames(video1_path, video2_path)

    print(H)

    if H is not None:
        stitch_video_frames(video1_path, video2_path, H, frame_shape)
    else:
        print("Failed to compute homography! Video stitching aborted!")
