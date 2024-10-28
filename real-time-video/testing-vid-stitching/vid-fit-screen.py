import cv2
import numpy as np

# Version 6.2
# Camera Position: Not Flipped
# Frames: 1280x720
# Adding Interactive Stitching with Aligned Normal Screen Size
# Adding Delayed Frame Extraction
# Adding Homography Calculation
dragging = False
x_start, y_start = 0, 0
x_offset = 0

def extract_frame_after_delay(video_path, delay_seconds):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_number = fps * delay_seconds

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not extract frame after {delay_seconds} seconds from {video_path}")
        return None
    return frame

def calculate_homography_from_delayed_frames(video1_path, video2_path, delay_seconds):
    img1 = extract_frame_after_delay(video1_path, delay_seconds)
    img2 = extract_frame_after_delay(video2_path, delay_seconds)

    if img1 is None or img2 is None:
        print("Error loading frames.")
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

def stitch_video_frames(video1_path, video2_path, output_path, H, frame_shape):
    global x_offset
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

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
    video1_path = "video1.mp4"  # Left video
    video2_path = "video2.mp4"  # Right video
    output_path = "stitched_output.mp4"

    # Delay in seconds to extract frames for homography calculation
    H, frame_shape = calculate_homography_from_delayed_frames(video1_path, video2_path, delay_seconds=1)

    if H is not None:
        stitch_video_frames(video1_path, video2_path, output_path, H, frame_shape)
        print(f"Stitched video saved as {output_path}")
    else:
        print("Failed to compute homography. Video stitching aborted.")
