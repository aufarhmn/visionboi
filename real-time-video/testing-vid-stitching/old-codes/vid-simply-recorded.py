import cv2
import numpy as np
import time

# Version 5
# Camera Position: Flipped
# Frames: 1280x720
# Method: Translation
# Feature Extraction: Manual
selected_points_img1 = []
selected_points_img2 = []

def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if param == 'img1':
            selected_points_img1.append((x, y))
            cv2.circle(img1_display, (x, y), 5, (0, 255, 0), -1)
        elif param == 'img2':
            selected_points_img2.append((x, y))
            cv2.circle(img2_display, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow(param, img1_display if param == 'img1' else img2_display)

def manual_calibrate_translation(img1, img2):
    global img1_display, img2_display
    img1_display = img1.copy()
    img2_display = img2.copy()

    cv2.imshow("img1", img1_display)
    cv2.setMouseCallback("img1", select_points, param='img1')

    cv2.imshow("img2", img2_display)
    cv2.setMouseCallback("img2", select_points, param='img2')

    print("Select corresponding points in both images. Press 'q' when done.")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    if len(selected_points_img1) != len(selected_points_img2):
        print("Error: The number of points selected on each image must be the same.")
        return None

    src_pts = np.float32(selected_points_img1).reshape(-1, 1, 2)
    dst_pts = np.float32(selected_points_img2).reshape(-1, 1, 2)

    affine_matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    translation_matrix = np.eye(3)
    translation_matrix[:2] = affine_matrix
    return translation_matrix

def stitch_frames(frame1, frame2, translation_matrix):
    # Determine the final output dimensions based on the maximum height and combined width of both frames
    height = max(frame1.shape[0], frame2.shape[0])
    width = frame1.shape[1] + frame2.shape[1]
    
    # Create a blank stitched frame to hold both images side by side
    stitched_frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Place frame1 on the left side
    stitched_frame[0:frame1.shape[0], 0:frame1.shape[1]] = frame1
    
    # Warp frame2 using the translation matrix
    warped_frame2 = cv2.warpPerspective(frame2, translation_matrix, (width, height))
    
    # Place frame2 to the right side of frame1
    stitched_frame[0:frame2.shape[0], frame1.shape[1]:frame1.shape[1] + frame2.shape[1]] = frame2

    # Blend the overlapping region with feathering
    blend_width = 100  # Width of the blending region
    overlap_start = frame1.shape[1] - blend_width
    
    if overlap_start > 0:
        # Create a linear gradient mask for smoother blending
        mask = np.linspace(0, 1, blend_width).reshape(1, blend_width, 1)
        mask = np.repeat(mask, height, axis=0)

        # Apply the mask to the overlapping regions
        blend_region1 = stitched_frame[:, overlap_start:overlap_start + blend_width].astype(float)
        blend_region2 = warped_frame2[:, overlap_start:overlap_start + blend_width].astype(float)

        blended_region = (blend_region1 * (1 - mask) + blend_region2 * mask).astype(np.uint8)
        
        # Replace the overlap area with the blended result
        stitched_frame[:, overlap_start:overlap_start + blend_width] = blended_region
    
    return stitched_frame

def capture_and_calibrate(video_path1="video1.mp4", video_path2="video2.mp4"):
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    if not cap1.isOpened():
        print(f"Error: Video {video_path1} could not be opened.")
        return None
    if not cap2.isOpened():
        print(f"Error: Video {video_path2} could not be opened.")
        return None

    print("Reading initial frames for calibration...")
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if ret1 and ret2:
        print("Calibration images saved: camera1_image.jpg, camera2_image.jpg")

        print("Calibrating translation manually...")
        translation_matrix = manual_calibrate_translation(frame1, frame2)
        if translation_matrix is None:
            print("Translation calculation failed.")
            return None

        print("Translation Matrix Calculated.")
        return translation_matrix, frame1, frame2
    else:
        print("Error: Failed to capture frames from one or both videos.")
        return None

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

def capture_and_stitch(video_path1="video1.mp4", video_path2="video2.mp4", translation_matrix=None):
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    if translation_matrix is None:
        print("Error: No translation matrix provided.")
        return

    # Ensure frame sizes are consistent
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not (ret1 and ret2):
        print("Error: Could not read initial frames from the videos.")
        return

    frame_width = frame1.shape[1] + frame2.shape[1]
    frame_height = frame1.shape[0]

    # Initialize Video Writer with correct size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('stitched_output.mp4', fourcc, 20.0, (frame_width, frame_height))

    print("Recording stitched video...")

    cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not (ret1 and ret2):
            print("Finished reading video frames or failed to capture frames.")
            break

        stitched_frame = stitch_frames(frame1, frame2, translation_matrix)

        cv2.imshow("Stitched Video", stitched_frame)

        out.write(stitched_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()
    print("Stitched video saved as stitched_output.mp4")

if __name__ == "__main__":
    result = capture_and_calibrate(video_path1="video1.mp4", video_path2="video2.mp4")

    if result is not None:
        translation_matrix, frame1, frame2 = result
        capture_and_stitch(video_path1="video1.mp4", video_path2="video2.mp4", translation_matrix=translation_matrix)