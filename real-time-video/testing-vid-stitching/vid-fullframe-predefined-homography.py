import cv2
import numpy as np

# Version 7
# Camera Position: Not Flipped
# Frames: 1280x720
# Adding Interactive Stitching
# Adding Predefined Homography
# Adding Dragging Functionality
dragging = False
start_x, start_y = 0, 0
offset_x, offset_y = 0, 0

def mouse_event_handler(event, x, y, flags, param):
    global dragging, start_x, start_y, offset_x, offset_y

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        start_x, start_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        offset_x += x - start_x
        offset_y += y - start_y
        start_x, start_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

def stitch_predefined_homography(video1_path, video2_path, H, frame_shape):
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    cv2.namedWindow("Interactive Stitched Video")
    cv2.setMouseCallback("Interactive Stitched Video", mouse_event_handler)

    global offset_x, offset_y
    offset_x, offset_y = 0, 0

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

        stitched_height, stitched_width = final_stitched.shape[:2]
        canvas_dragged = np.zeros_like(final_stitched)

        start_x = max(0, min(stitched_width - frame_shape[1], offset_x))
        start_y = max(0, min(stitched_height - frame_shape[0], offset_y))

        end_x = start_x + frame_shape[1]
        end_y = start_y + frame_shape[0]
        cropped_view = final_stitched[start_y:end_y, start_x:end_x]

        cv2.imshow("Interactive Stitched Video", cropped_view)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video1_path = "video1.mp4"  # Left video
    video2_path = "video2.mp4"  # Right video

    H = np.array([
        [7.51800017e-01, 7.79062995e-02, 5.86119297e+02],
        [-6.62387220e-02, 1.01489968e+00, 3.26206448e+01],
        [-3.02035622e-04, 8.65434608e-05, 1.00000000e+00]
    ])

    frame_shape = (720, 1280)

    stitch_predefined_homography(video1_path, video2_path, H, frame_shape)
