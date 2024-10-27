import cv2
import numpy as np
import time

# Version 3
# Camera Position: Flipped
# Frames: 1280x720
# Method: Full Homography
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

def manual_calibrate_homography(img1, img2):
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

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

def stitch_frames(frame1, frame2, homography):
    warped_frame1 = cv2.warpPerspective(frame1, homography, (frame1.shape[1] + frame2.shape[1], frame1.shape[0]))
    stitched_frame = np.copy(warped_frame1)
    stitched_frame[0:frame2.shape[0], 0:frame2.shape[1]] = frame2
    return stitched_frame

def capture_and_calibrate(camera_index1=2, camera_index2=0):
    cap1 = cv2.VideoCapture(camera_index1)
    cap2 = cv2.VideoCapture(camera_index2)

    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap1.isOpened():
        print(f"Error: Camera {camera_index1} could not be opened.")
        return None
    if not cap2.isOpened():
        print(f"Error: Camera {camera_index2} could not be opened.")
        return None

    print("Warming up cameras...")
    for _ in range(30):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not (ret1 and ret2):
            print("Error: Failed to capture frames during warm-up.")
            cap1.release()
            cap2.release()
            return None

        cv2.imshow(f'Camera {camera_index1}', frame1)
        cv2.imshow(f'Camera {camera_index2}', frame2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Capturing images for calibration...")
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if ret1 and ret2:
        frame1 = cv2.flip(frame1, 0)
        frame2 = cv2.flip(frame2, 0)

        cv2.imwrite("camera1_image.jpg", frame1)
        cv2.imwrite("camera2_image.jpg", frame2)
        print("Calibration images saved: camera1_image.jpg, camera2_image.jpg")

        print("Calibrating homography manually...")
        homography = manual_calibrate_homography(frame1, frame2)
        if homography is None:
            print("Homography calculation failed.")
            return None

        print("Homography Matrix Calculated.")
        return homography, frame1, frame2
    else:
        print("Error: Failed to capture frames from one or both cameras.")
        return None

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

def capture_and_stitch(camera_index1=2, camera_index2=0, homography=None):
    cap1 = cv2.VideoCapture(camera_index1)
    cap2 = cv2.VideoCapture(camera_index2)

    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if homography is None:
        print("Error: No homography matrix provided.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('stitched_output.mp4', fourcc, 20.0, (2560, 720))

    print("Recording stitched video for 5 seconds...")

    start_time = time.time()

    while cap1.isOpened() and cap2.isOpened() and (time.time() - start_time) < 5:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not (ret1 and ret2):
            print("Failed to capture frames from one or both cameras.")
            break

        frame1 = cv2.flip(frame1, 0)
        frame2 = cv2.flip(frame2, 0)

        stitched_frame = stitch_frames(frame1, frame2, homography)

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
    result = capture_and_calibrate(camera_index1=2, camera_index2=0)

    if result is not None:
        homography, frame1, frame2 = result
        capture_and_stitch(camera_index1=2, camera_index2=0, homography=homography)
