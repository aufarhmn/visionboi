import cv2
import numpy as np

# Version 2
# Camera Position: Flipped
# Frames: 1280x720
# Method: Full Homography
# Feature Extraction: ORB
def calibrate_homography(frame1, frame2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(frame1, None)
    kp2, des2 = orb.detectAndCompute(frame2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

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

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if ret1 and ret2:
        frame1 = cv2.flip(frame1, 0)
        frame2 = cv2.flip(frame2, 0)

        print("Calibrating homography...")
        homography = calibrate_homography(frame1, frame2)
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

    print("Starting real-time video stitching...")

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not (ret1 and ret2):
            print("Failed to capture frames from one or both cameras.")
            break

        frame1 = cv2.flip(frame1, 0)
        frame2 = cv2.flip(frame2, 0)

        stitched_frame = stitch_frames(frame1, frame2, homography)

        cv2.imshow("Stitched Video", stitched_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    result = capture_and_calibrate(camera_index1=2, camera_index2=0)

    if result is not None:
        homography, frame1, frame2 = result

        cv2.imwrite("camera1_image.jpg", frame1)
        cv2.imwrite("camera2_image.jpg", frame2)
        print("Calibration images saved: camera1_image.jpg, camera2_image.jpg")

        capture_and_stitch(camera_index1=2, camera_index2=0, homography=homography)
