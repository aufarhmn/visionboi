import cv2
import numpy as np

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
    
    warped_frame1[0:frame2.shape[0], 0:frame2.shape[1]] = frame2
    return warped_frame1

def main():
    cap1 = cv2.VideoCapture(0)  
    cap2 = cv2.VideoCapture(2) 

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if ret1 and ret2:
        print("Calibrating homography based on the captured images...")
        
        homography = calibrate_homography(frame1, frame2)
        if homography is None:
            print("Homography calculation failed.")
            return
        print("Homography Matrix Calculated:")
        print(homography)
        
        stitched_frame = stitch_frames(frame1, frame2, homography)

        cv2.imshow("Stitched Image", stitched_frame)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to capture images from one or both cameras.")

    cap1.release()
    cap2.release()

if __name__ == "__main__":
    main()
