import cv2
import numpy as np

def stitch_images_epipolar_constraint(image1_path, image2_path, output_path="stitched_image.jpg"):
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None or img2 is None:
        print("Error loading images.")
        return

    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    if height1 != height2 or width1 != width2:
        target_height = min(height1, height2)
        target_width = min(width1, width2)
        img1 = cv2.resize(img1, (target_width, target_height), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, (target_width, target_height), interpolation=cv2.INTER_AREA)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0))
    img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0))

    cv2.imwrite("img1_kp.jpg", img1_kp)
    cv2.imwrite("img2_kp.jpg", img2_kp)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    src_pts = np.float32([kp1[m.queryIdx].pt for m, n in matches if m.distance < 0.75 * n.distance]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m, n in matches if m.distance < 0.75 * n.distance]).reshape(-1, 1, 2)

    if len(src_pts) >= 8:
        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)
        inliers_src_pts = src_pts[mask.ravel() == 1]
        inliers_dst_pts = dst_pts[mask.ravel() == 1]

        good_matches = [cv2.DMatch(i, i, 0) for i in range(len(inliers_src_pts))]
        img_inlier_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite("inlier_matches.jpg", img_inlier_matches)

        H, mask = cv2.findHomography(inliers_dst_pts, inliers_src_pts, cv2.RANSAC, 5.0)

        height, width = img1.shape[:2]
        stitched_width = img1.shape[1] + img2.shape[1]
        canvas = np.zeros((height, stitched_width, 3), dtype=np.uint8)
        canvas[0:img1.shape[0], 0:img1.shape[1]] = img1

        warped_img2 = cv2.warpPerspective(img2, H, (stitched_width, height))

        alpha = 0.5
        overlap_area = (canvas > 0) & (warped_img2 > 0)
        blended_region = cv2.addWeighted(canvas, alpha, warped_img2, 1 - alpha, 0)

        final_stitched = np.where(overlap_area, blended_region, warped_img2)

        cv2.imwrite(output_path, final_stitched)
        print(f"Stitched image saved as {output_path}")
    else:
        print("Not enough good matches to compute a reliable homography.")

if __name__ == "__main__":
    image1_path = "1.jpg"  # Left image
    image2_path = "2.jpg"  # Right image
    stitch_images_epipolar_constraint(image1_path, image2_path)
