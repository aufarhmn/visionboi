import cv2
import numpy as np

def stitch_images_homography(image1_path, image2_path, output_path="stitched_image.jpg"):
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None or img2 is None:
        print("Error loading images.")
        return

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0))
    img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0))

    cv2.imshow("Image 1 - Keypoints", img1_kp)
    cv2.imshow("Image 2 - Keypoints", img2_kp)
    cv2.waitKey(0)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2) 

    good_matches = []
    ratio_thresh = 0.75
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    print(f"Number of good matches after Lowe's ratio test: {len(good_matches)}")

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Good Matches after Lowe's Ratio Test", img_matches)
    cv2.waitKey(0)

    if len(good_matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) 
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)  

        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        height, width = img1.shape[:2]
        stitched_width = img1.shape[1] + img2.shape[1]
        canvas = np.zeros((height, stitched_width, 3), dtype=np.uint8)

        canvas[0:img1.shape[0], 0:img1.shape[1]] = img1

        warped_img2 = cv2.warpPerspective(img2, H, (stitched_width, height))

        overlap_mask = np.zeros_like(canvas, dtype=np.uint8)
        overlap_mask[0:img1.shape[0], 0:img1.shape[1]] = 1  

        alpha = 0.5
        overlap_area = (overlap_mask & (warped_img2 > 0)).astype(np.uint8)
        blended_region = cv2.addWeighted(canvas, alpha, warped_img2, 1 - alpha, 0)

        non_overlap_region = np.where(warped_img2 > 0, warped_img2, canvas)

        final_stitched = np.where(overlap_area, blended_region, non_overlap_region)

        cv2.imshow("Stitched Image", final_stitched)
        cv2.imwrite(output_path, final_stitched)
        print(f"Stitched image saved as {output_path}")
        cv2.waitKey(0)
    else:
        print("Not enough good matches to compute a reliable homography.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    image1_path = "1a.jpg"  # Left image
    image2_path = "1b.jpg"  # Right image

    stitch_images_homography(image1_path, image2_path)
