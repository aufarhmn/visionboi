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

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Feature Matches", img_matches)
    cv2.waitKey(0)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    height, width = img2.shape[:2]
    warped_img1 = cv2.warpPerspective(img1, H, (width + img1.shape[1], height))

    blend_region = warped_img1[0:height, 0:width]

    mask = np.zeros_like(blend_region, dtype=np.uint8)
    mask[blend_region > 0] = 1

    alpha = 0.5
    blended_region = cv2.addWeighted(blend_region, alpha, img2, 1 - alpha, 0)

    warped_img1[0:height, 0:width] = blended_region

    cv2.imshow("Stitched Image", warped_img1)
    cv2.imwrite(output_path, warped_img1)
    print(f"Stitched image saved as {output_path}")
    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    image1_path = "1a.jpg"
    image2_path = "1b.jpg"

    stitch_images_homography(image1_path, image2_path)
