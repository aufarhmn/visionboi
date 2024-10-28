import cv2
import numpy as np

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

def manual_feature_matching(image1_path, image2_path, output_path="stitched_image_manual.jpg"):
    global img1_display, img2_display
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None or img2 is None:
        print("Error loading images.")
        return

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
        return

    src_pts = np.float32(selected_points_img1).reshape(-1, 1, 2)
    dst_pts = np.float32(selected_points_img2).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    height, width = img2.shape[:2]
    warped_img1 = cv2.warpPerspective(img1, H, (width + img1.shape[1], height))

    blend_region = warped_img1[0:height, 0:width]
    mask = np.zeros_like(blend_region, dtype=np.uint8)
    mask[blend_region > 0] = 1
    alpha = 0.5
    blended_region = cv2.addWeighted(blend_region, alpha, img2, 1 - alpha, 0)
    warped_img1[0:height, 0:width] = blended_region

    cv2.imshow("Stitched Image (Manual Matching)", warped_img1)
    cv2.imwrite(output_path, warped_img1)
    print(f"Stitched image saved as {output_path}")
    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    image1_path = "1a.jpg"
    image2_path = "1b.jpg"

    manual_feature_matching(image1_path, image2_path)
