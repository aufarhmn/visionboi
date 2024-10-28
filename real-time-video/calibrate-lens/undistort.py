import cv2
import numpy as np

# Replace with computed DIM, K, and D
DIM=(1920, 1080)
K=np.array([[1336.4921919893268, 0.0, 908.8692451447415], [0.0, 1340.0325057841567, 466.10657810710325], [0.0, 0.0, 1.0]])
D=np.array([[0.019735576742026275], [0.2600139930451915], [-1.1954519511846629], [0.8408293625138502]])

def undistort(img_path):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.imwrite("undistorted1.jpg", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Path to image
    undistort("1.jpg")