import cv2

# Taking image from camera directly
# Works well when cameras are placed inside with no extreme condition
def capture_and_save_images(camera_index1=2, camera_index2=0, filename1="camera1_image.jpg", filename2="camera2_image.jpg"):
    cap1 = cv2.VideoCapture(camera_index1)
    cap2 = cv2.VideoCapture(camera_index2)

    if not cap1.isOpened():
        print(f"Error: Camera {camera_index1} could not be opened.")
        return
    if not cap2.isOpened():
        print(f"Error: Camera {camera_index2} could not be opened.")
        return

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if ret1 and ret2:
        cv2.imwrite(filename1, frame1)
        cv2.imwrite(filename2, frame2)
        print(f"Images saved: {filename1}, {filename2}")
    else:
        print("Error: Failed to capture frames from one or both cameras.")

    cap1.release()
    cap2.release()

if __name__ == "__main__":
    capture_and_save_images()
