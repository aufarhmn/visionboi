import cv2

# Taking image from camera stream
# Works well when cameras are placed outside
def capture_from_stream(camera_index1=2, camera_index2=0, filename1="camera1_image.jpg", filename2="camera2_image.jpg"):
    cap1 = cv2.VideoCapture(camera_index1)
    cap2 = cv2.VideoCapture(camera_index2)

    if not cap1.isOpened():
        print(f"Error: Camera {camera_index1} could not be opened.")
        return
    if not cap2.isOpened():
        print(f"Error: Camera {camera_index2} could not be opened.")
        return

    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Warming up cameras...")

    for _ in range(30):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not (ret1 and ret2):
            print("Error: Failed to capture frames during warm-up.")
            cap1.release()
            cap2.release()
            return

        cv2.imshow(f'Camera {camera_index1}', frame1)
        cv2.imshow(f'Camera {camera_index2}', frame2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Capturing images...")
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
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_from_stream()
