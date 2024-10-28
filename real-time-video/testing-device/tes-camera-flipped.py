import cv2

def check_camera(camera_index1=2, camera_index2=0):
    cap = cv2.VideoCapture(camera_index1)
    cap2 = cv2.VideoCapture(camera_index2)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera {camera_index1} default resolution: {int(width)}x{int(height)}")

    width2 = cap2.get(cv2.CAP_PROP_FRAME_WIDTH)
    height2 = cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera {camera_index2} default resolution: {int(width2)}x{int(height2)}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_width1 = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height1 = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_width2 = cap2.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height2 = cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    print(f"Camera {camera_index1} resolution set to: {int(actual_width1)}x{int(actual_height1)}")
    print(f"Camera {camera_index2} resolution set to: {int(actual_width2)}x{int(actual_height2)}")

    if not cap.isOpened():
        print(f"Error: Camera with index {camera_index1} could not be opened.")
        return
    
    if not cap2.isOpened():
        print(f"Error: Camera with index {camera_index2} could not be opened.")
        return

    print(f"Camera {camera_index1} and Camera {camera_index2} are working. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        ret2, frame2 = cap2.read()

        if not ret or not ret2:
            print("Error: Failed to capture frame from one or both cameras.")
            break

        frame = cv2.flip(frame, 0)
        frame2 = cv2.flip(frame2, 0)

        cv2.imshow(f'Camera {camera_index1}', frame)
        cv2.imshow(f'Camera {camera_index2}', frame2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    check_camera()
