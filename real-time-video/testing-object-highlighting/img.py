import cv2
import numpy as np

def angle_to_coordinates(angle, distance, frame_width, frame_height):

    x_pixel = int((angle + 90) / 180 * frame_width)

    y_pixel = frame_height // 2

    return x_pixel, y_pixel

stitched_image_path = "stitched_image.jpg"
stitched_image = cv2.imread(stitched_image_path)

if stitched_image is None:
    print("Failed to load stitched image. Check the file path.")
else:
    frame_height, frame_width, _ = stitched_image.shape

    dummy_lidar_data = [
        (-60, 1.5),  # Left side
        (0, 2.0),    # Straight
        (45, 1.0),   # Right side
        (90, 1.2)    # Far-right
    ]

    for angle, distance in dummy_lidar_data:
        x, y = angle_to_coordinates(angle, distance, frame_width, frame_height)
        
        # Circle indicate the LIDAR detection
        cv2.circle(stitched_image, (x, y), 10, (0, 0, 255), -1)

        # Display the distance as text near the circle
        label = f"{distance:.1f}m"
        cv2.putText(stitched_image, label, (x + 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Stitched Image with LIDAR Highlights", stitched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
