"""Quick test: run red object segmentation on laptop front camera."""
import cv2
from perception import detect_red_object

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera")
    exit(1)

print("Showing camera feed. Hold a red object in front of the camera.")
print("Green outline = segmented object. Press 'q' to quit, 'm' to toggle mask view.")

show_mask = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detect_red_object(frame)
    display = frame.copy()

    if result is not None:
        u, v = result["centroid"]
        contour = result["contour"]
        mask = result["mask"]
        h, w = frame.shape[:2]
        cx, cy = int(u * w), int(v * h)

        if show_mask:
            # Show segmentation mask as red overlay on the image
            overlay = display.copy()
            overlay[mask > 0] = (0, 0, 255)
            display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)

        # Draw contour outline
        cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)
        # Draw centroid
        cv2.circle(display, (cx, cy), 6, (0, 255, 0), -1)
        cv2.putText(display, f"({u:.2f}, {v:.2f})", (cx + 15, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(display, "No red object", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Perception Test", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("m"):
        show_mask = not show_mask

cap.release()
cv2.destroyAllWindows()
