"""Quick test: run red + pink object segmentation on laptop front camera."""
import cv2
from perception import detect_object

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera")
    exit(1)

print("Showing camera feed. Hold red and/or pink objects in front of the camera.")
print("Green outline = red object, Magenta outline = pink object.")
print("Press 'q' to quit, 'm' to toggle mask view.")

show_mask = False

COLORS = {
    "red":  {"contour": (0, 255, 0),   "text": (0, 255, 0)},    # green drawing
    "pink": {"contour": (255, 0, 255), "text": (255, 0, 255)},   # magenta drawing
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    detected_any = False

    for color, style in COLORS.items():
        result = detect_object(frame, color)
        if result is not None:
            detected_any = True
            u, v = result["centroid"]
            contour = result["contour"]
            mask = result["mask"]
            h, w = frame.shape[:2]
            cx, cy = int(u * w), int(v * h)

            if show_mask:
                overlay = display.copy()
                overlay[mask > 0] = style["contour"]
                display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)

            cv2.drawContours(display, [contour], -1, style["contour"], 2)
            cv2.circle(display, (cx, cy), 6, style["contour"], -1)
            cv2.putText(display, f"{color} ({u:.2f}, {v:.2f})", (cx + 15, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, style["text"], 2)

    if not detected_any:
        cv2.putText(display, "No objects detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Perception Test", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("m"):
        show_mask = not show_mask

cap.release()
cv2.destroyAllWindows()
