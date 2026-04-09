import cv2
import numpy as np

# Red wraps around hue=0/180 in HSV, so we need two ranges
RED_LOWER_1 = np.array([0, 120, 70])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([160, 120, 70])
RED_UPPER_2 = np.array([180, 255, 255])

# Pink: lower saturation than red, higher value
PINK_LOWER = np.array([140, 40, 120])
PINK_UPPER = np.array([170, 255, 255])

MIN_CONTOUR_AREA = 100  # ignore noise smaller than this (pixels)


def segment_color(image: np.ndarray, color: str) -> np.ndarray:
    """Create a binary mask of pixels matching the given color in a BGR image.

    Args:
        image: BGR image.
        color: "red" or "pink".

    Returns:
        Binary mask (uint8, 0 or 255) same H x W as input.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if color == "red":
        mask = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1) | cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
    elif color == "pink":
        mask = cv2.inRange(hsv, PINK_LOWER, PINK_UPPER)
    else:
        raise ValueError(f"Unknown color: {color}. Supported: 'red', 'pink'")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_object(image: np.ndarray, color: str) -> dict | None:
    """Detect and segment an object of the given color in a BGR image.

    Args:
        image: BGR image.
        color: "red" or "pink".

    Returns a dict with:
        centroid: (u, v) normalized to [0, 1]
        contour: the largest contour (numpy array of points)
        mask: binary segmentation mask
    Or None if no object of that color found.
    """
    mask = segment_color(image, color)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    h, w = image.shape[:2]
    u = (M["m10"] / M["m00"]) / w
    v = (M["m01"] / M["m00"]) / h

    return {
        "centroid": (u, v),
        "contour": largest,
        "mask": mask,
    }


# Convenience wrappers
def detect_red_object(image: np.ndarray) -> dict | None:
    return detect_object(image, "red")


def detect_pink_object(image: np.ndarray) -> dict | None:
    return detect_object(image, "pink")
