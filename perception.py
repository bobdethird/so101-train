import cv2
import numpy as np

# Red wraps around hue=0/180 in HSV, so we need two ranges
RED_LOWER_1 = np.array([0, 120, 70])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([160, 120, 70])
RED_UPPER_2 = np.array([180, 255, 255])

MIN_CONTOUR_AREA = 100  # ignore noise smaller than this (pixels)


def segment_red_object(image: np.ndarray) -> np.ndarray:
    """Create a binary mask of all red pixels in a BGR image.

    Returns:
        Binary mask (uint8, 0 or 255) same H x W as input.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
    mask2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
    mask = mask1 | mask2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_red_object(image: np.ndarray) -> dict | None:
    """Detect and segment a red object in a BGR image.

    Returns a dict with:
        centroid: (u, v) normalized to [0, 1]
        contour: the largest contour (numpy array of points)
        mask: binary segmentation mask
    Or None if no red object found.
    """
    mask = segment_red_object(image)

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
