import cv2
import json
from pathlib import Path

CAMERA_INDEX = 1
WINDOW = "Top Camera"
CONFIG_PATH = Path(__file__).parent / "camera_overlay.json"
GRAB_RADIUS = 10

# ── State ──
box = [220, 140, 420, 340]
lines = []
mode = "idle"
drag_idx = -1
drag_offset = (0, 0)
line_pt1 = None
mouse_pos = (0, 0)


def load_config():
    global box, lines
    if CONFIG_PATH.exists():
        cfg = json.loads(CONFIG_PATH.read_text())
        box[:] = cfg.get("box", box)
        lines[:] = cfg.get("lines", lines)
        print(f"Loaded overlay from {CONFIG_PATH}")


def save_config():
    CONFIG_PATH.write_text(json.dumps({"box": box, "lines": lines}, indent=2))
    print(f"Saved overlay to {CONFIG_PATH}")


def box_corners():
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def near_corner(px, py):
    for i, (cx, cy) in enumerate(box_corners()):
        if (px - cx) ** 2 + (py - cy) ** 2 < GRAB_RADIUS ** 2:
            return i
    return -1


def inside_box(px, py):
    x1, y1 = min(box[0], box[2]), min(box[1], box[3])
    x2, y2 = max(box[0], box[2]), max(box[1], box[3])
    return x1 <= px <= x2 and y1 <= py <= y2


def near_line_endpoint(px, py):
    """Return (line_index, 0|1) if near an endpoint, else (-1, -1)."""
    for i, ln in enumerate(lines):
        for ep in range(2):
            lx, ly = ln[ep * 2], ln[ep * 2 + 1]
            if (px - lx) ** 2 + (py - ly) ** 2 < GRAB_RADIUS ** 2:
                return i, ep
    return -1, -1


def on_mouse(event, x, y, flags, _):
    global mode, drag_idx, drag_offset, line_pt1, mouse_pos

    mouse_pos = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        if mode == "line_p1":
            line_pt1 = (x, y)
            mode = "line_p2"
            return
        if mode == "line_p2":
            lines.append([line_pt1[0], line_pt1[1], x, y])
            line_pt1 = None
            mode = "idle"
            return

        ci = near_corner(x, y)
        if ci >= 0:
            mode = "drag_corner"
            drag_idx = ci
            return

        li, ep = near_line_endpoint(x, y)
        if li >= 0:
            mode = "drag_line_ep"
            drag_idx = li * 2 + ep
            return

        if inside_box(x, y):
            mode = "drag_box"
            drag_offset = (x - box[0], y - box[1])

    elif event == cv2.EVENT_MOUSEMOVE:
        if mode == "drag_corner":
            if drag_idx == 0:
                box[0], box[1] = x, y
            elif drag_idx == 1:
                box[2], box[1] = x, y
            elif drag_idx == 2:
                box[2], box[3] = x, y
            elif drag_idx == 3:
                box[0], box[3] = x, y
        elif mode == "drag_box":
            w, h = box[2] - box[0], box[3] - box[1]
            box[0] = x - drag_offset[0]
            box[1] = y - drag_offset[1]
            box[2] = box[0] + w
            box[3] = box[1] + h
        elif mode == "drag_line_ep":
            li, ep = divmod(drag_idx, 2)
            lines[li][ep * 2] = x
            lines[li][ep * 2 + 1] = y

    elif event == cv2.EVENT_LBUTTONUP:
        if mode in ("drag_corner", "drag_box", "drag_line_ep"):
            mode = "idle"

    elif event == cv2.EVENT_RBUTTONDOWN:
        if mode in ("line_p1", "line_p2"):
            mode = "idle"
            line_pt1 = None


# ── Main ──
load_config()

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera at index {CAMERA_INDEX}")

cv2.namedWindow(WINDOW)
cv2.setMouseCallback(WINDOW, on_mouse)

HELP = (
    "Drag corners/edges to reshape  |  Drag inside box to move  |  "
    "Drag line endpoints to adjust\n"
    "'l' add line   'u' undo line   's' save   'q' save & quit   ESC cancel"
)
print(HELP)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Box
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    for cx, cy in box_corners():
        cv2.circle(frame, (cx, cy), GRAB_RADIUS - 2, (0, 0, 255), -1)

    # Lines
    for ln in lines:
        cv2.line(frame, (ln[0], ln[1]), (ln[2], ln[3]), (255, 100, 0), 2)
        cv2.circle(frame, (ln[0], ln[1]), GRAB_RADIUS - 2, (255, 100, 0), -1)
        cv2.circle(frame, (ln[2], ln[3]), GRAB_RADIUS - 2, (255, 100, 0), -1)

    # In-progress line preview
    if mode == "line_p2" and line_pt1:
        cv2.line(frame, line_pt1, mouse_pos, (255, 100, 0), 1)
        cv2.circle(frame, line_pt1, 4, (255, 100, 0), -1)

    # Status bar
    status = {
        "idle": "Drag to edit | 'l' line  'u' undo  's' save  'q' quit",
        "line_p1": "Click first point (right-click or ESC to cancel)",
        "line_p2": "Click second point (right-click or ESC to cancel)",
        "drag_corner": "Dragging corner...",
        "drag_box": "Moving box...",
        "drag_line_ep": "Dragging line endpoint...",
    }.get(mode, "")
    cv2.rectangle(frame, (0, 0), (len(status) * 8, 24), (0, 0, 0), -1)
    cv2.putText(frame, status, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imshow(WINDOW, frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        save_config()
        break
    elif key == ord("s"):
        save_config()
    elif key == ord("l"):
        mode = "line_p1"
        line_pt1 = None
    elif key == ord("u") and lines:
        lines.pop()
    elif key == 27:
        mode = "idle"
        line_pt1 = None

cap.release()
cv2.destroyAllWindows()
