import cv2
import numpy as np
import csv

VIDEO_PATH = "cp1/pekka.mp4"
OUTPUT_CSV = "cp1/pekka_builtins.csv"

LOWER = np.array([110, 50, 50])
UPPER = np.array([150, 255, 255])

MIN_AREA = 150

def deriv(t, x):
    if len(x) < 3:
        z = np.zeros_like(x)
        return z,z,z,z
    v = np.gradient(x, t)
    a = np.gradient(v, t)
    j = np.gradient(a, t)
    s = np.gradient(j, t)
    return v,a,j,s

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
dt = 1.0 / fps

times = []
xs = []
ys = []

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER, UPPER)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest = None
    best_area = 0

    for c in cnts:
        area = cv2.contourArea(c)
        if area > MIN_AREA and area > best_area:
            biggest = c
            best_area = area

    if biggest is not None:
        M = cv2.moments(biggest)
        if M["m00"] != 0:
            cx = M["m10"]/M["m00"]
            cy = M["m01"]/M["m00"]

            xs.append(cx)
            ys.append(cy)
            times.append(frame_id * dt)

            cv2.circle(frame, (int(cx), int(cy)), 8, (0,255,0), -1)

    # ------ DISPLAY VIDEO AT CORRECT SPEED ------
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        break
    # ---------------------------------------------

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
