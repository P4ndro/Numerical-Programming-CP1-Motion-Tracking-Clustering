import cv2
import numpy as np
import csv

VIDEO_PATH = "cp1/pekka.mp4"             
OUTPUT_CSV = "cp1/pekka_tracking_scratch.csv"

# HSV color range for PEKKA (tweak if needed)
LOWER_PEKKA = np.array([110, 50, 50])   # approx blue/purple
UPPER_PEKKA = np.array([150, 255, 255])

MIN_PIXELS = 200   # minimum mask pixels to accept detection

def centroid(mask):
    # CALCULATION: Find all white pixels in mask
    ys, xs = np.where(mask == 255)
    if len(xs) == 0:
        return None
    # CALCULATION: Manual centroid = mean of all x and y coordinates
    return float(np.mean(xs)), float(np.mean(ys))

def deriv(t, x):
    if len(x) < 3:
        z = np.zeros_like(x)
        return z, z, z, z
    # CALCULATION: Velocity = first derivative of position
    v = np.gradient(x, t)
    # CALCULATION: Acceleration = second derivative (derivative of velocity)
    a = np.gradient(v, t)
    # CALCULATION: Jerk = third derivative (derivative of acceleration)
    j = np.gradient(a, t)
    # CALCULATION: Jounce = fourth derivative (derivative of jerk)
    s = np.gradient(j, t)
    return v, a, j, s

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Could not open video " + VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
# CALCULATION: Time step per frame (seconds)
dt = 1.0 / fps

ts = []
xs = []
ys = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Color-based mask for PEKKA
    mask = cv2.inRange(hsv, LOWER_PEKKA, UPPER_PEKKA)

    if mask.sum() > MIN_PIXELS:
        c = centroid(mask)
        if c is not None:
            # CALCULATION: Current time = frame number × time step
            t = frame_idx * dt
            ts.append(t)
            xs.append(c[0])
            ys.append(c[1])

    frame_idx += 1

cap.release()

ts = np.array(ts)
xs = np.array(xs)
ys = np.array(ys)

# CALCULATION: Compute derivatives for x and y coordinates
vx, ax, jx, sx = deriv(ts, xs)
vy, ay, jy, sy = deriv(ts, ys)

# CALCULATION: Speed magnitude = sqrt(vx² + vy²)
speed  = np.sqrt(vx**2 + vy**2)
# CALCULATION: Acceleration magnitude = sqrt(ax² + ay²)
accel  = np.sqrt(ax**2 + ay**2)
# CALCULATION: Jerk magnitude = sqrt(jx² + jy²)
jerk   = np.sqrt(jx**2 + jy**2)
# CALCULATION: Jounce magnitude = sqrt(sx² + sy²)
jounce = np.sqrt(sx**2 + sy**2)

with open(OUTPUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t", "x", "y", "speed", "accel", "jerk", "jounce"])
    for i in range(len(ts)):
        w.writerow([ts[i], xs[i], ys[i],
                    speed[i], accel[i], jerk[i], jounce[i]])

print("Saved PEKKA tracking CSV:", OUTPUT_CSV)
