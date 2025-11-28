import cv2
import numpy as np
import csv

VIDEO_PATH = "goblingang.mp4"
OUTPUT_CSV = "goblin_gang_motion.csv"

# Background subtractor (NOT a pretrained model)
bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)

MIN_AREA = 60        # minimum moving area (goblin size)
MAX_GOBLINS = 5      # expected goblins
MAX_DIST = 50        # max distance for ID assignment

# ========================================
# Derivative function
# ========================================
def deriv(t, x):
    if len(x) < 3:
        z = np.zeros_like(x)
        return z,z,z,z
    v = np.gradient(x, t)
    a = np.gradient(v, t)
    j = np.gradient(a, t)
    s = np.gradient(j, t)
    return v,a,j,s

# ========================================
# Assign IDs by nearest centroid
# ========================================
def assign_ids(prev, curr):
    if prev is None:
        return list(range(len(curr))), curr.copy()

    prev = np.array(prev)
    curr = np.array(curr)

    d = np.linalg.norm(prev[:,None,:] - curr[None,:,:], axis=2)

    used = set()
    mapping = {}

    for i in range(len(prev)):
        j = np.argmin(d[i])
        if j not in used and d[i][j] < MAX_DIST:
            mapping[i] = j
            used.add(j)

    next_id = len(prev)
    for j in range(len(curr)):
        if j not in used:
            mapping[next_id] = j
            next_id += 1

    ordered = [curr[mapping[k]] for k in sorted(mapping.keys())]
    return sorted(mapping.keys()), ordered


# ========================================
# Video setup
# ========================================
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
dt = 1 / fps

cv2.namedWindow("Goblin Tracking (Motion Based)", cv2.WINDOW_NORMAL)

tracks = {}
prev_centroids = None
frame_idx = 0

# ========================================
# Processing Loop
# ========================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    mask = bg.apply(frame)

    mask = cv2.GaussianBlur(mask, (7,7), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=2)
    mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)[1]

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []

    for c in cnts:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            centroids.append((cx, cy))

    centroids = sorted(centroids, key=lambda x: x[0])[:MAX_GOBLINS]

    if centroids:
        ids, ordered = assign_ids(prev_centroids, centroids)
        prev_centroids = ordered
        t = frame_idx * dt

        for obj_id, (x,y) in zip(ids, ordered):
            tracks.setdefault(obj_id, []).append((t, x, y))
            cv2.circle(frame, (int(x), int(y)), 8, (0,255,0), -1)
            cv2.putText(frame, f"ID {obj_id}", (int(x)+8, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # ---- SHOW VIDEO AT CORRECT SPEED ----
    cv2.imshow("Goblin Tracking (Motion Based)", frame)
    if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# ========================================
# Save CSV
# ========================================
rows = []

for gob_id, series in tracks.items():
    series = np.array(series)
    t = series[:,0]
    x = series[:,1]
    y = series[:,2]

    vx, ax, jx, sx = deriv(t,x)
    vy, ay, jy, sy = deriv(t,y)

    speed  = np.sqrt(vx**2 + vy**2)
    accel  = np.sqrt(ax**2 + ay**2)
    jerk   = np.sqrt(jx**2 + jy**2)
    jounce = np.sqrt(sx**2 + sy**2)

    for i in range(len(t)):
        rows.append([
            gob_id, t[i], x[i], y[i],
            speed[i], accel[i], jerk[i], jounce[i]
        ])

with open(OUTPUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id","t","x","y","speed","accel","jerk","jounce"])
    w.writerows(rows)

print("Saved motion-based Goblin Gang tracking →", OUTPUT_CSV)
