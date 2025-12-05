import cv2
import numpy as np
import csv

VIDEO_PATH = "cp1/goblingang.mp4"
OUTPUT_CSV = "cp1/goblin_gang_scratch.csv"

# HSV color range for goblins (adjust if needed)
LOWER = np.array([110, 50, 50])   # approx blue/purple
UPPER = np.array([150, 255, 255])

MIN_AREA = 60        # minimum blob area (goblin size)
MAX_GOBLINS = 5      # expected goblins
MAX_DIST = 50        # max distance for ID assignment

# ========================================
# Manual centroid computation (from scratch)
# ========================================
def centroid_from_mask(mask, contour):
    """Compute centroid manually using np.where and np.mean"""
    # Create a mask for just this contour
    mask_single = np.zeros_like(mask)
    cv2.drawContours(mask_single, [contour], -1, 255, -1)
    
    # CALCULATION: Find all white pixels in this blob
    ys, xs = np.where(mask_single == 255)
    if len(xs) == 0:
        return None
    
    # CALCULATION: Manual centroid = mean of all x and y coordinates
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    return cx, cy

# ========================================
# Derivative function (same as scratch1)
# ========================================
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

# ========================================
# Manual ID assignment (nearest neighbor)
# ========================================
def assign_ids(prev, curr):
    """Assign IDs by finding nearest centroids between frames"""
    if prev is None:
        return list(range(len(curr))), curr.copy()

    prev = np.array(prev)
    curr = np.array(curr)

    # CALCULATION: Distance matrix = Euclidean distance between all previous and current centroids
    d = np.linalg.norm(prev[:,None,:] - curr[None,:,:], axis=2)

    used = set()
    mapping = {}

    # Match each previous ID to nearest current centroid
    for i in range(len(prev)):
        # CALCULATION: Find index of nearest current centroid
        j = np.argmin(d[i])
        # CALCULATION: Check if distance is within threshold
        if j not in used and d[i][j] < MAX_DIST:
            mapping[i] = j
            used.add(j)

    # Assign new IDs to unmatched current centroids
    next_id = len(prev)
    for j in range(len(curr)):
        if j not in used:
            mapping[next_id] = j
            next_id += 1

    # Return IDs and ordered centroids
    ordered = [curr[mapping[k]] for k in sorted(mapping.keys())]
    return sorted(mapping.keys()), ordered

# ========================================
# Video setup
# ========================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Could not open video " + VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
# CALCULATION: Time step per frame (seconds)
dt = 1.0 / fps

cv2.namedWindow("Multi-Object Tracking (Scratch)", cv2.WINDOW_NORMAL)

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

    # Convert to HSV for color-based detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Color-based mask (from scratch approach)
    mask = cv2.inRange(hsv, LOWER, UPPER)

    # Find separate blobs (using contours to identify individual objects)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []

    # Compute centroid manually for each blob
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        
        # Manual centroid computation (from scratch)
        cent = centroid_from_mask(mask, c)
        if cent is not None:
            centroids.append(cent)

    # Limit to max goblins, sort by x-coordinate for consistency
    centroids = sorted(centroids, key=lambda x: x[0])[:MAX_GOBLINS]

    if centroids:
        # Assign IDs using nearest neighbor matching
        ids, ordered = assign_ids(prev_centroids, centroids)
        prev_centroids = ordered
        # CALCULATION: Current time = frame number × time step
        t = frame_idx * dt

        # Store tracking data for each object
        for obj_id, (x, y) in zip(ids, ordered):
            tracks.setdefault(obj_id, []).append((t, x, y))
            
            # Visualize tracking
            cv2.circle(frame, (int(x), int(y)), 8, (0, 255, 0), -1)
            cv2.putText(frame, f"ID {obj_id}", (int(x) + 8, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display video
    cv2.imshow("Multi-Object Tracking (Scratch)", frame)
    if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# ========================================
# Compute derivatives and save CSV
# ========================================
rows = []

for obj_id, series in tracks.items():
    series = np.array(series)
    t = series[:, 0]
    x = series[:, 1]
    y = series[:, 2]

    # CALCULATION: Compute derivatives for x and y coordinates
    vx, ax, jx, sx = deriv(t, x)
    vy, ay, jy, sy = deriv(t, y)

    # CALCULATION: Speed magnitude = sqrt(vx² + vy²)
    speed = np.sqrt(vx**2 + vy**2)
    # CALCULATION: Acceleration magnitude = sqrt(ax² + ay²)
    accel = np.sqrt(ax**2 + ay**2)
    # CALCULATION: Jerk magnitude = sqrt(jx² + jy²)
    jerk = np.sqrt(jx**2 + jy**2)
    # CALCULATION: Jounce magnitude = sqrt(sx² + sy²)
    jounce = np.sqrt(sx**2 + sy**2)

    # Add rows for this object
    for i in range(len(t)):
        rows.append([
            obj_id, t[i], x[i], y[i],
            speed[i], accel[i], jerk[i], jounce[i]
        ])

# Write CSV file
with open(OUTPUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "t", "x", "y", "speed", "accel", "jerk", "jounce"])
    w.writerows(rows)

print("Saved multi-object tracking CSV:", OUTPUT_CSV)

