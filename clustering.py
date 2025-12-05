import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# Try different CSV paths
CSV_PATHS = ["goblin_gang_builtins.csv", "goblin_gang_motion.csv", "cp1/goblin_gang_builtins.csv"]
CSV_PATH = None
for path in CSV_PATHS:
    if os.path.exists(path):
        CSV_PATH = path
        break

if CSV_PATH is None:
    raise FileNotFoundError(f"Could not find CSV file. Tried: {CSV_PATHS}")

print(f"Loading data from: {CSV_PATH}")   

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(CSV_PATH)

# group per goblin id
groups = {gid: g for gid, g in df.groupby("id")}
print("Found goblins:", list(groups.keys()))

# ======================================================
# SPEED-BASED FEATURE EXTRACTION
# ======================================================
def extract_speed_features(g):
    """Extract speed-based features only."""
    speed = g["speed"].values
    
    # Speed-only features
    features = [
        np.mean(speed),           # Average speed
        np.std(speed),            # Speed variability
        np.max(speed),            # Peak speed
        np.median(speed),         # Median speed (robust to outliers)
        np.percentile(speed, 75) - np.percentile(speed, 25),  # Speed IQR (spread)
        np.std(speed) / (np.mean(speed) + 1e-6),  # Speed CV (coefficient of variation)
    ]
    return features

X = []
ids = []
feature_names = [
    "Mean Speed", "Speed Std", "Max Speed", "Median Speed", "Speed IQR", "Speed CV"
]

for gid, g in groups.items():
    X.append(extract_speed_features(g))
    ids.append(gid)

X = np.array(X)
print(f"Feature matrix shape: {X.shape} ({len(feature_names)} speed-based features)")
print(f"Features: {', '.join(feature_names)}")

# ======================================================
# FEATURE SCALING
# ======================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================================================
# K-MEANS CLUSTERING (Speed-based)
# ======================================================
n_goblins = len(groups)
k = min(3, max(2, n_goblins // 2))  # 2-3 clusters typically

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_scaled)

print(f"\n{'='*60}")
print(f"K-MEANS CLUSTERING (k={k}) - Speed-Based")
print(f"{'='*60}")
for gid, lab in zip(ids, labels_kmeans):
    mean_speed = X[ids.index(gid), 0]  # Mean speed is first feature
    print(f"Goblin {gid:2d} -> Cluster {lab} (Avg Speed: {mean_speed:.1f})")

# Analyze cluster characteristics
print(f"\n{'='*60}")
print("CLUSTER CHARACTERISTICS")
print(f"{'='*60}")
for cluster_id in range(k):
    cluster_mask = labels_kmeans == cluster_id
    cluster_goblins = [ids[i] for i in range(len(ids)) if cluster_mask[i]]
    cluster_speeds = [X[ids.index(gid), 0] for gid in cluster_goblins]
    print(f"\nCluster {cluster_id} ({len(cluster_goblins)} goblins): {cluster_goblins}")
    print(f"  Average Speed: {np.mean(cluster_speeds):.1f} ± {np.std(cluster_speeds):.1f}")
    print(f"  Speed Range: {np.min(cluster_speeds):.1f} - {np.max(cluster_speeds):.1f}")

# ======================================================
# DBSCAN (density-based, speed-based)
# ======================================================
dbscan = DBSCAN(eps=1.5, min_samples=1)
labels_db = dbscan.fit_predict(X_scaled)

n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise = list(labels_db).count(-1)

print(f"\n{'='*60}")
print(f"DBSCAN CLUSTERING ({n_clusters_db} clusters, {n_noise} noise points) - Speed-Based")
print(f"{'='*60}")
for gid, lab in zip(ids, labels_db):
    status = "Noise" if lab == -1 else f"Cluster {lab}"
    print(f"Goblin {gid:2d} -> {status}")

# ======================================================
# PCA VISUALIZATION
# ======================================================
pca = PCA(n_components=2)
P = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(P[:,0], P[:,1], c=labels_kmeans, cmap="tab10", s=200, alpha=0.7, edgecolors='black')

for i, gid in enumerate(ids):
    plt.text(P[i,0]+0.05, P[i,1]+0.05, f"ID {gid}",
             fontsize=12, weight='bold')

plt.title(f"Speed-Based Clustering (K-Means, k={k})", fontsize=14, fontweight='bold')
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.show()

print(f"\n{'='*60}")
print("Speed-based clustering visualization complete!")
print(f"{'='*60}")
