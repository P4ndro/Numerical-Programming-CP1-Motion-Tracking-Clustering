import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

CSV_PATH = "goblin_gang_builtins.csv"   

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(CSV_PATH)

# group per goblin id
groups = {gid: g for gid, g in df.groupby("id")}
print("Found goblins:", list(groups.keys()))

# ======================================================
# FEATURE EXTRACTION
# ======================================================
def extract_features(g):
    speed = g["speed"].values
    accel = g["accel"].values
    jerk  = g["jerk"].values
    s     = g["jounce"].values
    
    features = [
        np.mean(speed),  np.std(speed),
        np.mean(accel),  np.std(accel),
        np.mean(jerk),   np.std(jerk),
        np.mean(s),      np.std(s)
    ]
    return features

X = []
ids = []

for gid, g in groups.items():
    X.append(extract_features(g))
    ids.append(gid)

X = np.array(X)
print("Feature matrix shape:", X.shape)

# ======================================================
# FEATURE SCALING
# ======================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================================================
# K-MEANS CLUSTERING
# (Choose k = how many “styles” of goblins you want)
# ======================================================
k = 2   # try: fast goblins vs slow goblins
kmeans = KMeans(n_clusters=k, random_state=0)
labels_kmeans = kmeans.fit_predict(X_scaled)

print("\n=== KMEANS CLUSTERS ===")
for gid, lab in zip(ids, labels_kmeans):
    print(f"Goblin {gid} -> Cluster {lab}")

# ======================================================
# DBSCAN (density-based)
# ======================================================
dbscan = DBSCAN(eps=1.4, min_samples=1)
labels_db = dbscan.fit_predict(X_scaled)

print("\n=== DBSCAN CLUSTERS ===")
for gid, lab in zip(ids, labels_db):
    print(f"Goblin {gid} -> Cluster {lab}")

# ======================================================
# PCA VISUALIZATION
# ======================================================
pca = PCA(n_components=2)
P = pca.fit_transform(X_scaled)

plt.figure(figsize=(7,6))
plt.scatter(P[:,0], P[:,1], c=labels_kmeans, cmap="tab10", s=200)

for i, gid in enumerate(ids):
    plt.text(P[i,0]+0.02, P[i,1]+0.02, f"ID {gid}",
             fontsize=14, weight='bold')

plt.title("Goblin Gang Clustering (K-Means)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()
