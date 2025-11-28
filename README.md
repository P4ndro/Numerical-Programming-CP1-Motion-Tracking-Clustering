# Numerical Programming CP1 – Motion Tracking & Clustering

This repository packages the completed KIU Computational Project (CP1) submission so it can be reused as a template for future numerical programming work.

## Project Scope
- Track moving objects from video, extract position, speed, acceleration, jerk, and jounce (pixel units).
- Cover **two scenarios**: a single-object video (`pekka*.mp4`) and a multi-object video (`goblingang.mp4`).
- Provide **two implementations**: one written “from scratch” (`scratch1.py`) and one leveraging OpenCV built-ins (`builtin1.py` / `builtin2.py`).
- Cluster multi-object motion profiles with derivative-aware features (`clustering.py`) and visualize the styles with PCA.
- Analyze one video where the pipeline succeeds and another (`badvideo.mp4`) where it fails, documenting the limitations.

Deliverables also include `reportcp1.pdf` (2–5 page write-up) and exported CSVs for reproducibility.

## Repository Layout
```
cp1/
├── builtin1.py                # Single-object tracker using morph ops + contours
├── builtin2.py                # Multi-object tracker using background subtraction
├── scratch1.py                # Minimal single-object tracker (manual thresholding)
├── clustering.py              # Feature scaling + KMeans & DBSCAN + PCA plot
├── *.mp4                      # Example input videos (success & failure cases)
├── *.csv                      # Tracking outputs ready for clustering/reporting
├── reportcp1.pdf              # Project report
├── requirements.txt
└── README.md
```

## Environment Setup
```bash
python -m venv .venv
. .venv/Scripts/activate  # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Reproducing the Pipelines
1. **Single-object (scratch)**  
   ```
   python scratch1.py
   ```
   - Adjust `VIDEO_PATH`, HSV bounds, and `MIN_PIXELS` for new footage.
   - Outputs `pekka_tracking_scratch.csv`.

2. **Single-object (built-ins)**  
   ```
   python builtin1.py
   ```
   - Uses morphological opening and contour filtering for more robust centroids.
   - Writes `pekka_builtins.csv`.

3. **Multi-object (built-ins)**  
   ```
   python builtin2.py
   ```
   - Uses `cv2.createBackgroundSubtractorMOG2`. Tunable knobs: `MIN_AREA`, `MAX_GOBLINS`, `MAX_DIST`.
   - Saves `goblin_gang_motion.csv`.

4. **Clustering & Visualization**  
   ```
   python clustering.py
   ```
   - Reads `goblin_gang_builtins.csv`, extracts derivative statistics, standardizes them, runs KMeans (k=2) and DBSCAN, and plots PCA embeddings.

## Notes on Physical Units
All derivatives are in pixel-based units. To convert to meters/seconds, record a calibration factor (e.g., known ground tile size) and multiply positions before differentiation.

## Failure Analysis
`badvideo.mp4` demonstrates adverse lighting/occlusion where color thresholding and the nearest-neighbour ID assignment break down. Highlighting such counterexamples is part of the CP1 requirements and should be expanded when adding new datasets.

## Publishing Workflow
1. Initialize a Git repository (if not already done).
2. Add this GitHub remote (empty repo mentioned in instructions):  
   ```
   git remote add origin https://github.com/P4ndro/Numerical-ComputationalProjects.git
   ```
3. Commit project files and push:
   ```
   git add .
   git commit -m "Add CP1 motion tracking & clustering project"
   git push -u origin main
   ```

> ℹ️ As of the latest check, the GitHub repository `P4ndro/Numerical-ComputationalProjects` is empty, so pushing this project will populate it [source](https://github.com/P4ndro/Numerical-ComputationalProjects).

## Next Steps
- Extend `requirements.txt` or pin versions if needed for class submissions.
- Replace/augment the current videos with new ones before reusing the template, ensuring originality per CP guidelines.
- Update the report and README each time new datasets or algorithms are added.

