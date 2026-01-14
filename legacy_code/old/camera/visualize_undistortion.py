import glob
import os

import cv2
import numpy as np

# --- User parameters ---
# Folder of original images to visualize
IMAGE_FOLDER = "calibration_images"
IMAGE_PATTERN = os.path.join(IMAGE_FOLDER, "*.jpg")

# Calibration file produced by compute_camera_calibration.py
CALIB_FILE = "camera_calibration_results.npz"

# Output folder for visualizations
OUTPUT_FOLDER = "undistortion_visualization"

# Maximum number of images to visualize (set to None for all)
MAX_IMAGES = 12


def load_calibration(path: str):
    data = np.load(path, allow_pickle=True)
    K = data["camera_matrix"]
    dist = data["dist_coeffs"]
    return K, dist


def make_side_by_side(distorted: np.ndarray, undistorted: np.ndarray) -> np.ndarray:
    """Create a simple side‑by‑side comparison image."""
    # Resize undistorted to match distorted size if needed
    h, w = distorted.shape[:2]
    undist_resized = cv2.resize(undistorted, (w, h))

    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w] = distorted
    canvas[:, w:] = undist_resized

    # Add labels
    cv2.putText(
        canvas,
        "Original",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Undistorted",
        (w + 10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return canvas


def make_difference_visual(distorted: np.ndarray, undistorted: np.ndarray) -> np.ndarray:
    """
    Create a visualization of per‑pixel differences:
    - Left: original
    - Middle: undistorted
    - Right: heatmap of |original - undistorted| magnitude
    """
    h, w = distorted.shape[:2]
    undist_resized = cv2.resize(undistorted, (w, h))

    # Compute per‑pixel delta magnitude in grayscale
    gray_orig = cv2.cvtColor(distorted, cv2.COLOR_BGR2GRAY)
    gray_undist = cv2.cvtColor(undist_resized, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_orig, gray_undist)

    # Amplify differences for visibility and apply colormap
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    diff_color = cv2.applyColorMap(diff_norm.astype(np.uint8), cv2.COLORMAP_JET)

    canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
    canvas[:, :w] = distorted
    canvas[:, w : 2 * w] = undist_resized
    canvas[:, 2 * w :] = diff_color

    # Labels
    cv2.putText(
        canvas,
        "Original",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Undistorted",
        (w + 10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Delta magnitude",
        (2 * w + 10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return canvas


def main() -> None:
    if not os.path.exists(CALIB_FILE):
        print(f"Calibration file '{CALIB_FILE}' not found. Run compute_camera_calibration.py first.")
        return

    K, dist = load_calibration(CALIB_FILE)
    print("Loaded calibration:")
    print("K =\n", K)
    print("dist =", dist.ravel())

    image_paths = sorted(glob.glob(IMAGE_PATTERN))
    if not image_paths:
        print(f"No images found matching {IMAGE_PATTERN}")
        return

    if MAX_IMAGES is not None:
        image_paths = image_paths[:MAX_IMAGES]

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Could not read image: {path}")
            continue

        h, w = img.shape[:2]

        # Compute optimal new camera matrix and undistort
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
        undistorted = cv2.undistort(img, K, dist, None, new_K)

        # Simple side‑by‑side
        comparison = make_side_by_side(img, undistorted)
        out_name_side = os.path.join(OUTPUT_FOLDER, os.path.basename(path))
        cv2.imwrite(out_name_side, comparison)

        # Three‑panel with delta magnitude heatmap
        diff_vis = make_difference_visual(img, undistorted)
        root, ext = os.path.splitext(os.path.basename(path))
        out_name_diff = os.path.join(OUTPUT_FOLDER, f"{root}_diff{ext}")
        cv2.imwrite(out_name_diff, diff_vis)

        print(f"[OK] Wrote {out_name_side} and {out_name_diff}")

    print(
        f"\nSaved visualizations to '{OUTPUT_FOLDER}'. The '*_diff' images show per‑pixel delta magnitude as a heatmap."
    )


if __name__ == "__main__":
    main()
