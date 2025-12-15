import glob
import os
import shutil

import cv2
import numpy as np

# --- User parameters (EDIT THESE) ---
# Number of **inner** corners per a chessboard row and column
# (i.e. corners, not squares). For a board with 9x6 squares, inner corners are 8x5.
CHECKERBOARD_ROWS = 4   # inner corners along the short side
CHECKERBOARD_COLS = 7   # inner corners along the long side

# Physical size of one square on the checkerboard (in meters)
# e.g. 0.0245 for 24.5 mm
SQUARE_SIZE = 0.01

# Folder where you stored the calibration images
IMAGE_FOLDER = "calibration_images"
IMAGE_PATTERN = os.path.join(IMAGE_FOLDER, "*.jpg")

# Folder to store overlay images with drawn detected grid/corners
DETECTED_OVERLAY_FOLDER = "calibration_images_detected_overlay"

# Output file for calibration results
OUTPUT_FILE = "camera_calibration_results.npz"


def build_object_points(rows: int, cols: int, square_size: float) -> np.ndarray:
    """Create the 3D object points for the checkerboard pattern.

    The pattern lies on the Z=0 plane, with origin at one corner.
    """
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def main() -> None:
    # Prepare object points based on your checkerboard configuration
    objp = build_object_points(CHECKERBOARD_ROWS, CHECKERBOARD_COLS, SQUARE_SIZE)

    # Prepare folder for overlays
    if os.path.exists(DETECTED_OVERLAY_FOLDER):
        shutil.rmtree(DETECTED_OVERLAY_FOLDER)
    os.makedirs(DETECTED_OVERLAY_FOLDER, exist_ok=True)

    # Arrays to store object points and image points from all images
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    # Criteria for cornerSubPix refinement
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1e-6,
    )

    image_paths = sorted(glob.glob(IMAGE_PATTERN))
    if not image_paths:
        print(f"No images found in {IMAGE_FOLDER}. Pattern: {IMAGE_PATTERN}")
        return

    print(f"Found {len(image_paths)} images for calibration.")

    image_size = None

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Could not read image: {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (width, height)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            (CHECKERBOARD_COLS, CHECKERBOARD_ROWS),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK,
        )

        if not ret:
            print(f"[WARN] Chessboard not found in {os.path.basename(path)}")
            continue

        # Refine corner locations
        corners_subpix = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=criteria,
        )

        objpoints.append(objp)
        imgpoints.append(corners_subpix)

        print(f"[OK] Detected corners in {os.path.basename(path)}")

        # Draw detected grid/corners on a copy and save to overlay folder
        annotated = img.copy()

        # Draw the chessboard corners and connecting grid
        cv2.drawChessboardCorners(
            annotated,
            (CHECKERBOARD_COLS, CHECKERBOARD_ROWS),
            corners_subpix,
            True,
        )

        # Additionally draw explicit grid lines between neighboring corners
        # (rows: along COLS, columns: along ROWS)
        for r in range(CHECKERBOARD_ROWS):
            for c in range(CHECKERBOARD_COLS - 1):
                p1 = tuple(corners_subpix[r * CHECKERBOARD_COLS + c][0].astype(int))
                p2 = tuple(corners_subpix[r * CHECKERBOARD_COLS + c + 1][0].astype(int))
                cv2.line(annotated, p1, p2, (0, 255, 0), 2)

        for c in range(CHECKERBOARD_COLS):
            for r in range(CHECKERBOARD_ROWS - 1):
                p1 = tuple(corners_subpix[r * CHECKERBOARD_COLS + c][0].astype(int))
                p2 = tuple(corners_subpix[(r + 1) * CHECKERBOARD_COLS + c][0].astype(int))
                cv2.line(annotated, p1, p2, (0, 255, 0), 2)

        overlay_path = os.path.join(DETECTED_OVERLAY_FOLDER, os.path.basename(path))
        cv2.imwrite(overlay_path, annotated)

    if not objpoints:
        print("No valid images with detected chessboard corners. Aborting calibration.")
        return

    print("\nRunning camera calibration...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )

    print(f"Reprojection RMS error: {ret:.6f}")
    print("Camera matrix (intrinsics):\n", camera_matrix)
    print("Distortion coefficients: ", dist_coeffs.ravel())

    # Compute per-image reprojection error
    total_error = 0
    total_points = 0
    per_image_errors = []

    for i, (objp_i, imgp_i, rvec, tvec) in enumerate(
        zip(objpoints, imgpoints, rvecs, tvecs)
    ):
        imgpoints_proj, _ = cv2.projectPoints(objp_i, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(imgp_i, imgpoints_proj, cv2.NORM_L2) / len(imgpoints_proj)
        per_image_errors.append(error)
        total_error += error**2 * len(imgpoints_proj)
        total_points += len(imgpoints_proj)

    overall_rmse = np.sqrt(total_error / total_points)
    print(f"Overall per-point reprojection RMSE: {overall_rmse:.6f} pixels")

    # Save intrinsics and extrinsics
    np.savez(
        OUTPUT_FILE,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=rvecs,
        tvecs=tvecs,
        image_paths=np.array(image_paths, dtype=object),
        checkerboard_rows=CHECKERBOARD_ROWS,
        checkerboard_cols=CHECKERBOARD_COLS,
        square_size=SQUARE_SIZE,
        per_image_errors=np.array(per_image_errors),
        reprojection_rms=ret,
        overall_rmse=overall_rmse,
    )

    print(f"\nSaved calibration results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
