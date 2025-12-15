import cv2
import os

# --- Configuration ---
SAVE_FOLDER = 'calibration_images'
CAMERA_INDEX = 1  # 0 is usually the default webcam. Change to 1 or 2 if you have multiple cameras.

# Create the folder if it doesn't exist
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)
    print(f"Created folder: {SAVE_FOLDER}")

# Initialize the camera
cap = cv2.VideoCapture(CAMERA_INDEX)

# Optional: Set resolution (if your camera supports higher res, it's good to force it)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

img_counter = 0

print("\n--- Controls ---")
print("SPACE BAR: Save current frame")
print("ESC:       Quit")
print("----------------")

while True:
    # Read a new frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Show the frame in a window
    cv2.imshow("Calibration Image Collector", frame)

    # Wait for a key press
    k = cv2.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = os.path.join(SAVE_FOLDER, f"calib_{img_counter}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"{img_name} saved!")
        img_counter += 1

# Cleanup
cap.release()
cv2.destroyAllWindows()