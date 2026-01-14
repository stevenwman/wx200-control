# OpenArm Control: Developer's Guide & Lessons Learned

**Status:** Working / Stable (as of Dec 2025)

This guide summarizes key architectural decisions, performance optimizations, and "gotchas" discovered during development. **Read this before debugging low-level issues.**

---

## 🚀 Critical Performance Tweaks

### 1. USB Latency Timer (The "16ms Bug")
**Symptom:** Control loop blocking, encoder reads taking >15ms, choppy motion.
**Cause:** Default Linux FTDI/USB-Serial driver buffers data for 16ms.
**Fix:** Set latency timer to **1ms**.
```bash
# Check current status
python helper/check_usb_latency.py

# Fix immediately (resets on reboot)
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```
*See `USB_LATENCY_TIMER_OPTIMIZATION.md` for permanent udev rules.*

### 2. GStreamer vs. OpenCV
**Decision:** Use **GStreamer** backend.
**Reason:** Native OpenCV backend often caps at 15fps or blocks. GStreamer with `appsink` allows non-blocking reads at full 30fps.
**Verification:**
```bash
python helper/check_gstreamer_pipeline.py --test
```
*See `helper/GSTREAMER_GUIDE.md` for installation and pipeline debugging.*

---

## 🏗 Architecture Decisions

### Threading vs. Multiprocessing
**Decision:** **Threading** (standard Python `threading` library).
**Why:**
*   **I/O Bound:** Robot communication and Camera reads release the GIL.
*   **Shared State:** We need low-latency access to the singleton `RobotDriver` and `Camera` objects from both the inner loop (120Hz control) and outer loop (20Hz policy).
*   **Overhead:** Multiprocessing introduced unacceptable IPC overhead (>1ms/msg) for our high-frequency needs.
*See `THREADING_VS_MULTIPROCESSING.md`.*

### Control Loops
*   **Inner Loop (~120Hz):** Handles IK solving (`mink`/`quadprog`) and Motor write/read.
*   **Outer Loop (~20Hz):** Handles Camera observation, policy inference, and logging.
*   **Safety:** Inner loop has a "skip threshold" - if we are too close to the outer loop deadline, we skip an inner step to prevent blocking the policy.

---

## 🛠 Hardware Configuration

**Robot:** Trossen WX200 (6 DOF + Gripper)
**Motors:** Dynamixel X-Series
*   **IDs:** 1 through 7 (Base to Gripper).
*   **Baudrate:** 4 Mbps (Required for 100Hz+ control).
*   **Protocol:** 2.0.

**Camera:**
*   **Device:** Typically `/dev/video2` (varies, check `v4l2-ctl --list-devices`).
*   **Format:** 1920x1080 @ 30FPS.
*   **Processing:** Downscaled (e.g., 4x) for ArUco detection to save CPU time.

---

## 📂 Key Scripts & Workflows

| Task | Script | Notes |
|------|--------|-------|
| **Teleop & Record** | `wx200_robot_collect_demo_encoders.py` | Main entry point for data collection. |
| **Replay** | `wx200_real_robot_replay_trajectory.py` | Verify recorded `.npz` files. |
| **Env Test** | `wx200_gym_test.py` | Test the Gym environment wrapper. |
| **Latency Check** | `helper/check_usb_latency.py` | **Run this first if robot feels laggy.** |
| **Cam Check** | `helper/check_gstreamer_pipeline.py` | Verify camera is accessible and fast. |

---

## ❓ Troubleshooting Cheat Sheet

**"Permission Denied" on `/dev/ttyUSB0`**
*   Add user to dialout group: `sudo usermod -a -G dialout $USER`.
*   Relogin.

**"GStreamer pipeline failed"**
*   Check if device is busy: `lsof /dev/video2`.
*   Check if path changed: `ls /dev/video*`. Update `robot_config.py`.

**"Robot Jitter / Stuttering"**
*   **CHECK USB LATENCY.** It's almost always the 16ms timer resetting after a reboot.
*   Check if you are printing too much to console (blocking I/O).

**"ImportError: No module named 'robot_control'"**
*   Ensure you are running scripts from the root directory.
*   `export PYTHONPATH=$PYTHONPATH:.`

---

## 📦 Directory Structure Dependencies

*   `robot_control/`: The "brain". Strictly coupled.
*   `wx200/`: MuJoCo assets. **Must be a sibling** of `robot_control/` due to relative paths in `robot_control_base.py`.
*   `camera/`: Vision logic. Separated for modularity.

*See `BC_DEPENDENCIES.md` for a full dependency map.*
