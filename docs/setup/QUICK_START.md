# Quick Start Guide

## Running Data Collection

### First Time Setup
```bash
# Fix USB latency automatically (requires sudo password)
python compact_code/wx200_robot_collect_demo_encoders_compact.py --auto-fix
```

This will:
1. ✅ Check GStreamer (automatic fix already applied)
2. ✅ Fix USB latency timer (1ms for fast encoder reads)
3. ✅ Check camera device
4. 🚀 Start robot control

### Normal Usage
```bash
python compact_code/wx200_robot_collect_demo_encoders_compact.py
```

The script automatically checks your system and warns if anything needs attention.

## What Got Fixed

### 1. GStreamer Camera Pipeline ✅ Permanent
- **Problem:** Conda had incomplete GStreamer plugins (missing `v4l2src`)
- **Solution:** Created `fix_gstreamer_env.py` module that configures system GStreamer
- **Status:** Integrated into all camera scripts - works automatically
- **Persists:** ✅ Yes (survives reboot)

### 2. USB Latency Timer ⚠️ Temporary (by default)
- **Problem:** Default 16ms latency causes slow encoder reads (16-20ms)
- **Solution:** Set to 1ms for fast reads (10-12ms)
- **Status:** Automatic checks integrated, optional auto-fix available
- **Persists:** ❌ No (resets on reboot unless you set up udev rule)

### 3. Camera Device ID ✅ Permanent
- **Problem:** Was using `/dev/video2` (metadata device)
- **Solution:** Changed to `/dev/video1` (capture device)
- **Status:** Fixed in `robot_config.py`
- **Persists:** ✅ Yes

## Expected Performance

### With Optimal Settings (1ms USB latency)
```
Control loop: ~3-5ms ✓
Encoder reads: ~10-12ms ✓
ArUco polling: ~8ms @ 30Hz ✓
No warnings ✓
```

### With Default Settings (16ms USB latency)
```
Control loop: ~3-5ms ✓
Encoder reads: ~16-20ms ⚠️
ArUco polling: ~8ms @ 30Hz ✓
Encoder warnings every ~5 seconds ⚠️
```

## Making USB Latency Permanent

If you don't want to run `--auto-fix` every time or see warnings, set up a udev rule:

```bash
# 1. Find your USB device IDs
lsusb
udevadm info --name=/dev/ttyUSB0 | grep -E "ID_VENDOR_ID|ID_MODEL_ID"

# 2. Create udev rule
sudo nano /etc/udev/rules.d/99-usb-serial-latency.rules

# 3. Add this line (replace XXXX and YYYY with your vendor/product IDs):
SUBSYSTEM=="tty", ATTRS{idVendor}=="XXXX", ATTRS{idProduct}=="YYYY", RUN+="/bin/sh -c 'echo 1 > /sys/bus/usb-serial/devices/%k/latency_timer'"

# 4. Reload udev
sudo udevadm control --reload-rules
sudo udevadm trigger

# 5. Verify (replug USB or reboot)
cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
# Should show: 1
```

## Command Options

```bash
# Normal run (recommended)
python compact_code/wx200_robot_collect_demo_encoders_compact.py

# Auto-fix USB latency on startup
python compact_code/wx200_robot_collect_demo_encoders_compact.py --auto-fix

# Skip system checks (faster startup, not recommended)
python compact_code/wx200_robot_collect_demo_encoders_compact.py --skip-checks

# Verbose profiling output
python compact_code/wx200_robot_collect_demo_encoders_compact.py --verbose

# Disable video visualization
python compact_code/wx200_robot_collect_demo_encoders_compact.py --no-vis

# Custom output file
python compact_code/wx200_robot_collect_demo_encoders_compact.py --output demo.npz

# Combine options
python compact_code/wx200_robot_collect_demo_encoders_compact.py --auto-fix --no-vis --output demo.npz
```

## Troubleshooting

### Check System Status
```bash
python system_checks.py
```

### Manually Fix USB Latency
```bash
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

### Check USB Latency
```bash
python helper/check_usb_latency.py
```

### List Camera Devices
```bash
v4l2-ctl --list-devices
```

## Files Reference

- **system_checks.py** - Automatic system checks (runs before robot control)
- **fix_gstreamer_env.py** - GStreamer environment configuration (imported automatically)
- **SYSTEM_CHECKS_GUIDE.md** - Detailed guide on system checks
- **ENCODER_WARNING_ANALYSIS.md** - Deep dive on encoder warnings
- **USB_LATENCY_TIMER_OPTIMIZATION.md** - USB latency optimization details
- **GSTREAMER_FIX_SUMMARY.md** - GStreamer fix documentation

## Summary

🎯 **What You Need to Know:**
1. GStreamer fix is automatic and permanent ✅
2. USB latency needs to be 1ms for optimal performance ⚠️
3. Use `--auto-fix` to set USB latency automatically
4. Set up udev rule for permanent USB latency fix (one-time setup)

🚀 **Recommended First Run:**
```bash
python compact_code/wx200_robot_collect_demo_encoders_compact.py --auto-fix
```

After that, the system will check and warn you if anything needs attention.
