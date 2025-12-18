# USB Latency Timer Optimization for Dynamixel Motors

## The Problem

When controlling Dynamixel motors at high frequencies (100+ Hz control loops), encoder reads were taking **~18-20ms**, causing:

- **Control loop blocking**: Inner loop (100Hz) blocked for ~15ms waiting for encoder reads
- **Choppy performance**: ~30-40% of inner loop iterations blocked
- **Bottleneck**: Serial communication artificially limited by OS USB driver

### Observed Performance Before Fix

```
txRxPacket() Timing (actual Dynamixel communication):
  avg=18.43ms, min=9.42ms, max=25.64ms
  p95=24.94ms, p99=25.54ms
  Near timeout (18-22ms): 8.7%

Encoder read time: avg=20.22ms
Inner loop wait time: avg=15.2ms (blocking!)
```

## Root Cause: USB Latency Timer

The **USB-to-Serial driver** (FTDI/CP210x) has a configurable **latency timer** that buffers data before sending it to the CPU. 

**Default value: 16ms** - This means the driver waits up to 16ms to batch data, creating an artificial bottleneck.

**For high-frequency control:**
- 16ms latency = maximum ~60 Hz communication rate
- Even with 4 Mbps baudrate, you're capped by this OS-level setting
- Motor responses are delayed by this buffering

## The Solution

Set the USB latency timer to **1ms**:

```bash
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

**Replace `ttyUSB0` with your actual device name** (check with `ls /dev/ttyUSB*`)

### Verify It Worked

Check the current value:
```bash
cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

Should output: `1`

Or use our helper script:
```bash
python helper/check_usb_latency.py
```

Should show: `✅ OPTIMAL: Latency timer is set to 1ms`

## Performance Improvement

### After Setting to 1ms

```
txRxPacket() Timing (actual Dynamixel communication):
  avg=10.28ms, min=8.95ms, max=10.81ms
  p95=10.73ms, p99=10.78ms
  Near timeout (18-22ms): 0.0%

Encoder read time: avg=12.43ms (down from 20ms!)
Inner loop wait time: avg=6.3ms (down from 15ms!)
```

### Improvements

| Metric | Before (16ms) | After (1ms) | Improvement |
|--------|---------------|-------------|-------------|
| **txRxPacket avg** | 18.43ms | 10.28ms | **44% faster** |
| **Encoder read avg** | 20.22ms | 12.43ms | **38% faster** |
| **Inner loop blocking** | 15.2ms | 6.3ms | **59% reduction** |
| **Control loop iteration** | ~26ms | ~18ms | **31% faster** |
| **Timeout-bound reads** | 8.7% | 0.0% | **Eliminated** |

## Making It Permanent

The setting resets after reboot. To make it permanent:

### Option 1: udev Rule (Recommended)

Create a udev rule that automatically sets the latency timer when the device is plugged in:

```bash
sudo nano /etc/udev/rules.d/99-usb-serial-latency.rules
```

Add this line (replace vendor/product IDs if needed):
```
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", ATTRS{serial}=="*", SYMLINK+="ttyUSB_DXL", RUN+="/bin/sh -c 'echo 1 > /sys/bus/usb-serial/devices/%k/latency_timer'"
```

**To find your device's vendor/product IDs:**
```bash
lsusb
# Look for your USB serial adapter, then:
udevadm info --name=/dev/ttyUSB0 | grep ID_VENDOR_ID
udevadm info --name=/dev/ttyUSB0 | grep ID_MODEL_ID
```

Then reload udev rules:
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Option 2: Systemd Service

Create a service that runs on boot:
```bash
sudo nano /etc/systemd/system/usb-latency-timer.service
```

Add:
```ini
[Unit]
Description=Set USB Serial Latency Timer to 1ms
After=sys-devices-virtual-tty-ttyUSB0.device

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'echo 1 > /sys/bus/usb-serial/devices/ttyUSB0/latency_timer'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable usb-latency-timer.service
sudo systemctl start usb-latency-timer.service
```

### Option 3: Startup Script

Add to your robot startup script:
```bash
# Set USB latency timer for Dynamixel communication
if [ -f /sys/bus/usb-serial/devices/ttyUSB0/latency_timer ]; then
    echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer > /dev/null
fi
```

## Why This Matters

For **100Hz control loops** with **20Hz encoder polling**:

- **Before**: Encoder reads (20ms) blocked inner loop for 15ms = **150% of control period**
- **After**: Encoder reads (12ms) block inner loop for 6ms = **60% of control period**

This means:
- ✅ Inner loop can maintain 100Hz more reliably
- ✅ Fewer missed control deadlines
- ✅ Smoother robot motion
- ✅ Less jitter in control commands

## Technical Details

### How the Latency Timer Works

The USB-to-Serial driver buffers data to reduce CPU overhead. The latency timer controls how long it waits before sending buffered data:

- **High latency (16ms)**: Less CPU usage, but higher communication delay
- **Low latency (1ms)**: More CPU usage, but much faster response

For robotics applications, **fast response is critical**, so 1ms is the right choice.

### Trade-offs

**Pros:**
- Much faster communication
- Better real-time performance
- No code changes needed

**Cons:**
- Slightly higher CPU usage (negligible for modern systems)
- Must be set on each boot (unless automated)

## Troubleshooting

### "Permission denied" error
You need sudo/root access to change the latency timer:
```bash
sudo echo 1 | tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

### Device path changes
If your device isn't always `/dev/ttyUSB0`, use:
```bash
# Find your device
ls /dev/ttyUSB*

# Set for all USB serial devices (be careful!)
for dev in /sys/bus/usb-serial/devices/*/latency_timer; do
    echo 1 | sudo tee "$dev" > /dev/null
done
```

### Verify improvement
After setting, run your profiler and compare `txRxPacket()` timing:
```bash
python wx200_robot_profile_camera.py --record --record-frames --output test_1ms.npz --no-vis
```

Look for:
- ✅ `txRxPacket()` avg should be ~5-12ms (not 18-20ms)
- ✅ `Near timeout (18-22ms): 0.0%` (not 8-10%)
- ✅ `inner_waits` should be much lower

## References

- [DynamixelSDK GitHub Issue #325](https://github.com/ROBOTIS-GIT/DynamixelSDK/issues/325)
- [FTDI Application Note: AN_107](https://www.ftdichip.com/Support/Documents/AppNotes/AN_107_AdvancedDriverOptions_AN_000107.pdf)
- Check script: `helper/check_usb_latency.py`

## Summary

**Bottom line**: For high-frequency robot control with Dynamixel motors, always set the USB latency timer to **1ms**. This is the single most impactful optimization you can make without changing any code.

**Before you start any robot control session:**
```bash
python helper/check_usb_latency.py
```

If it shows 16ms or higher, fix it:
```bash
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```
