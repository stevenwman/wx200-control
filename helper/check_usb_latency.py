#!/usr/bin/env python3
"""
Check USB latency timer for Dynamixel serial port.

The USB latency timer is critical for fast Dynamixel communication.
Default is 16ms, which caps control frequency at ~60 Hz.
Should be set to 1ms for high-frequency control (100+ Hz).

Usage:
    python helper/check_usb_latency.py [device]
    
Example:
    python helper/check_usb_latency.py /dev/ttyUSB0
"""
import sys
import os
import subprocess
from pathlib import Path

def check_usb_latency(device_path):
    """Check USB latency timer for a serial device."""
    device_path = Path(device_path)
    
    if not device_path.exists():
        print(f"❌ Device {device_path} does not exist")
        return False
    
    # Find the USB serial device in /sys/bus/usb-serial/devices/
    device_name = device_path.name  # e.g., "ttyUSB0"
    latency_timer_path = None
    
    # Check all USB serial devices
    usb_serial_dir = Path("/sys/bus/usb-serial/devices")
    if usb_serial_dir.exists():
        for device_dir in usb_serial_dir.iterdir():
            if device_dir.is_dir():
                # Check if this device matches
                try:
                    with open(device_dir / "latency_timer", 'r') as f:
                        # Get the symlink target to match device name
                        port_path = (device_dir / "port").resolve() if (device_dir / "port").exists() else None
                        if port_path and device_name in str(port_path):
                            latency_timer_path = device_dir / "latency_timer"
                            break
                except (IOError, PermissionError):
                    continue
    
    # Alternative: try direct path
    if latency_timer_path is None:
        # Try to find via udev
        try:
            result = subprocess.run(
                ["udevadm", "info", "--name", str(device_path), "--query", "symlink"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                # Try to find latency_timer from udev info
                result2 = subprocess.run(
                    ["udevadm", "info", "--name", str(device_path), "--attribute-walk"],
                    capture_output=True, text=True, timeout=2
                )
                # This is complex, let's try simpler approach
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    
    # Simplest: try to read from common locations
    possible_paths = [
        Path(f"/sys/bus/usb-serial/devices/{device_name}/latency_timer"),
        Path(f"/sys/class/tty/{device_name}/device/../../../latency_timer"),
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    latency = int(f.read().strip())
                    print(f"✓ Found latency timer at: {path}")
                    print(f"  Current value: {latency} ms")
                    if latency == 1:
                        print(f"  ✅ OPTIMAL: Latency timer is set to 1ms")
                        return True
                    elif latency <= 4:
                        print(f"  ⚠️  ACCEPTABLE: Latency timer is {latency}ms (could be better)")
                        return True
                    else:
                        print(f"  ❌ POOR: Latency timer is {latency}ms (should be 1ms)")
                        print(f"  💡 To fix:")
                        print(f"     echo 1 | sudo tee {path}")
                        print(f"  💡 To make permanent, add a udev rule:")
                        print(f"     See: https://github.com/ROBOTIS-GIT/DynamixelSDK/issues/325")
                        return False
            except (IOError, PermissionError, ValueError) as e:
                continue
    
    # If we can't find it, try via lsusb
    print(f"⚠️  Could not find latency_timer file for {device_path}")
    print(f"  Trying alternative method...")
    
    try:
        # Get USB device info
        result = subprocess.run(
            ["lsusb"],
            capture_output=True, text=True, timeout=2
        )
        print(f"  USB devices found:")
        for line in result.stdout.split('\n'):
            if line.strip():
                print(f"    {line}")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print(f"\n  💡 Manual check:")
    print(f"     cat /sys/bus/usb-serial/devices/*/latency_timer")
    print(f"     # Look for values > 4ms")
    return None


def main():
    if len(sys.argv) > 1:
        device = sys.argv[1]
    else:
        # Default to common Dynamixel port
        # Try to import config, fall back to default if not available
        try:
            from pathlib import Path
            # Add parent directory to path
            script_dir = Path(__file__).parent.parent
            sys.path.insert(0, str(script_dir))
            from robot_control.robot_config import robot_config
            device = robot_config.devicename
        except ImportError:
            # Fall back to default
            device = '/dev/ttyUSB0'
    
    print("="*70)
    print("USB LATENCY TIMER CHECK")
    print("="*70)
    print(f"Checking device: {device}")
    print()
    
    result = check_usb_latency(device)
    
    print()
    print("="*70)
    if result is True:
        print("✅ Latency timer is properly configured")
    elif result is False:
        print("❌ Latency timer needs to be set to 1ms")
        print("   This is CRITICAL for high-frequency control (100+ Hz)")
    else:
        print("⚠️  Could not determine latency timer status")
        print("   Check manually using the commands above")
    print("="*70)


if __name__ == "__main__":
    main()
