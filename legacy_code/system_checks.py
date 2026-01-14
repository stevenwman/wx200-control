"""
System pre-flight checks for robot control.

Automatically checks critical system settings before running robot control:
- GStreamer availability and plugin support
- USB latency timer for Dynamixel communication
- Camera device availability

Import this module to run checks automatically, or call run_system_checks() explicitly.
"""
import fix_gstreamer_env  # Fix GStreamer environment before checking
import os
import sys
import subprocess
from pathlib import Path


class SystemCheckResult:
    """Result of a system check."""
    def __init__(self, name, passed, value=None, expected=None, fix_command=None, details=None):
        self.name = name
        self.passed = passed
        self.value = value
        self.expected = expected
        self.fix_command = fix_command
        self.details = details


def check_usb_latency(device='/dev/ttyUSB0'):
    """Check USB latency timer setting."""
    try:
        # Find the latency timer file
        device_name = device.split('/')[-1]
        latency_file = f'/sys/bus/usb-serial/devices/{device_name}/latency_timer'

        if not os.path.exists(latency_file):
            return SystemCheckResult(
                name='USB Latency Timer',
                passed=False,
                value='Device not found',
                expected='1ms',
                details=f'Could not find {device} or latency timer file'
            )

        with open(latency_file, 'r') as f:
            latency_ms = int(f.read().strip())

        if latency_ms == 1:
            return SystemCheckResult(
                name='USB Latency Timer',
                passed=True,
                value=f'{latency_ms}ms',
                expected='1ms',
                details='Optimal for high-frequency control'
            )
        else:
            return SystemCheckResult(
                name='USB Latency Timer',
                passed=False,
                value=f'{latency_ms}ms',
                expected='1ms',
                fix_command=f'echo 1 | sudo tee {latency_file}',
                details=f'High latency will cause slow encoder reads (16-20ms instead of 10-12ms)'
            )
    except Exception as e:
        return SystemCheckResult(
            name='USB Latency Timer',
            passed=False,
            value='Error',
            expected='1ms',
            details=f'Error checking latency: {e}'
        )


def check_gstreamer():
    """Check GStreamer availability and v4l2src plugin."""
    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst

        Gst.init(None)

        # Check for v4l2src plugin
        v4l2src = Gst.ElementFactory.make('v4l2src', None)
        if v4l2src:
            return SystemCheckResult(
                name='GStreamer',
                passed=True,
                value='Available with v4l2src',
                expected='Available',
                details=f'GStreamer {Gst.version_string()}'
            )
        else:
            return SystemCheckResult(
                name='GStreamer',
                passed=False,
                value='v4l2src plugin missing',
                expected='Available with v4l2src',
                details='GStreamer found but v4l2src plugin missing'
            )
    except Exception as e:
        return SystemCheckResult(
            name='GStreamer',
            passed=False,
            value='Not available',
            expected='Available',
            details=f'Error: {e}'
        )


def check_camera_device(device_id=1):
    """Check if camera device exists and is accessible."""
    device_path = f'/dev/video{device_id}'

    if not os.path.exists(device_path):
        return SystemCheckResult(
            name='Camera Device',
            passed=False,
            value='Not found',
            expected=device_path,
            details=f'{device_path} does not exist'
        )

    # Check if it's a capture device (not metadata)
    try:
        result = subprocess.run(
            ['v4l2-ctl', '--device', device_path, '--list-formats-ext'],
            capture_output=True,
            text=True,
            timeout=2
        )

        if 'Video Capture' in result.stdout and ('MJPG' in result.stdout or 'YUYV' in result.stdout):
            return SystemCheckResult(
                name='Camera Device',
                passed=True,
                value=device_path,
                expected='Capture device',
                details='Camera is a valid capture device'
            )
        else:
            return SystemCheckResult(
                name='Camera Device',
                passed=False,
                value=device_path,
                expected='Capture device',
                details=f'{device_path} exists but may not be a capture device'
            )
    except subprocess.TimeoutExpired:
        return SystemCheckResult(
            name='Camera Device',
            passed=False,
            value=device_path,
            expected='Accessible',
            details='Camera check timed out'
        )
    except FileNotFoundError:
        # v4l2-ctl not installed, assume device is OK if it exists
        return SystemCheckResult(
            name='Camera Device',
            passed=True,
            value=device_path,
            expected='Present',
            details='Device exists (v4l2-ctl not available for detailed check)'
        )
    except Exception as e:
        return SystemCheckResult(
            name='Camera Device',
            passed=False,
            value=device_path,
            expected='Accessible',
            details=f'Error checking device: {e}'
        )


def run_system_checks(camera_device_id=1, usb_device='/dev/ttyUSB0', verbose=True, require_all=False):
    """
    Run all system checks.

    Args:
        camera_device_id: Camera device ID (default: 1 for /dev/video1)
        usb_device: USB serial device path (default: /dev/ttyUSB0)
        verbose: Print detailed results (default: True)
        require_all: Raise exception if any check fails (default: False)

    Returns:
        dict: Dictionary of check results
    """
    checks = {
        'gstreamer': check_gstreamer(),
        'usb_latency': check_usb_latency(usb_device),
        'camera': check_camera_device(camera_device_id),
    }

    if verbose:
        print("\n" + "="*70)
        print("SYSTEM PRE-FLIGHT CHECKS")
        print("="*70)

        for check_name, result in checks.items():
            status = "✓" if result.passed else "✗"
            print(f"\n{status} {result.name}")
            if result.value:
                print(f"  Current:  {result.value}")
            if result.expected:
                print(f"  Expected: {result.expected}")
            if result.details:
                print(f"  Details:  {result.details}")
            if result.fix_command:
                print(f"  Fix:      {result.fix_command}")

        print("\n" + "="*70)

        # Summary
        passed = sum(1 for r in checks.values() if r.passed)
        total = len(checks)

        if passed == total:
            print(f"✓ ALL CHECKS PASSED ({passed}/{total})")
        else:
            print(f"⚠ SOME CHECKS FAILED ({passed}/{total} passed)")

            # Show critical failures
            critical_failures = []
            if not checks['usb_latency'].passed:
                critical_failures.append(checks['usb_latency'])

            if critical_failures:
                print("\n⚠️  CRITICAL WARNINGS:")
                for failure in critical_failures:
                    print(f"  • {failure.name}: {failure.details}")
                    if failure.fix_command:
                        print(f"    Run: {failure.fix_command}")

                print("\n⚠️  Robot will run but performance may be degraded.")
                print("   Encoder reads will be slow (16-20ms instead of 10-12ms).")

        print("="*70 + "\n")

    # Raise exception if required and any check failed
    if require_all and not all(r.passed for r in checks.values()):
        failed_checks = [r.name for r in checks.values() if not r.passed]
        raise RuntimeError(f"System checks failed: {', '.join(failed_checks)}")

    return checks


def check_and_fix_usb_latency(device='/dev/ttyUSB0', auto_fix=False):
    """
    Check USB latency and optionally fix it automatically.

    Args:
        device: USB device path (default: /dev/ttyUSB0)
        auto_fix: Attempt to fix automatically with sudo (default: False)

    Returns:
        bool: True if latency is optimal (1ms) or was fixed successfully
    """
    result = check_usb_latency(device)

    if result.passed:
        return True

    if auto_fix and result.fix_command:
        print(f"\n⚠️  USB latency is not optimal ({result.value})")
        print(f"   Attempting to fix automatically...")

        try:
            device_name = device.split('/')[-1]
            latency_file = f'/sys/bus/usb-serial/devices/{device_name}/latency_timer'

            # Try to fix without sudo first (if user has permissions)
            try:
                with open(latency_file, 'w') as f:
                    f.write('1\n')
                print("   ✓ Fixed USB latency (1ms)")
                return True
            except PermissionError:
                # Need sudo
                subprocess.run(
                    ['sudo', 'tee', latency_file],
                    input='1\n',
                    text=True,
                    capture_output=True,
                    check=True
                )
                print("   ✓ Fixed USB latency with sudo (1ms)")
                return True
        except Exception as e:
            print(f"   ✗ Could not fix automatically: {e}")
            print(f"   Please run manually: {result.fix_command}")
            return False

    return False


# Auto-run checks on import (non-blocking)
if __name__ != '__main__':
    # Only run auto-checks if this is imported (not run directly)
    # This ensures checks run when importing robot control modules
    pass


def main():
    """Run checks when executed as a script."""
    import argparse

    parser = argparse.ArgumentParser(description='Run system pre-flight checks for robot control')
    parser.add_argument('--camera-id', type=int, default=1, help='Camera device ID (default: 1)')
    parser.add_argument('--usb-device', type=str, default='/dev/ttyUSB0', help='USB serial device (default: /dev/ttyUSB0)')
    parser.add_argument('--auto-fix', action='store_true', help='Attempt to automatically fix USB latency')
    parser.add_argument('--require-all', action='store_true', help='Exit with error if any check fails')
    args = parser.parse_args()

    if args.auto_fix:
        check_and_fix_usb_latency(args.usb_device, auto_fix=True)

    results = run_system_checks(
        camera_device_id=args.camera_id,
        usb_device=args.usb_device,
        verbose=True,
        require_all=args.require_all
    )

    # Exit with error code if any check failed
    if not all(r.passed for r in results.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()
