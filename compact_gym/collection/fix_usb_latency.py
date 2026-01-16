#!/usr/bin/env python3
"""
USB Latency Auto-Fix Script

Automatically sets USB latency timer to 1ms for better robot control performance.
Can be run standalone or imported and called from other scripts.
"""
import subprocess
import sys
import os


def fix_usb_latency(device='/dev/ttyUSB0', target_latency=1, verbose=True):
    """
    Fix USB latency timer for specified device.

    Args:
        device: USB device path (default: /dev/ttyUSB0)
        target_latency: Target latency in milliseconds (default: 1)
        verbose: Print status messages

    Returns:
        bool: True if fix was applied successfully, False otherwise
    """
    if not os.path.exists(device):
        if verbose:
            print(f"⚠️  Device {device} not found")
        return False

    try:
        # Get current latency
        result = subprocess.run(
            ['cat', f'/sys/bus/usb-serial/devices/{os.path.basename(device)}/latency_timer'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            if verbose:
                print(f"⚠️  Could not read latency timer for {device}")
            return False

        current_latency = int(result.stdout.strip())

        if current_latency == target_latency:
            if verbose:
                print(f"✓ USB latency already optimal ({current_latency}ms)")
            return True

        # Need to fix - requires sudo
        if verbose:
            print(f"⚠️  USB latency is {current_latency}ms (target: {target_latency}ms)")
            print(f"   Attempting to fix with sudo...")

        # Try to fix with sudo
        result = subprocess.run(
            ['sudo', 'sh', '-c',
             f'echo {target_latency} > /sys/bus/usb-serial/devices/{os.path.basename(device)}/latency_timer'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            if verbose:
                print(f"✗ Failed to set latency: {result.stderr}")
            return False

        # Verify
        result = subprocess.run(
            ['cat', f'/sys/bus/usb-serial/devices/{os.path.basename(device)}/latency_timer'],
            capture_output=True,
            text=True
        )

        new_latency = int(result.stdout.strip())
        if new_latency == target_latency:
            if verbose:
                print(f"✓ USB latency fixed: {current_latency}ms → {new_latency}ms")
            return True
        else:
            if verbose:
                print(f"⚠️  Latency changed but not to target: {new_latency}ms (wanted {target_latency}ms)")
            return False

    except Exception as e:
        if verbose:
            print(f"✗ Error fixing USB latency: {e}")
        return False


def main():
    """Command-line interface for USB latency fix."""
    import argparse

    parser = argparse.ArgumentParser(description='Fix USB latency timer for robot control')
    parser.add_argument('--device', default='/dev/ttyUSB0', help='USB device path (default: /dev/ttyUSB0)')
    parser.add_argument('--latency', type=int, default=1, help='Target latency in ms (default: 1)')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    success = fix_usb_latency(
        device=args.device,
        target_latency=args.latency,
        verbose=not args.quiet
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
