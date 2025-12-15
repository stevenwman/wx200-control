#!/usr/bin/env python3
"""
Verify that a smoothed trajectory file is identical to the original,
except for the added smoothed_* attributes.

This script ensures that the smoothing process didn't corrupt any original data.
"""

import numpy as np
from pathlib import Path
import sys


def verify_smoothed_trajectory(original_path: Path, smoothed_path: Path):
    """
    Compare original and smoothed trajectory files.
    
    Returns:
        bool: True if files match (except smoothed_* keys), False otherwise
    """
    print(f"Loading original: {original_path}")
    data_orig = np.load(original_path, allow_pickle=True)
    
    print(f"Loading smoothed: {smoothed_path}")
    data_smooth = np.load(smoothed_path, allow_pickle=True)
    
    # Get all keys from both files
    keys_orig = set(data_orig.keys())
    keys_smooth = set(data_smooth.keys())
    
    # Expected smoothed keys
    smoothed_keys = {
        "smoothed_aruco_ee_in_world",
        "smoothed_aruco_object_in_world",
        "smoothed_aruco_ee_in_object",
        "smoothed_aruco_object_in_ee",
    }
    
    # Check that smoothed file has all original keys
    missing_in_smooth = keys_orig - keys_smooth
    if missing_in_smooth:
        print(f"❌ ERROR: Original keys missing in smoothed file: {missing_in_smooth}")
        return False
    
    # Check that smoothed file only adds expected smoothed_* keys
    extra_keys = keys_smooth - keys_orig
    unexpected_keys = extra_keys - smoothed_keys
    if unexpected_keys:
        print(f"⚠️  WARNING: Unexpected extra keys in smoothed file: {unexpected_keys}")
        # Don't fail, but warn
    
    print(f"\n✓ Original file has {len(keys_orig)} keys")
    print(f"✓ Smoothed file has {len(keys_smooth)} keys")
    print(f"✓ Added {len(extra_keys & smoothed_keys)} smoothed_* keys")
    
    # Compare all original keys
    print("\nComparing original keys...")
    all_match = True
    
    for key in sorted(keys_orig):
        arr_orig = data_orig[key]
        arr_smooth = data_smooth[key]
        
        # Handle metadata specially (it's a dict)
        if key == "metadata":
            if isinstance(arr_orig, np.ndarray) and arr_orig.dtype == object:
                meta_orig = arr_orig.item()
                meta_smooth = arr_smooth.item()
                
                # Metadata might have extra fields, so compare original fields
                orig_fields = set(meta_orig.keys())
                smooth_fields = set(meta_smooth.keys())
                
                # Check that all original metadata fields match
                for field in orig_fields:
                    if field not in meta_smooth:
                        print(f"❌ ERROR: Metadata field '{field}' missing in smoothed file")
                        all_match = False
                    elif meta_orig[field] != meta_smooth[field]:
                        print(f"❌ ERROR: Metadata field '{field}' differs")
                        print(f"   Original: {meta_orig[field]}")
                        print(f"   Smoothed: {meta_smooth[field]}")
                        all_match = False
                
                if all_match:
                    print(f"  ✓ {key}: metadata matches")
            else:
                if not np.array_equal(arr_orig, arr_smooth):
                    print(f"❌ ERROR: {key} differs")
                    all_match = False
                else:
                    print(f"  ✓ {key}: matches")
        else:
            # Compare arrays
            if not isinstance(arr_orig, np.ndarray) or not isinstance(arr_smooth, np.ndarray):
                if arr_orig != arr_smooth:
                    print(f"❌ ERROR: {key} differs (non-array)")
                    all_match = False
                else:
                    print(f"  ✓ {key}: matches")
            else:
                # Check shape
                if arr_orig.shape != arr_smooth.shape:
                    print(f"❌ ERROR: {key} shape differs")
                    print(f"   Original: {arr_orig.shape}")
                    print(f"   Smoothed: {arr_smooth.shape}")
                    all_match = False
                # Check dtype
                elif arr_orig.dtype != arr_smooth.dtype:
                    print(f"❌ ERROR: {key} dtype differs")
                    print(f"   Original: {arr_orig.dtype}")
                    print(f"   Smoothed: {arr_smooth.dtype}")
                    all_match = False
                # Check values
                elif not np.array_equal(arr_orig, arr_smooth):
                    # Check if difference is just floating point precision
                    if np.allclose(arr_orig, arr_smooth, rtol=1e-10, atol=1e-10):
                        print(f"  ✓ {key}: matches (within floating point precision)")
                    else:
                        max_diff = np.abs(arr_orig - arr_smooth).max()
                        print(f"❌ ERROR: {key} values differ")
                        print(f"   Max difference: {max_diff}")
                        print(f"   First differing element:")
                        diff_mask = arr_orig != arr_smooth
                        if np.any(diff_mask):
                            idx = np.unravel_index(np.argmax(diff_mask), arr_orig.shape)
                            print(f"     Index {idx}: orig={arr_orig[idx]}, smooth={arr_smooth[idx]}")
                        all_match = False
                else:
                    print(f"  ✓ {key}: matches")
    
    # Verify smoothed keys exist and have correct shape
    print("\nVerifying smoothed keys...")
    for key in sorted(smoothed_keys):
        if key not in data_smooth:
            print(f"⚠️  WARNING: {key} not found in smoothed file")
            continue
        
        # Get corresponding original key
        orig_key = key.replace("smoothed_", "")
        if orig_key not in data_orig:
            print(f"⚠️  WARNING: Original key {orig_key} not found for comparison")
            continue
        
        arr_orig = data_orig[orig_key]
        arr_smooth = data_smooth[key]
        
        if arr_orig.shape != arr_smooth.shape:
            print(f"❌ ERROR: {key} shape doesn't match original")
            print(f"   Original {orig_key}: {arr_orig.shape}")
            print(f"   Smoothed {key}: {arr_smooth.shape}")
            all_match = False
        else:
            print(f"  ✓ {key}: shape matches ({arr_smooth.shape})")
    
    return all_match


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verify that smoothed trajectory matches original (except smoothed_* keys)"
    )
    parser.add_argument(
        "original",
        type=Path,
        help="Path to original trajectory file (.npz)",
    )
    parser.add_argument(
        "smoothed",
        type=Path,
        nargs="?",
        default=None,
        help="Path to smoothed trajectory file (.npz). If not provided, infers from original name.",
    )
    
    args = parser.parse_args()
    
    if not args.original.exists():
        print(f"Error: Original file not found: {args.original}")
        sys.exit(1)
    
    if args.smoothed is None:
        # Infer smoothed filename
        args.smoothed = args.original.parent / f"{args.original.stem}_smoothed.npz"
    
    if not args.smoothed.exists():
        print(f"Error: Smoothed file not found: {args.smoothed}")
        print(f"Expected: {args.smoothed}")
        sys.exit(1)
    
    print("="*60)
    print("VERIFYING SMOOTHED TRAJECTORY")
    print("="*60)
    print()
    
    success = verify_smoothed_trajectory(args.original, args.smoothed)
    
    print()
    print("="*60)
    if success:
        print("✓ VERIFICATION PASSED: Files match (except smoothed_* keys)")
        sys.exit(0)
    else:
        print("❌ VERIFICATION FAILED: Files differ")
        sys.exit(1)


if __name__ == "__main__":
    main()

