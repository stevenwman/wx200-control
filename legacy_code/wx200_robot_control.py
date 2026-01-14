"""
WX200 Real Robot Control - Unified Script

Supports three modes:
1. Teleop: Real-time control with SpaceMouse (default)
2. Record: Teleop with trajectory recording
3. Replay: Replay recorded trajectories

Usage:
    # Teleop mode (no recording)
    python wx200_robot_control.py
    
    # Record mode
    python wx200_robot_control.py --record [--output OUTPUT_FILE]
    
    # Replay mode
    python wx200_robot_control.py --replay TRAJECTORY_FILE.npz [--start-index INDEX] [--end-index INDEX]

Press Ctrl+C to stop and execute shutdown sequence.
"""
import argparse
from pathlib import Path
from datetime import datetime

from wx200_robot_teleop_control import TeleopControl
from wx200_robot_replay_trajectory import ReplayControl, load_trajectory


def main():
    parser = argparse.ArgumentParser(
        description='WX200 Real Robot Control - Unified Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Teleop mode (no recording)
  python wx200_robot_control.py
  
  # Record mode
  python wx200_robot_control.py --record
  python wx200_robot_control.py --record --output my_trajectory.npz
  
  # Replay mode
  python wx200_robot_control.py --replay trajectory_20240101_120000.npz
  python wx200_robot_control.py --replay trajectory.npz --start-index 100 --end-index 500
        """
    )
    
    # Mode selection
    parser.add_argument('--record', action='store_true', help='Enable trajectory recording (teleop mode)')
    parser.add_argument('--replay', type=str, metavar='FILE', help='Replay mode: path to trajectory NPZ file')
    
    # Recording options
    parser.add_argument('--output', type=str, help='Output file path for trajectory (recording mode)')
    
    # Replay options
    parser.add_argument('--start-index', type=int, default=0, help='Start replaying from this index (replay mode)')
    parser.add_argument('--end-index', type=int, default=None, help='Stop replaying at this index (replay mode)')
    
    args = parser.parse_args()
    
    if args.replay:
        # Replay mode
        trajectory_path = Path(args.replay)
        if not trajectory_path.exists():
            print(f"Error: Trajectory file not found: {trajectory_path}")
            return
        
        print("WX200 Real Robot Control - REPLAY MODE")
        print("="*60)
        print(f"Trajectory file: {trajectory_path}")
        print("="*60)
        
        trajectory = load_trajectory(trajectory_path)
        
        try:
            ReplayControl(trajectory, start_idx=args.start_index, end_idx=args.end_index).run()
        except KeyboardInterrupt:
            print("\n\nKeyboard interrupt detected during initialization...")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Teleop mode (with optional recording)
        enable_recording = args.record
        if enable_recording:
            if args.output:
                output_path = Path(args.output)
            else:
                data_dir = Path("data")
                data_dir.mkdir(exist_ok=True)
                output_path = data_dir / f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        else:
            output_path = None
        
        mode_str = "RECORD MODE" if enable_recording else "TELEOP MODE"
        print(f"WX200 Real Robot Control - {mode_str}")
        print("="*60)
        print("Features:")
        print("- SpaceMouse control")
        print("- Safe startup and shutdown sequences")
        if enable_recording:
            print(f"- Trajectory recording enabled")
            print(f"- Output file: {output_path}")
        print("="*60)
        
        TeleopControl(enable_recording=enable_recording, output_path=output_path).run()


if __name__ == "__main__":
    main()
