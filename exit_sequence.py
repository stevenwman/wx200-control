import time
import sys
from dynamixel_sdk import *

# --- POSE DEFINITIONS ---
# -1 means "Ignore this motor / Keep current position"
# IDs:                 [1,    2,    3,    4,    5,    6,    7]
REASONABLE_HOME_POSE = [-1,   1382, 2712, 1568, 1549, 2058, 1784]
BASE_HOME_POSE       = [2040, -1,   -1,   -1,   -1,   -1,   -1]
FOLDED_HOME_POSE     = [2040, 846,  3249, 958,  1944, 2057, 1784]

MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7]

# --- CONFIGURATION ---
VELOCITY_LIMIT = 50       # Speed limit (0=Max, 30=Slow, 100=Medium)
MOVE_DELAY     = 2.0      # Seconds to wait between moves
BAUDRATE       = 1000000
DEVICENAME     = '/dev/ttyUSB0'
PROTOCOL_VERSION = 2.0

# Addresses
ADDR_TORQUE_ENABLE    = 64
ADDR_GOAL_POSITION    = 116
ADDR_PROFILE_VELOCITY = 112

# --- FUNCTIONS ---

def move_to_pose(port, packet, ids, positions, speed_limit):
    """
    Moves motors to positions. Skips any motor where position is -1.
    """
    # 1. Set Speed Limit first
    for dxl_id in ids:
        packet.write4ByteTxRx(port, dxl_id, ADDR_PROFILE_VELOCITY, speed_limit)
        # Ensure torque is on (just in case)
        packet.write1ByteTxRx(port, dxl_id, ADDR_TORQUE_ENABLE, 1)

    # 2. Send Move Commands (Ignoring -1s)
    print(f"Moving to target pose...")
    for dxl_id, goal_pos in zip(ids, positions):
        if goal_pos == -1:
            continue # SKIP this motor, let it stay where it is
        
        dxl_comm_result, dxl_error = packet.write4ByteTxRx(port, dxl_id, ADDR_GOAL_POSITION, goal_pos)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"  [ID {dxl_id}] Write Error: {packet.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"  [ID {dxl_id}] Packet Error: {packet.getRxPacketError(dxl_error)}")

def disable_torque_all(port, packet, ids):
    print("Disabling Torque for all motors...")
    for dxl_id in ids:
        packet.write1ByteTxRx(port, dxl_id, ADDR_TORQUE_ENABLE, 0)

def main_work():
    # SETUP
    portHandler = PortHandler(DEVICENAME)
    packetHandler = PacketHandler(PROTOCOL_VERSION)

    if not portHandler.openPort():
        print("Failed to open port")
        quit()
    if not portHandler.setBaudRate(BAUDRATE):
        print("Failed to set baudrate")
        quit()

    print("System Ready. Press Ctrl+C to trigger Shutdown Sequence...")

    try:
        # --- YOUR MAIN PROGRAM LOOP WOULD GO HERE ---
        while True:
            # Doing robot stuff...
            time.sleep(1)
            print("Robot is running... (Press Ctrl+C to stop)")
            
    except KeyboardInterrupt:
        print("\n\n!!! ESCAPE DETECTED - STARTING SHUTDOWN SEQUENCE !!!")
        
    finally:
        # --- THIS BLOCK ALWAYS RUNS, EVEN ON ERROR OR EXIT ---
        
        # 1. Go to Reasonable Home (skips ID 1)
        print("Step 1: Reasonable Home")
        move_to_pose(portHandler, packetHandler, MOTOR_IDS, REASONABLE_HOME_POSE, VELOCITY_LIMIT)
        time.sleep(MOVE_DELAY)

        # 2. Go to Base Home (Only moves ID 1)
        print("Step 2: Aligning Base")
        move_to_pose(portHandler, packetHandler, MOTOR_IDS, BASE_HOME_POSE, VELOCITY_LIMIT)
        time.sleep(MOVE_DELAY)

        # 3. Fold Up (Moves everything)
        print("Step 3: Folding to Rest")
        move_to_pose(portHandler, packetHandler, MOTOR_IDS, FOLDED_HOME_POSE, VELOCITY_LIMIT)
        time.sleep(MOVE_DELAY)

        # 4. Kill Power
        disable_torque_all(portHandler, packetHandler, MOTOR_IDS)
        
        portHandler.closePort()
        print("Shutdown Complete. Robot is limp.")

if __name__ == "__main__":
    main_work()