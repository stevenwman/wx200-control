import time
import sys
from dynamixel_sdk import *

# --- USER CONFIGURATION ---
HOME_POSITIONS = [2043, 1581, 2515, 1710, 1619, 2057, 1784]
MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7] 

# Speed Limit (0 = Max, 30 = Slow/Safe)
VELOCITY_LIMIT = 30  

BAUDRATE = 1000000
DEVICENAME = '/dev/ttyUSB0'
PROTOCOL_VERSION = 2.0

# Addresses
ADDR_TORQUE_ENABLE    = 64
ADDR_GOAL_POSITION    = 116
ADDR_PROFILE_VELOCITY = 112 

# --- HELPER FUNCTIONS ---

def disable_torque(port, packet, ids):
    """Turns off torque for all specified IDs."""
    print("\n--- DISABLING TORQUE (MOTORS WILL GO LIMP) ---")
    for dxl_id in ids:
        packet.write1ByteTxRx(port, dxl_id, ADDR_TORQUE_ENABLE, 0)
    print("Torque disabled.")

def move_to_home(port, packet, ids, positions, speed_limit):
    print(f"Enabling Torque and moving to Home (Speed: {speed_limit})...")
    
    # 1. Enable Torque
    for dxl_id in ids:
        packet.write1ByteTxRx(port, dxl_id, ADDR_TORQUE_ENABLE, 1)

    # 2. Set Speed Limit
    for dxl_id in ids:
        packet.write4ByteTxRx(port, dxl_id, ADDR_PROFILE_VELOCITY, speed_limit)

    # 3. Send Move Command
    for dxl_id, goal_pos in zip(ids, positions):
        print(f"Moving ID {dxl_id} to {goal_pos}")
        packet.write4ByteTxRx(port, dxl_id, ADDR_GOAL_POSITION, goal_pos)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    portHandler = PortHandler(DEVICENAME)
    packetHandler = PacketHandler(PROTOCOL_VERSION)

    # Open Port
    if not portHandler.openPort():
        print("Failed to open port")
        quit()

    # Set Baud Rate
    if not portHandler.setBaudRate(BAUDRATE):
        print("Failed to set baudrate")
        quit()

    try:
        # --- NORMAL OPERATION ---
        move_to_home(portHandler, packetHandler, MOTOR_IDS, HOME_POSITIONS, VELOCITY_LIMIT)
        
        print("Movement sent. Holding position...")
        print(">> PRESS CTRL+C TO ABORT AND KILL TORQUE <<")
        
        # Keep the script running so we can catch Ctrl+C
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        # --- EXCEPTION HANDLING ---
        # This block runs ONLY when you press Ctrl+C
        print("\nKeyboard Interrupt detected!")
        disable_torque(portHandler, packetHandler, MOTOR_IDS)
        
    finally:
        # --- CLEANUP ---
        # This always runs, ensuring the port closes properly
        portHandler.closePort()
        print("Port closed. Exiting.")