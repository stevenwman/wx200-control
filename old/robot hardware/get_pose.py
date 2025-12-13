import os
from dynamixel_sdk import *

# --- Setup ---
# Based on your previous success
BAUDRATE = 1000000
DEVICENAME = '/dev/ttyUSB0'
PROTOCOL_VERSION = 2.0

# DYNAMIXEL X-Series Addresses
ADDR_TORQUE_ENABLE = 64
ADDR_PRESENT_POSITION = 132

# The IDs you want to read
target_ids = [1, 2, 3, 4, 5, 6, 7]

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open Port
if not portHandler.openPort():
    print("Failed to open port")
    quit()

if not portHandler.setBaudRate(BAUDRATE):
    print("Failed to set baudrate")
    quit()

print("--------------------------------")
print("Reading positions for IDs: ", target_ids)
print("Move the robot to your desired HOME position now.")
input("Press Enter when ready to capture...")
print("--------------------------------")

captured_positions = {}

# Loop through each ID and read position
for dxl_id in target_ids:
    # Read Present Position (Length 4 bytes)
    dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRESENT_POSITION)

    if dxl_comm_result != COMM_SUCCESS:
        # If the motor doesn't answer (e.g., ID 6 doesn't exist), just skip it
        print(f"[ID {dxl_id}] Failed to respond (Check ID or Cable)")
    elif dxl_error != 0:
        print(f"[ID {dxl_id}] Error: {packetHandler.getRxPacketError(dxl_error)}")
    else:
        # Success!
        print(f"[ID {dxl_id}] Current Position: {dxl_present_position}")
        captured_positions[dxl_id] = dxl_present_position

# Disable Torque for all found motors (Optional, just to be safe)
for dxl_id in captured_positions:
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)

print("--------------------------------")
print("COPY THIS LIST FOR YOUR NEXT SCRIPT:")
print(f"HOME_POSITIONS = {list(captured_positions.values())}")
print("--------------------------------")

portHandler.closePort()