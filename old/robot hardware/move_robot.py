import os
from dynamixel_sdk import * # Uses Dynamixel SDK library

# --- Configuration ---
# Control Table Addresses for DYNAMIXEL X-Series (XL430-W250, etc.)
ADDR_TORQUE_ENABLE      = 64
ADDR_GOAL_POSITION      = 116
ADDR_PRESENT_POSITION   = 132

# Data Byte Lengths
LEN_GOAL_POSITION       = 4
LEN_PRESENT_POSITION    = 4

# Protocol Version
PROTOCOL_VERSION        = 2.0

# Default ID for OpenManipulator Joint 1 is usually 11
DXL_ID                  = 1
BAUDRATE                = 1000000             # Default for OpenManipulator
DEVICENAME              = '/dev/ttyUSB0'      # Check your port name

TORQUE_ENABLE           = 1                   # Value for enabling the torque
TORQUE_DISABLE          = 0                   # Value for disabling the torque
DXL_MINIMUM_POSITION_VALUE  = 0               # Dynamixel will rotate between this value
DXL_MAXIMUM_POSITION_VALUE  = 4095            # and this value (approx 0 to 360 degrees)
MOVING_STATUS_THRESHOLD     = 20              # Dynamixel moving status threshold

# --- Setup ---

# Initialize PortHandler instance
portHandler = PortHandler(DEVICENAME)

# Initialize PacketHandler instance
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port
if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    quit()

# Set port baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    quit()

# --- Execution ---

# 1. Enable Torque
# write1ByteTxRx means "Write 1 Byte, Transmit & Receive response"
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Torque enabled for ID: %d" % DXL_ID)

# 2. Write Goal Position (Move to 2048 - roughly center)
goal_pos = 2048
print(f"Moving to position: {goal_pos}")

# write4ByteTxRx is used because Position data is 4 bytes
dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_POSITION, goal_pos)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))

# 3. Read Present Position
dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
print("Present Position of ID %d: %03d" % (DXL_ID, dxl_present_position))

# 4. Disable Torque (Optional, if you want to move it by hand afterwards)
# packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)

# Close port
portHandler.closePort()