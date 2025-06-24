import serial
import time
import sys

def send_alternating_serial_data(port, baud_rate, interval_seconds=10):
    """
    Sends alternating '1' and '0' values to a serial port at a specified interval.

    Args:
        port (str): The serial port to use (e.g., 'COM1' on Windows, '/dev/ttyUSB0' on Linux).
        baud_rate (int): The baud rate for the serial communication (e.g., 9600, 115200).
        interval_seconds (int): The time interval in seconds between sending data (default: 10).
    """
    try:
        # Initialize serial connection
        ser = serial.Serial(port, baud_rate, timeout=1)
        print(f"Opened serial port {port} at {baud_rate} baud.")

        data_to_send = 1
        while True:
            # Send the current data
            ser.write(str(data_to_send).encode())
            print(f"Sent: {data_to_send}")

            # Toggle data for the next send
            data_to_send = 1 - data_to_send  # This will alternate between 1 and 0

            # Wait for the specified interval
            time.sleep(interval_seconds)

    except serial.SerialException as e:
        print(f"Error: Could not open or communicate with serial port {port}. Please check:")
        print(f"  - Is the port correct? ({port})")
        print(f"  - Is the device connected?")
        print(f"  - Are the drivers installed?")
        print(f"  - Is another program using the port?")
        print(f"  Error details: {e}")
    except KeyboardInterrupt:
        print("\nScript terminated by user.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed.")

if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Replace with your actual serial port and baud rate
    SERIAL_PORT = '/dev/cu.usbserial-0001'  # Example for Windows. Use '/dev/ttyUSB0' or '/dev/ttyACM0' for Linux/Raspberry Pi
    BAUD_RATE = 9600      # Common baud rate. Match your device's baud rate.
    SEND_INTERVAL = 10    # Send data every 10 seconds

    print("Starting serial data sender. Press Ctrl+C to stop.")
    send_alternating_serial_data(SERIAL_PORT, BAUD_RATE, SEND_INTERVAL)