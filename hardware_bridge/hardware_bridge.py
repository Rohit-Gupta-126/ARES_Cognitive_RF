import serial
import serial.tools.list_ports
import os
import hashlib
import threading
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ZenTropyBridge:
    """
    Interfaces with the physical ZenTropy Key (ZEK) or fallback software entropy sources.
    Uses SHAKE-128 to expand entropy into uniformly distributed frequency hopping choices.
    """
    # Common USB-to-Serial VID/PID list (e.g. CP2102, CH340, FTDI, Arduino)
    KNOWN_VID_PIDS = [
        (0x10C4, 0xEA60), # CP2102 USB to UART
        (0x1A86, 0x7523), # CH340 USB to UART
        (0x0403, 0x6001), # FTDI FT232R USB to UART
        (0x2341, 0x0043), # Arduino Uno
        (0x16C0, 0x0483), # Teensy CDC
    ]

    def __init__(self, port=None, baudrate=115200, mock=False):
        self.port = port
        self.baudrate = baudrate
        self.mock_mode = mock
        self.serial_conn = None
        self.is_connected = False
        self.lock = threading.Lock()
        
        # Reconnection thread variables
        self.reconnect_thread = None
        self.stop_reconnect = threading.Event()
        
        # Telemetry info
        self.entropy_source = "SOFTWARE_ENTROPY" # "HARDWARE_ENTROPY" or "SOFTWARE_ENTROPY"
        
        if not self.mock_mode:
            self.connect()

    def find_zek_port(self):
        """Scans COM ports looking for known VID/PID or serial devices."""
        ports = serial.tools.list_ports.comports()
        for p in ports:
            # Check if VID/PID matches known chips
            if p.vid is not None and p.pid is not None:
                if (p.vid, p.pid) in self.KNOWN_VID_PIDS:
                    logging.info(f"Auto-detected compatible USB-Serial chip at {p.device} (VID:{p.vid:04X} PID:{p.pid:04X})")
                    return p.device
            # Check description keywords
            desc = p.description.lower()
            if any(keyword in desc for keyword in ["zentropy", "zek", "truerng", "serial", "ch340", "cp210", "ftdi"]):
                logging.info(f"Auto-detected port based on description: {p.device} ({p.description})")
                return p.device
        return None

    def connect(self):
        """Attempts connection to the ZEK serial device."""
        with self.lock:
            # If explicit port not provided, try auto-detection
            target_port = self.port
            if not target_port:
                target_port = self.find_zek_port()
            
            # If still no port, default to COM3 on Windows
            if not target_port:
                target_port = "COM3"
                logging.warning(f"No ZEK port auto-detected. Defaulting to {target_port}")
                
            logging.info(f"Attempting connection to ZEK on {target_port} at {self.baudrate} baud...")
            try:
                self.serial_conn = serial.Serial(
                    port=target_port,
                    baudrate=self.baudrate,
                    timeout=0.1, # Short read timeout to keep loops non-blocking
                    write_timeout=0.1
                )
                self.is_connected = True
                self.entropy_source = "HARDWARE_ENTROPY"
                logging.info(f"Successfully connected to physical ZEK on {target_port}!")
                
                # Stop any active reconnection threads if we are successfully connected
                if self.reconnect_thread and self.reconnect_thread.is_alive():
                    self.stop_reconnect.set()
            except serial.SerialException as e:
                self.is_connected = False
                self.entropy_source = "SOFTWARE_ENTROPY"
                self.serial_conn = None
                logging.error(f"Failed to open port {target_port}: {e}")
                logging.warning("Entering SOFTWARE_ENTROPY fallback mode.")
                self.start_reconnection_thread()

    def start_reconnection_thread(self):
        """Starts a background thread that attempts to reconnect to the ZEK device."""
        if self.reconnect_thread and self.reconnect_thread.is_alive():
            return # Thread is already running
        
        self.stop_reconnect.clear()
        self.reconnect_thread = threading.Thread(target=self._reconnect_loop, daemon=True)
        self.reconnect_thread.start()
        logging.info("Background ZEK reconnection thread started.")

    def _reconnect_loop(self):
        """Reconnection loop executed in the background thread."""
        while not self.stop_reconnect.is_set():
            time.sleep(5.0) # Check every 5 seconds
            if self.is_connected:
                break
                
            # Attempt to find the port and connect
            target_port = self.port or self.find_zek_port()
            if target_port:
                try:
                    conn = serial.Serial(
                        port=target_port,
                        baudrate=self.baudrate,
                        timeout=0.1
                    )
                    with self.lock:
                        self.serial_conn = conn
                        self.is_connected = True
                        self.entropy_source = "HARDWARE_ENTROPY"
                    logging.info(f"ZEK reconnected successfully on {target_port}!")
                    break
                except serial.SerialException:
                    pass # Continue retrying

    def read_raw_entropy(self, num_bytes=16):
        """Reads raw binary entropy from ZEK, falling back to os.urandom if disconnected."""
        if self.mock_mode or not self.is_connected:
            self.entropy_source = "SOFTWARE_ENTROPY"
            return os.urandom(num_bytes)
            
        with self.lock:
            try:
                # Flush input buffer to ensure fresh entropy
                self.serial_conn.reset_input_buffer()
                # Read raw bytes
                raw = self.serial_conn.read(num_bytes)
                
                # If we read fewer bytes than requested (due to timeout), fallback
                if len(raw) < num_bytes:
                    logging.warning(f"ZEK read timeout. Requested {num_bytes} bytes, got {len(raw)} bytes. Falling back to software.")
                    self.entropy_source = "SOFTWARE_ENTROPY"
                    # Supplement with software bytes
                    raw += os.urandom(num_bytes - len(raw))
                    
                return raw
            except (serial.SerialException, OSError) as e:
                logging.error(f"ZEK read failed (hardware likely unplugged): {e}")
                self.is_connected = False
                self.entropy_source = "SOFTWARE_ENTROPY"
                self.serial_conn = None
                self.start_reconnection_thread()
                return os.urandom(num_bytes)

    def get_random_channel(self, safe_channels):
        """
        Selects a random channel index from the list of safe channels.
        Uses SHAKE-128 to sponge and expand raw ZEK entropy.
        """
        if not safe_channels:
            raise ValueError("safe_channels list cannot be empty.")
            
        # 1. Harvest 16 bytes of raw entropy
        raw_bytes = self.read_raw_entropy(num_bytes=16)
        
        # 2. Feed through SHAKE-128 cryptographic sponge
        sponge = hashlib.shake_128(raw_bytes)
        # Squeeze out 4 bytes to form a 32-bit unsigned integer
        digest = sponge.digest(4)
        
        # 3. Translate to integer
        val = int.from_bytes(digest, byteorder='big')
        
        # 4. Uniformly select index
        selected_idx = val % len(safe_channels)
        return safe_channels[selected_idx]

    def close(self):
        """Closes serial connection."""
        self.stop_reconnect.set()
        with self.lock:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
                self.is_connected = False
                logging.info("ZEK Serial connection closed.")
