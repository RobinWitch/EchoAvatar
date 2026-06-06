# ------------------------------
# TextClient for sending actions to the server
# ------------------------------
# Action server configuration
import socket
import sys
import time
import os  # for clearing screen
import keyboard  # for global hotkey listening
ACTION_SERVER_HOST = "10.160.199.224"
ACTION_SERVER_PORT = 12346

action_map = {
    "1":"none",
    "2": "raise up left hand",
    "3": "raise up right hand",
    "4": "raise up both hands",
    "5": "raise up both hands higher",
    "6": "point to left",
    "7": "point to right",
}

class TextClient:
    def __init__(self, host: str = ACTION_SERVER_HOST, port: int = ACTION_SERVER_PORT):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
    
    def connect(self) -> bool:
        """Connect to the server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            return True
        except Exception as e:
            print(f"⚠️ Connection failed: {e}")
            try:
                if self.socket:
                    self.socket.close()
            except Exception:
                pass
            finally:
                self.socket = None
            self.connected = False
            return False
    
    def send_message(self, text: str) -> bool:
        """Send a message to the server"""
        if not self.connected:
            print("⚠️ Not connected to server")
            return False
        
        try:
            data = text.encode('utf-8')
            msg_length = len(data)
            length_bytes = msg_length.to_bytes(4, byteorder='big')
            self.socket.sendall(length_bytes)
            self.socket.sendall(data)
            print(f"Sent action: {text}")
            return True
        except Exception as e:
            print(f"Send failed: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from the server"""
        if self.socket:
            try:
                self.socket.close()
                print("🔌 Disconnected from action server")
            except Exception:
                pass
            finally:
                self.connected = False
                self.socket = None


def _connect_with_retry(client: TextClient, retry_interval_s: float = 2.0) -> None:
    """Keep retrying until connected (Ctrl+C to stop)."""
    while not client.connected:
        print(f"🔌 Connecting to action server {client.host}:{client.port} ...")
        if client.connect():
            print("✅ Connected")
            return
        print(f"⏳ Retry in {retry_interval_s:.1f}s (Ctrl+C to quit)")
        time.sleep(retry_interval_s)


def _read_keypress_blocking() -> str:
    """
    Read a single keypress.
    - On Windows: uses msvcrt for immediate (no-Enter) input.
    - Fallback: uses input() (requires Enter).
    """
    if sys.platform.startswith("win"):
        import msvcrt  # Windows only

        ch = msvcrt.getwch()
        # Handle special keys (arrow keys, function keys, etc.)
        if ch in ("\x00", "\xe0"):
            try:
                msvcrt.getwch()  # swallow the second char
            except Exception:
                pass
            return ""
        return ch

    # Non-Windows fallback (press Enter)
    try:
        s = input("> ").strip()
        return s[:1] if s else ""
    except EOFError:
        return "q"


def main():
    client = TextClient()
    _connect_with_retry(client)

    print("\n🎮 Global hotkey control started (works in background):")
    for k in sorted(action_map.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        print(f"  {k}: {action_map[k]}")
    print("  c: Clear screen")
    print("  q / ESC: Exit program\n")
    print("⚡ You can now minimize the window, the program will listen for keys in the background!")

    def on_key_press(key_event):
        """Handle key press event"""
        key = key_event.name
        
        # Handle exit keys
        if key in ("q", "esc"):
            print("\n👋 Exiting...")
            client.disconnect()
            keyboard.unhook_all()
            sys.exit(0)
        
        # Handle screen clearing
        if key == "c":
            os.system('cls' if os.name == 'nt' else 'clear')
            print("\n🎮 Global hotkey control started (works in background):")
            for k in sorted(action_map.keys(), key=lambda x: int(x) if x.isdigit() else 999):
                print(f"  {k}: {action_map[k]}")
            print("  c: Clear screen")
            print("  q / ESC: Exit program\n")
            print("⚡ You can now minimize the window, the program will listen for keys in the background!")
            return
        
        # Handle action mapping
        action = action_map.get(key)
        if not action:
            return
        
        # Ensure connection
        if not client.connected:
            _connect_with_retry(client)
        
        ok = client.send_message(action)
        if not ok:
            client.disconnect()
            _connect_with_retry(client)
            client.send_message(action)

    try:
        # Register global hotkey listener
        keyboard.on_press(on_key_press)
        
        # Keep program running, waiting for ESC or q to exit
        keyboard.wait()
        
    except KeyboardInterrupt:
        print("\n👋 Received interrupt signal...")
    finally:
        keyboard.unhook_all()
        client.disconnect()
        print("👋 Bye")


if __name__ == "__main__":
    main()