"""
ElevenLabs Conversational AI Agent with Music Control Tools
Supports local music playback and stop controls
"""

import signal
import sys
import os
import re
import socket
from typing import Optional, List, Dict, Any
import pygame

# ================= Configuration =================
API_KEY = "sk_***"
AGENT_ID = 'agent_2501kebqaav4f38tx61hbjhx6t7f'
MUSIC_BASE_DIR = r"./music"
SUPPORTED_MUSIC_EXTS = {".mp3", ".wav", ".ogg", ".flac"}

# Action server configuration
ACTION_SERVER_HOST = "10.160.199.224"
ACTION_SERVER_PORT = 12346

# Allowed action list
VALID_ACTIONS = [
    "raise up left hand",
    "raise up right hand",
    "raise up both hands",
    "raise up left hand higher",
    "raise up right hand higher",
    "raise up both hands higher",
    "look around",
    "thinking",
    "disagree",
    "give up",
    "go left",
    "go right",
    "old",
    "angry",
    "sad",
    "neutral",
    "point to left",
    "point to right",
    "relax stand",
    "none_tem0",
    "none_tem1"
]
# ===========================================

# Music playback state
music_state = {
    "initialized": False,
    "current_file": None,
}

# Mapping words to Arabic digits (supports fuzzy matching for speech recognition)
WORD_TO_NUMBER = {
    "zero": "0",
    "one": "1",
    "two": "2", 
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    # Chinese digits (kept via Unicode escapes to avoid literal CJK characters in source)
    "\u96f6": "0",  # ling
    "\u4e00": "1",  # yi
    "\u4e8c": "2",  # er
    "\u4e09": "3",  # san
    "\u56db": "4",  # si
    "\u4e94": "5",  # wu
    "\u516d": "6",  # liu
    "\u4e03": "7",  # qi
    "\u516b": "8",  # ba
    "\u4e5d": "9",  # jiu
    "\u5341": "10", # shi
}


def normalize_query(query: str) -> List[str]:
    """
    Normalize a query string and generate multiple possible search terms.
    Example: "music one" -> ["music one", "music 1", "music1"]
    """
    if not query:
        return []
    
    query = query.strip().lower()
    variants = [query]
    
    # Replace English/Chinese number words with Arabic digits
    converted = query
    for word, number in WORD_TO_NUMBER.items():
        if word in converted:
            converted = converted.replace(word, number)
    
    if converted != query:
        variants.append(converted)
        # Also add a version without spaces
        variants.append(converted.replace(" ", ""))
    
    # Add the original version without spaces
    variants.append(query.replace(" ", ""))
    
    return list(set(variants))  # de-duplicate


def ensure_mixer_initialized() -> bool:
    """Initialize pygame mixer"""
    global music_state
    try:
        
        if not music_state["initialized"]:
            os.environ.setdefault("SDL_AUDIODRIVER", "directsound")
            pygame.mixer.init()
            music_state["initialized"] = True
        return True
    except Exception as e:
        print(f"⚠️ Mixer initialization failed: {e}")
        return False


def search_music_files(base_dir: str, title: Optional[str] = None) -> List[str]:
    """
    Search music files with fuzzy matching
    """
    if not os.path.isdir(base_dir):
        return []
    
    matches: List[str] = []
    
    # Generate all possible search terms
    search_terms = normalize_query(title) if title else [""]
    
    for root, _, files in os.walk(base_dir):
        for f in files:
            _, ext = os.path.splitext(f)
            if ext.lower() not in SUPPORTED_MUSIC_EXTS:
                continue
            
            file_lower = f.lower()
            
            # If no search term, return all files
            if not title:
                matches.append(os.path.join(root, f))
                continue
            
            # Check whether any search term matches
            for term in search_terms:
                if term in file_lower:
                    matches.append(os.path.join(root, f))
                    break
    
    # Sort by modification time (newest first)
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches


def list_available_music() -> List[str]:
    """List all available music files"""
    if not os.path.isdir(MUSIC_BASE_DIR):
        return []
    
    music_files = []
    for root, _, files in os.walk(MUSIC_BASE_DIR):
        for f in files:
            _, ext = os.path.splitext(f)
            if ext.lower() in SUPPORTED_MUSIC_EXTS:
                music_files.append(f)
    return music_files

import time
def play_music(title: Optional[str] = None) -> Dict[str, Any]:
    """Play music"""
    
    
    if not os.path.isdir(MUSIC_BASE_DIR):
        return {"ok": False, "error": f"Music directory not found: {MUSIC_BASE_DIR}"}
    
    if not ensure_mixer_initialized():
        return {"ok": False, "error": "Failed to initialize audio system"}
    
    candidates = search_music_files(MUSIC_BASE_DIR, title)
    
    if not candidates:
        available = list_available_music()
        return {
            "ok": False, 
            "error": f"No matching music found for: {title}",
            "available": available[:5]  # Return the first 5 available files for reference
        }
    
    target = candidates[0]
    try:
        # Stop any current playback first
        try:
            pygame.mixer.music.stop()
        except:
            pass
        
        pygame.mixer.music.load(target)
        pygame.mixer.music.play()
        music_state["current_file"] = target
        
        filename = os.path.basename(target)

        print(f"🎵 Now playing: {filename}")
        return {"ok": True, "playing": filename, "path": target}
    
    except Exception as e:
        return {"ok": False, "error": f"Playback failed: {str(e)}", "file": target}


def stop_music() -> Dict[str, Any]:
    """Stop music playback"""
    
    
    if not ensure_mixer_initialized():
        return {"ok": False, "error": "Failed to initialize audio system"}
    
    try:
        pygame.mixer.music.stop()
        prev_file = music_state.get("current_file")
        music_state["current_file"] = None
        
        print("⏹️ Music stopped")
        return {"ok": True, "stopped": True, "previous": os.path.basename(prev_file) if prev_file else None}
    
    except Exception as e:
        return {"ok": False, "error": f"Stop failed: {str(e)}"}


def handle_play_music(parameters: Dict[str, Any]) -> str:
    """
    Handle play_music tool call
    """
    print(f"🔧 Tool called: play_music")
    print(f"   Parameters: {parameters}")
    
    title = parameters.get("title") or parameters.get("song_name") or parameters.get("keyword")
    result = play_music(title)
    if action_client.connect():
        action_client.send_message('none_tem2')
        time.sleep(3)
        action_client.send_message('none_tem0')

    time.sleep(2000)
    if result["ok"]:
        return f"Now playing: {result['playing']}"
    else:
        error_msg = result.get("error", "Unknown error")
        if "available" in result and result["available"]:
            available_list = ", ".join(result["available"])
            return f"Could not find '{title}'. Available songs: {available_list}"
        return f"Failed to play music: {error_msg}"


def handle_stop_music(parameters: Dict[str, Any]) -> str:
    """
    Handle stop_music tool call
    """
    print(f"🔧 Tool called: stop_music")
    print(f"   Parameters: {parameters}")
    
    result = stop_music()
    send_action("relax stand")
    time.sleep(1)
    send_action('none_tem1')
    if result["ok"]:
        if result.get("previous"):
            return f"Stopped playing: {result['previous']}"
        return "Music stopped"
    else:
        return f"Failed to stop music: {result.get('error', 'Unknown error')}"


# ------------------------------
# TextClient for sending actions to the server
# ------------------------------

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
            print(f"📤 Sent action: {text}")
            return True
        except Exception as e:
            print(f"⚠️ Send failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect"""
        if self.socket:
            try:
                self.socket.close()
                print("🔌 Disconnected from action server")
            except:
                pass
            finally:
                self.connected = False
                self.socket = None
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# Global action client
action_client = TextClient()


def normalize_action(action: str) -> Optional[str]:
    """
    Normalize a user-provided action into a valid action.
    Supports fuzzy matching.
    """
    if not action:
        return None
    
    action_lower = action.strip().lower()
    
    # Direct match
    for valid_action in VALID_ACTIONS:
        if valid_action == action_lower:
            return valid_action
    
    # Fuzzy match - check for substring containment
    for valid_action in VALID_ACTIONS:
        if action_lower in valid_action or valid_action in action_lower:
            return valid_action
    
    # Keyword match
    if "left" in action_lower and "hand" in action_lower:
        if "higher" in action_lower:
            return "raise up left hand higher"
        return "raise up left hand"
    elif "right" in action_lower and "hand" in action_lower:
        if "higher" in action_lower:
            return "raise up right hand higher"
        return "raise up right hand"
    elif "both" in action_lower and "hand" in action_lower:
        if "higher" in action_lower:
            return "raise up both hands higher"
        return "raise up both hands"
    elif "old" in action_lower:
        return "old"
    elif "angry" in action_lower or "anger" in action_lower:
        return "angry"
    elif "sad" in action_lower:
        return "sad"
    elif "neutral" in action_lower or "normal" in action_lower:
        return "neutral"
    
    return None


def send_action(action: str) -> Dict[str, Any]:
    """Send an action to the server"""
    normalized = normalize_action(action)
    
    if not normalized:
        return {
            "ok": False, 
            "error": f"Unknown action: {action}",
            "valid_actions": VALID_ACTIONS
        }
    
    if action_client.connect():
        if action_client.send_message(normalized):
            return {"ok": True, "action": normalized}
        else:
            return {"ok": False, "error": "Failed to send action to server"}
    else:
        return {"ok": False, "error": f"Failed to connect to action server at {ACTION_SERVER_HOST}:{ACTION_SERVER_PORT}"}


def handle_send_action(parameters: Dict[str, Any]) -> str:
    """
    Handle send_action tool call
    """
    print(f"🔧 Tool called: send_action")
    print(f"   Parameters: {parameters}")
    
    action = parameters.get("action")
    
    if not action:
        return f"No action specified. Valid actions: {', '.join(VALID_ACTIONS)}"
    
    result = send_action(action)
    
    if result["ok"]:
        return f"Action executed: {result['action']}"
    else:
        error_msg = result.get("error", "Unknown error")
        if "valid_actions" in result:
            return f"{error_msg}. Valid actions: {', '.join(result['valid_actions'])}"
        return f"Failed to send action: {error_msg}"


def main():
    # Lazy import to avoid failing at module import time
    try:
        from elevenlabs.client import ElevenLabs
        from elevenlabs.conversational_ai.conversation import Conversation, ClientTools
        from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface
    except ImportError as e:
        print(f"❌ Failed to import ElevenLabs SDK: {e}")
        print("   Please install: pip install elevenlabs")
        sys.exit(1)
    
    # Check music directory
    if not os.path.isdir(MUSIC_BASE_DIR):
        print(f"⚠️ Warning: Music directory not found: {MUSIC_BASE_DIR}")
        print("   Creating directory...")
        try:
            os.makedirs(MUSIC_BASE_DIR, exist_ok=True)
        except Exception as e:
            print(f"   Failed to create directory: {e}")
    else:
        music_files = list_available_music()
        print(f"🎵 Found {len(music_files)} music files in {MUSIC_BASE_DIR}")
        if music_files:
            print("   Available songs:")
            for f in music_files[:10]:
                print(f"   - {f}")
            if len(music_files) > 10:
                print(f"   ... and {len(music_files) - 10} more")
    
    # Initialize ElevenLabs client
    client = ElevenLabs(api_key=API_KEY)

    # Create and register client tools
    client_tools = ClientTools()
    client_tools.register("play_music", handle_play_music)
    client_tools.register("stop_music", handle_stop_music)
    client_tools.register("send_action", handle_send_action)
    print("🔧 Registered tools: play_music, stop_music, send_action")

    # Define callback functions
    def on_agent_response(response_text):
        print(f"🤖 Agent: {response_text}")

    def on_user_transcript(user_text):
        print(f"👤 You: {user_text}")
    
    def on_latency(latency_ms):
        print(f"📶 Latency: {latency_ms}ms")

    # Create conversation instance
    conversation = Conversation(
        client=client,
        agent_id=AGENT_ID,
        requires_auth=True,
        audio_interface=DefaultAudioInterface(),
        callback_agent_response=on_agent_response,
        callback_user_transcript=on_user_transcript,
        callback_latency_measurement=on_latency,
        client_tools=client_tools,  # registered client tools
    )

    print("\n" + "="*50)
    print("🎙️ ElevenLabs Voice Agent with Music Control")
    print("="*50)
    print("Connecting to ElevenLabs Agent...")
    
    conversation.start_session()
    print("✅ Connected! Start speaking (Press Ctrl+C to exit)...")
    print("="*50 + "\n")

    # Graceful shutdown handling
    def signal_handler(sig, frame):
        print("\n\n👋 Ending session...")
        try:
            conversation.end_session()
        finally:
            raise KeyboardInterrupt  # or sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Block until the session ends
    conversation_id = conversation.wait_for_session_end()
    print(f"\n✅ Session ended. ID: {conversation_id}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

