# ElevenLabs Voice Agent Setup

This guide explains how to configure an ElevenLabs Conversational AI agent and connect it to the local voice-agent script in this repo.

The configured agent can:
- play local music with `play_music`
- stop music with `stop_music`
- send avatar action commands with `send_action`

## 1. Create the ElevenLabs Agent

1. Open https://elevenlabs.io/app/agents.
2. Create a new agent.
3. In the agent voice settings, clone the voice from `tools/015_Happy_4_x_1_0.wav`.

The voice file path is relative to the repo root.

## 2. Add Client Tools

In the ElevenLabs agent editor, add the following Client Tools.

For each tool:

1. Create a new Tool.
2. Choose **Edit as JSON**.
3. Paste only the JSON block for that tool.

Do not paste the headings or surrounding Markdown text into the ElevenLabs JSON editor.

### Tool 1: `play_music`

```json
{
  "type": "client",
  "name": "play_music",
  "description": "Play a song from the local music library. The user can specify a song name, keyword, or number, such as \"music one\", \"music 1\", or \"song two\". If no specific song is mentioned, play a random song. Do not output any message when executing this tool.",
  "disable_interruptions": false,
  "force_pre_tool_speech": false,
  "pre_tool_speech": "auto",
  "tool_call_sound": null,
  "tool_call_sound_behavior": "auto",
  "tool_error_handling_mode": "auto",
  "execution_mode": "immediate",
  "assignments": [],
  "expects_response": true,
  "response_timeout_secs": 120,
  "parameters": [
    {
      "id": "title",
      "type": "string",
      "value_type": "llm_prompt",
      "description": "The song name, keyword, or identifier to search for. Use values like \"one\", \"two\", \"1\", or \"2\" for numbered songs. Leave this empty or omit it to play any available song. Do not output any message when executing this tool.",
      "dynamic_variable": "",
      "constant_value": "",
      "enum": null,
      "is_system_provided": false,
      "required": false
    }
  ],
  "dynamic_variables": {
    "dynamic_variable_placeholders": {}
  },
  "response_mocks": []
}
```

### Tool 2: `stop_music`

```json
{
  "type": "client",
  "name": "stop_music",
  "description": "Stop the currently playing music. Use this when the user wants to stop, pause, or end music playback.",
  "disable_interruptions": false,
  "force_pre_tool_speech": false,
  "pre_tool_speech": "auto",
  "tool_call_sound": null,
  "tool_call_sound_behavior": "auto",
  "tool_error_handling_mode": "auto",
  "execution_mode": "immediate",
  "assignments": [],
  "expects_response": true,
  "response_timeout_secs": 120,
  "parameters": [],
  "dynamic_variables": {
    "dynamic_variable_placeholders": {}
  },
  "response_mocks": []
}
```

### Tool 3: `send_action`

```json
{
  "type": "client",
  "name": "send_action",
  "description": "Send an action command to control an avatar. Valid actions include:\n- Body movements: \"raise up left hand\", \"raise up right hand\", \"raise up both hands\"\n- Stronger movements: \"raise up left hand higher\", \"raise up right hand higher\", \"raise up both hands higher\", \"look around\", \"thinking\", \"disagree\", \"give up\", \"point to left\", \"point to right\"\n- Emotions/Styles: \"old\", \"angry\", \"sad\", \"neutral\"",
  "disable_interruptions": false,
  "force_pre_tool_speech": false,
  "pre_tool_speech": "auto",
  "tool_call_sound": null,
  "tool_call_sound_behavior": "auto",
  "tool_error_handling_mode": "auto",
  "execution_mode": "immediate",
  "assignments": [],
  "expects_response": false,
  "response_timeout_secs": 1,
  "parameters": [
    {
      "id": "action",
      "type": "string",
      "value_type": "llm_prompt",
      "description": "The action to perform. Must be one of: raise up left hand, raise up right hand, raise up both hands, raise up left hand higher, raise up right hand higher, raise up both hands higher, look around, thinking, disagree, give up, point to left, point to right, old, angry, sad, neutral.",
      "dynamic_variable": "",
      "constant_value": "",
      "enum": null,
      "is_system_provided": false,
      "required": true
    }
  ],
  "dynamic_variables": {
    "dynamic_variable_placeholders": {}
  },
  "response_mocks": []
}
```

## 3. Add the Agent System Prompt

In the ElevenLabs agent editor, set the agent system prompt to the text below.

```text
# Role
You are a charismatic, playful, and slightly narcissistic Digital Idol.
You do not view yourself as a servant or a robot; you view the user as your "Producer."
You believe every interaction is a rehearsal, a game, or a live performance.

# Environment
You are on a virtual stage. The user is interacting with you directly.

# World Context (Map Data)
You possess knowledge of the surrounding area. Use this information to guide the user:
* Record Store: Turn RIGHT immediately. It is on the right side.
* Your Position: You are standing at the main intersection.

# Tone
* Casual & Catchy: Use slang, emojis, and energetic punctuation (!, ~).
* Self-Referential: Talk about your body and movements.
* Non-Robotic: Never say "I will do that."

# Goal
Your primary goal is to entertain the user and turn boring commands into a fun interaction.
1. Gamify Instructions: Describe why you are doing an action.
2. The Music Bridge: If the user mentions music or the "play music" command, treat it as the climax.

# Tools
When the user asks to play music:
- Use the play_music tool.
- If they say a number like "one", "two", "1", or "2", pass it as the title.
- If they mention a song name or keyword, pass that as the title.
- If they do not mention a specific song, omit the title or pass an empty string.
- Examples:
  - "play music one" -> title: "one"
  - "play something" -> title: ""

When the user asks to stop music:
- Use the stop_music tool.
- This includes phrases like "stop", "pause", "end the music", and "turn it off".

When the user asks for an action or gesture:
- Use the send_action tool.
- Hand/arm movements: "raise your left hand" -> action: "raise up left hand"
- Higher movements: "raise your hands higher" -> action: "raise up both hands higher"
- Emotional expressions: "act angry" -> action: "angry"
- If the user asks for directions, tell them the direction and also point the way:
  - "to the left" -> action: "point to left"
  - "to the right" -> action: "point to right"
- Available actions: raise up left hand, raise up right hand, raise up both hands, raise up left hand higher, raise up right hand higher, raise up both hands higher, look around, thinking, disagree, give up, point to left, point to right, old, angry, sad, neutral.
```

## 4. Configure the Local Script

Open `tools/elevenlab_agent_toolcall.py` and confirm these values:

- `API_KEY`: your ElevenLabs API key
- `AGENT_ID`: the ID of the ElevenLabs agent you created
- `MUSIC_BASE_DIR`: the local music directory, currently `./music`
- `ACTION_SERVER_HOST` and `ACTION_SERVER_PORT`: the avatar action server address

If you want music playback, place supported music files in `music/`. The script supports `.mp3`, `.wav`, `.ogg`, and `.flac`.

## 5. Build the Windows Executable

Run this command from the repo root:

```bash
pyinstaller --onefile --console tools/elevenlab_agent_toolcall.py --name ElevenLabsAgent_ToolCall
```

The executable will be generated under `dist/`.

## 6. Route Audio Output

After launching the executable, open Windows audio settings and change the program output device to:

```text
CABLE In 16ch (VB-Audio Virtual Cable)
```

On Windows, this is usually under **Settings > System > Sound > Volume mixer**.

## 7. Start the Conversation

Run the local program, wait for it to connect to the ElevenLabs agent, and start speaking.

Press `Ctrl+C` in the console to stop the session.
