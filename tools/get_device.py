import sounddevice as sd

# List all available audio devices
devices = sd.query_devices()

# Print information for all devices
print("All devices:")
for i, device in enumerate(devices):
    print(
        f"Device {i}: {device['name']} "
        # f"(input channels: {device['max_input_channels']}; "
        # f"output channels: {device['max_output_channels']})"
    )
default_input = sd.query_devices(kind="input")
print(f"\nDefault input device: {default_input['name']} (index: {sd.default.device[0]})")
