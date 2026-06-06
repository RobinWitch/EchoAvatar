import sounddevice as sd
import keyboard
import socket
import pickle

SERVER_IP = '10.160.199.224' # change it to your server's IP
SERVER_PORT = 12345
input_device_index = 2 # change it to the device index of "CABLE Output (VB-Audio Virtual Cable)" on your machine, you can find it by running get_device.py
channels = 1
rate = 24000
chunk = 1000

# setup client socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, SERVER_PORT))

# open the audio stream
stream = sd.InputStream(
    device=input_device_index,
    channels=channels,
    samplerate=rate,
    blocksize=chunk
)

try:
    print("Start transmitting audio...")
    with stream:
        while True:
            # Load audio data from the stream
            data, overflowed = stream.read(chunk)
            
            # Convert numpy array to bytes for transmission
            data_bytes = pickle.dumps(data)
            
            # send data size
            size = len(data_bytes)
            # print(f"Sending chunk of size: {size}")
            client_socket.send(size.to_bytes(4, byteorder='big'))
            
            # send data
            client_socket.send(data_bytes)
            
            if keyboard.is_pressed('ctrl+space'):
                print("Detected ctrl+space key, stopping recording.")
                # send end signal
                client_socket.send((0).to_bytes(4, byteorder='big'))
                break

finally:
    client_socket.close()
    print("Connection closed.")