import numpy as np
import pyaudio
import winsound
from datetime import datetime
import time

frequency = 2500
duration = 1000

def audio_detection(log_list=None):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    THRESHOLD = 2000

    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"[ERROR] Could not initialize audio stream: {e}")
        return

    print("Listening for audio...")

    suspicious_audio_detected = False

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)

            if np.max(np.abs(audio_data)) > THRESHOLD and not suspicious_audio_detected:
                print("Suspicious audio detected!")
                winsound.Beep(frequency, duration)
                suspicious_audio_detected = True

                if log_list is not None:
                    log_entry = f"Suspicious audio detected at {datetime.now().strftime('%H:%M:%S.%f')}"
                    print(log_entry)
                    log_list.append(log_entry)

            elif np.max(np.abs(audio_data)) < THRESHOLD:
                suspicious_audio_detected = False

            time.sleep(0.1)

        except Exception as e:
            print(f"[ERROR] Audio detection error: {e}")
            break

    print("Stopping audio detection...")
    stream.stop_stream()
    stream.close()
    p.terminate()
