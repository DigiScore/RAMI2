import logging
import numpy as np
import pyaudio
from random import random
from scipy import signal
from scipy.io import wavfile
from time import time

import config
from nebula.hivemind import DataBorg


def buffer_scaler(in_feature, mins, maxs):
    in_feature = np.array(in_feature)
    mins = np.array(mins)[:, np.newaxis]
    maxs = np.array(maxs)[:, np.newaxis]
    in_feature = (in_feature - mins) / (maxs - mins)
    in_feature = in_feature.clip(0, 1)
    return in_feature


class Listener:
    def __init__(self):
        """
        controls audio listening by opening up a stream in Pyaudio.
        """
        print("Starting listener")

        # self.running = True
        self.connected = False
        self.logging = False
        self.mic_logging = config.mic_in_logging

        # Set up mic listening
        self.CHUNK = 2**11
        self.RATE = 44100
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.RATE,
                                  input=True,
                                  frames_per_buffer=self.CHUNK)

        self.mic_sensitivity = config.mic_sensitivity

        print(f"Checking mic volume - Please play a sustained note or chords for at least {config.volume_seconds} seconds at the correct volume.")
        start_time = time()
        counting_time = False
        while True:
            data = np.fromstring(self.stream.read(self.CHUNK), dtype=np.int16)
            peak = int(np.average(np.abs(data)) * 2)
            bars = "#" * int(50 * peak / 2 ** 16)
            if peak < config.volume_range[0]:
                counting_time = False
                print(f"\rSound Check: {peak} {bars} - VOLUME TOO LOW. Please turn it up! ", flush=True, end="")
            elif peak > config.volume_range[1]:
                counting_time = False
                print(f"\rSound Check: {peak} {bars} - VOLUME TOO LOUD. Please turn it up! ", flush=True, end="")
            else:
                print("\rVOLUME OK! Please keep playing a sustained note or chords", flush=True, end="")
                if not counting_time:
                    start_time = time()
                    counting_time = True
            if counting_time and time() - start_time > config.volume_seconds:
                print("\rVOLUME OK!!", flush=True, end="\n")
                break

        # Plug into the hive mind data borg
        self.hivemind = DataBorg()

    def snd_listen(self):
        """
        Loop thread listening to live sound and analysing amplitude. Normalises
        then stores this into the nebula dataclass for shared use.
        """
        logging.info("Starting mic listening stream & thread")
        data_buffer = np.empty(0)

        # Set silence listener to 10 seconds in future
        silence_timer = time() + 10
        first_minute = time() + 60

        # Main loop
        while self.hivemind.running:
            # Get amplitude from mic input
            data = np.frombuffer(self.stream.read(
                self.CHUNK,
                exception_on_overflow=False),
                dtype=np.int16)

            # Make audio envelope buffer for nets
            data_buffer = np.append(data_buffer, data)
            self.hivemind.audio_buffer_raw = np.append(
                self.hivemind.audio_buffer_raw, data)
            if len(data_buffer) > self.RATE*5:  # 5 sec buffer
                data_buffer = data_buffer[-(self.RATE*5):]
                hb_data = signal.hilbert(data_buffer)
                envelope = np.abs(hb_data)
                num = int(len(envelope)/self.RATE*10.0)
                envelope = signal.resample(envelope, num)[np.newaxis, :]
                envelope_norm = buffer_scaler(envelope,
                                              self.hivemind.audio_mins,
                                              self.hivemind.audio_maxs)
                self.hivemind.audio_buffer = envelope_norm

            peak = np.average(np.abs(data)) * 2
            if peak > 1000:
                bars = "#" * int(50 * peak / 2 ** 16)
                logging.debug(f"MIC LISTENER: {peak} {bars}")
                if self.mic_logging:
                    print(f"MIC LISTENER: {peak} {bars}")

            # Reset the silence listener
            silence_timer = time() + 5   # 5 seconds ahead

            # Normalise it for range 0.0 - 1.0
            normalised_peak = ((peak - 0) / (self.mic_sensitivity - 0)) * (1 - 0) + 0  # peak / self.mic_sensitivity
            if normalised_peak > 1.0:
                normalised_peak = 1.0

            # Put normalised amplitude into Nebula's dictionary for use
            self.hivemind.mic_in = normalised_peak

            # If loud sound then 63% affect gesture manager
            if normalised_peak > 0.8:
                if random() > 0.63:
                    self.hivemind.interrupted = True
                    self.hivemind.randomiser()
                    print("-------------- MICROPHONE INTERRUPT --------------")

            # Check human musician induced ending (wait for 5 secs)
            if config.silence_listener:
                if time() > first_minute:
                    if time() >= silence_timer:
                        self.hivemind.running = False

            # Make all sense data for self_flow prediction in hivemind
            self.hivemind.make_all_sense_data()

        logging.info('quitting listener thread')
        # self.terminate_listener()

    def terminate_listener(self):
        wavfile.write(f'data/{self.hivemind.session_date}/{self.hivemind.session_date}.wav', self.RATE,
                      self.hivemind.audio_buffer_raw.astype(np.int16))
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

