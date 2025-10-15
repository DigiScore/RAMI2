import numpy as np
import pickle
from datetime import datetime
from random import random, randrange


# DataBorg Pattern
# https://www.oreilly.com/library/view/python-cookbook/0596001673/ch05s23.html
# https://stackoverflow.com/questions/1318406/why-is-the-borg-pattern-better-than-the-singleton-pattern-in-python
class DataBorg:
    __hivemind = None

    def __init__(self):
        if not DataBorg.__hivemind:
            DataBorg.__hivemind = self.__dict__

            self.session_date = datetime.now().strftime("%Y_%m_%d_%H%M")

            ######################
            # Outputs from NNets in AI Factory rework
            ######################
            self.audio2eda: float = random()
            self.audio2eda_2d: np.array = np.random.uniform(size=(1, 50))

            self.flow2core: float = random()
            self.flow2core_2d: np.array = np.random.uniform(size=(2, 50))

            self.core2flow: float = random()
            self.core2flow_2d: np.array = np.random.uniform(size=(1, 50))

            self.audio2core: float = random()
            self.audio2core_2d: np.array = np.random.uniform(size=(2, 50))

            self.audio2flow: float = random()
            self.audio2flow_2d: np.array = np.random.uniform(size=(1, 50))

            self.flow2audio: float = random()
            self.flow2audio_2d: np.array = np.random.uniform(size=(1, 50))

            self.eda2flow: float = random()
            self.eda2flow_2d: np.array = np.random.uniform(size=(1, 50))

            self.all2flow: float = random()
            self.all2flow_2d: np.array = np.random.uniform(size=(3, 50))

            ######################
            # Human inputs
            ######################
            self.mic_in: float = random()
            """Percept input stream from client e.g. live mic level"""

            with open('./nebula/models/audio2core_minmax.pickle', 'rb') as f:
                audio_mins, audio_maxs = pickle.load(f)
            self.audio_mins: list = audio_mins
            self.audio_maxs: list = audio_maxs

            self.audio_buffer_raw = np.empty(0)
            self.audio_buffer: np.array = np.random.uniform(size=(1, 50))

            self.eeg_buffer_raw: np.array = np.random.uniform(size=(4, 50))
            """Live 5 sec buffered raw data from brainbit"""

            self.eeg_buffer: np.array = np.random.uniform(size=(4, 50))
            """Live 5 sec buffered normalised data from brainbit"""

            self.eda_buffer_raw: np.array = np.random.uniform(size=(1, 50))
            """Live 5 sec buffered raw data from bitalino"""

            self.eda_buffer: np.array = np.random.uniform(size=(1, 50))
            """Live 5 sec buffered normalised data from bitalino"""

            self.all_sense_input: np.array = np.random.uniform(size=(3, 50))
            """Live 5 sec buffered normalised data from all sense input
            eda, eeg 1-4, core 1-2"""

            ######################
            # Bitalino streams
            ######################

            # self.bitalino_x: int = 0
            #
            # self.bitalino_y: int = 0
            #
            # self.bitalino_z: int = 0
            #
            # self.bitalino_eda: int = 0
            #
            # self.bitalino_ecg: int = 0
            #
            # self.bitalino_rsp: int = 0

            # self.bitalino_button: int = 0

            ######################
            # Additional streams
            ######################
            self.master_stream: float = random()
            """Master output from the affect process"""

            self.rnd_poetry: float = random()
            """Random stream to spice things up"""

            self.thought_train_stream: str = " "
            """Current stream chosen by affect process"""

            self.rhythm_rate: float = randrange(30, 100) / 100
            """Internal clock/ rhythm sub division"""

            self.design_decision: str = " "
            """Logs the current design decision"""


            ######################
            # Robot vars
            ######################
            self.current_robot_x_y_z: tuple = (0, 0, 0)
            """Normalised cartesian robot coords"""

            self.current_robot_x_y: np.array = np.zeros((2, 50))
            """Normalised 5 sec xy cartesian robot coords buffer"""

            self.current_nnet_x_y_z: tuple = (0, 0, 0)
            # TODO: 2 first elements could be assigned based on the NNets out
            # eg. flow2core or audio2core
            """Generated output of robot movement from NNets"""

            ######################
            # Running vars
            ######################
            self.interrupted: bool = False
            """Signals an interrupt to the gesture manager"""

            self.running: bool = False
            """Local running bool for single experiments"""

            self.MASTER_RUNNING: bool = True
            """Master running bool for whole script"""

        else:
            self.__dict__ = DataBorg.__hivemind

    def make_all_sense_data(self):
        """concat all sense input data for self_flow prediction"""
        self_flow_data = np.empty((3, 0))

        eda_data = self.eda_buffer_raw[0].tolist()
        core_data0 = self.current_robot_x_y[0].tolist()
        core_data1 = self.current_robot_x_y[1].tolist()
        values = [
            [eda_data[0]],
            [core_data0[0]],
            [core_data1[0]],
            ]

        self_flow_data = np.append(self_flow_data, values, axis=1)
        return self_flow_data

    def randomiser(self):
        """ Blitz's the DataBorg dict with random numbers"""
        self.master_stream = random()
        self.mic_in = random()
        self.rnd_poetry = random()
        self.rhythm_rate = randrange(30, 100) / 100

        self.eeg_buffer = np.random.uniform(size=(4, 50))
        self.eda_buffer = np.random.uniform(size=(1, 50))
        self.audio_buffer = np.random.uniform(size=(1, 50))

        self.audio2eda = random()
        self.audio2eda_2d = np.random.uniform(size=(1, 50))

        self.flow2core = random()
        self.flow2core_2d = np.random.uniform(size=(2, 50))

        self.core2flow = random()
        self.core2flow_2d = np.random.uniform(size=(1, 50))

        self.audio2core = random()
        self.audio2core_2d = np.random.uniform(size=(2, 50))

        self.audio2flow = random()
        self.audio2flow_2d = np.random.uniform(size=(1, 50))

        self.flow2audio = random()
        self.flow2audio_2d = np.random.uniform(size=(1, 50))

        self.eda2flow = random()
        self.eda2flow_2d = np.random.uniform(size=(1, 50))
