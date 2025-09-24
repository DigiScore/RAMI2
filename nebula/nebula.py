"""
Embodied AI Engine Prototype AKA "Nebula". This object takes a live signal
(such as body tracking, or real-time sound analysis) and generates a response
that aims to be felt as co-creative. The response is a flow of neural network
emissions data packaged as a dictionary, and is gestural over time. This, when
plugged into a responding script (such as a sound generator, or QT graphics)
gives the feeling of the AI creating in-the-moment with the human in-the-loop.

© Craig Vear 2022-24
craig.vear@nottingham.ac.uk

Dedicated to Fabrizio Poltronieri
"""
import logging
import numpy as np
import warnings
from scipy import signal
from threading import Thread
from time import sleep, time

import config
from modules.listener import Listener
from nebula.ai_factory import AIFactoryRAMI


def scaler(in_feature, mins, maxs):
    """
    Min-max scaler with clipping.
    """
    warnings.filterwarnings('error')
    in_feature = np.array(in_feature)
    mins = np.array(mins)
    maxs = np.array(maxs)
    try:
        norm_feature = (in_feature - mins) / (maxs - mins)
    except RuntimeWarning:
        logging.info("Scaler encountered zero division")
        norm_feature = in_feature
    norm_feature = norm_feature.clip(0, 1)
    warnings.simplefilter("always")
    return norm_feature


class Nebula(Listener, AIFactoryRAMI):
    """
    Nebula is the core "director" of an AI factory. It generates data in
    response to incoming percepts from human-in-the-loop interactions, and
    responds in-the-moment to the gestural input of live data.
    There are 4 components:
        - Nebula: as "director" it coordinates the overall operations of the AI
        Factory.
        - AIFactory: builds the neural nets that form the factory, coordinates
        data exchange, and liases with the common data dict.
        - Hivemind: is the central dataclass that holds and shares all the data
        exchanges in the AI factory.
        - Conducter: receives the live percept input from the client and
        produces an affectual response to it's energy input, which in turn
        interferes with the data generation.
    """
    def __init__(self): #, eda):  #, speed=1):
        """
        Parameters
        ----------
        speed
            General tempo/ feel of Nebula's response (0.5 ~ moderate fast,
            1 ~ moderato, 2 ~ presto).
        """
        print('Building engine server')
        Listener.__init__(self)

        # Set global vars
        # self.hivemind.running = True

        # Build the AI factory and pass it the data dict
        AIFactoryRAMI.__init__(self)  #, speed)

        # Work out master timing then collapse hivemind.running
        self.endtime = 0

    def main_loop(self):
        """
        Starts the server / AI threads and gets the data rolling.
        """
        print('Starting the Nebula director')
        # Declare all threads
        t1 = Thread(target=self.make_data)
        t2 = Thread(target=self.snd_listen)
        t3 = Thread(target=self.human_input)

        # Start them all
        t1.start()
        t2.start()
        t3.start()

    def human_input(self):
        """
        Listen to live human input.
        """
        while self.hivemind.running:
            if time() >= self.endtime:
                break

            # TODO: connect to the new neural network
            self.hivemind.eda_buffer = np.random.uniform(size=(1, 50))

            sleep(0.01)  # for 100 Hz

        self.hivemind.running = False
        # self.hivemind.MASTER_RUNNING = False
