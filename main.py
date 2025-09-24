import logging
from time import time, sleep
import os
import art
import subprocess
from threading import Thread
from pathlib import Path

import config
from modules.conducter import Conducter
from modules.ai_robot_data_writer import AIRobotDataWriter
from modules.biodata_data_writer import BiodataDataWriter
from nebula.hivemind import DataBorg
from nebula.nebula import Nebula
from modules.randomize_modes import generate_random_modes
# from modules.rami_main import Rami_Main
# from clock import Clock

DATA_LOGGING = config.data_logging
MAIN_PATH = config.path

class Main:
    """
    Main kickstarts the RAMI process. If there is datalogging (Bitalino, Pupil)
    then it will init the connections.
    Essentially, it wil iterate through a series of RAMI experiments, and log
    data for each one separately, while maintaining connection the sensors.
    This is managed through an input question in main_loop.
    """
    def __init__(self):
        # Logging for all modules
        logging.basicConfig(level=logging.WARNING)

        # Build initial dataclass filled with random numbers
        self.hivemind = DataBorg()

        # # Init data logging
        # if DATA_LOGGING:
        #     ###################
        #     # Start Bitalino
        #     ###################
        #     # get relevant libraries
        #     from modules.bitalino import BITalino
        #
        #     # start bitalino
        #     BITALINO_MAC_ADDRESS = config.mac_address
        #     BITALINO_BAUDRATE = config.baudrate
        #     BITALINO_ACQ_CHANNELS = config.channels
        #
        #     eda_started = False
        #     while not eda_started:
        #         try:
        #             self.eda = BITalino(BITALINO_MAC_ADDRESS)
        #             eda_started = True
        #         except OSError:
        #             print("Unable to connect to Bitalino")
        #             retry = input("Retry (y/N)? ")
        #             if retry.lower() != "y":  #  or retry.lower() != "yes":
        #                 eda_started = True
        #
        #     self.eda.start(BITALINO_BAUDRATE, BITALINO_ACQ_CHANNELS)
        #     first_eda_data = self.eda.read(1)[0]
        #     logging.info(f'Data from BITalino = {first_eda_data}')

        # else:
        self.eda = None

        ###################
        # Start Nebula AI
        ###################
        art.tprint("IMPROV2")

        answer = input("Click enter when you are ready to go, after STARTING CLOCK & OPEN SIGNALS")

        # Init the AI factory (inherits AIFactory, Listener)
        self.nebula = Nebula() # eda=self.eda)

    def main_loop(self):
        """
        Manage the experiment loop.
        """
        # while self.hivemind.MASTER_RUNNING:
        random_experiment_list = generate_random_modes()

        print("\nMODES FOR THIS SESSION:")
        for i in random_experiment_list:
            print(f"\t{i}")

        repeat = 1
        for i, experiment_mode in enumerate(random_experiment_list):
            # Init Conducter & Gesture management (controls XArm)
            self.robot = Conducter()

            print(f"=========================================         Running experimental mode:  {repeat} - {experiment_mode}")
            # reset variables
            self.hivemind.MASTER_RUNNING = True
            self.first_time_through = True
            while self.hivemind.MASTER_RUNNING:
                # is this first time through with a new experiment
                # if self.ui.go_flag:
                # make new directory for this log e.g. ../data/20240908_123456
                if DATA_LOGGING:
                    if self.first_time_through:
                        self.master_path = Path(f"{MAIN_PATH}/{self.hivemind.session_date}/WOLFF1_block_{repeat}_performance_{i+1}_mode_{experiment_mode}")
                        self.makenewdir(self.master_path)
                else:
                    self.master_path = None

                # run all systems
                if self.first_time_through:
                    self.wolff1_main(experiment_mode)
                    self.first_time_through = False

                else:
                    sleep(1)
            self.robot.terminate()
            print(f"=========================================         Completed experiment mode  {repeat} - {experiment_mode}.")
            if i < len(random_experiment_list)- 1:
                answer = input("Next Experiment?")
            else:
                print("TERMINATING experiment mode.")
            repeat += 1
            self.first_time_through = True
        answer = input("Next Experiment?")

        # end of experiments so close things down
        self.hivemind.MASTER_RUNNING = False
        self.hivemind.running = False

        # close everything like a grown up
        self.terminate_all()
        print("Close the clock window")

    def terminate_all(self):
        """
        Terminates all active agents and threads
        """
        # if DATA_LOGGING:
        #     self.eda.close()
        self.robot.terminate()
        self.nebula.terminate_listener()

    def wolff1_main(self, experiment_mode):
        """
        Main script to start a single robot arm digital score work.
        Conducter calls the local interpreter for project specific functions. This
        communicates directly to the robot libraries.
        Nebula kick-starts the AI Factory for generating NNet data and affect
        flows.
        This script also controls the live mic audio analyser.
        Paramaters are to be modified in config.py.
        """
        # Init data writer
        if DATA_LOGGING:
            aidw = AIRobotDataWriter(self.master_path)
            # bdw = BiodataDataWriter(self.master_path)

        # Start Nebula AI Factory after conducter starts data moving
        self.nebula.endtime = time() + config.duration_of_piece
        self.hivemind.running = True
        # self.robot = Conducter()
        self.robot.main_loop(experiment_mode)
        self.nebula.main_loop()

        if DATA_LOGGING:
            aidw.main_loop()
            # bdw.main_loop()

    def makenewdir(self, timestamp):
        try:
            # os.mkdir(timestamp)
            timestamp.mkdir(parents=True)
            print(f"Created dir {timestamp}")
        except OSError:
            print(f"OS Make error. Could not make {timestamp}")


if __name__ == "__main__":
    go = Main()
    go.main_loop()

