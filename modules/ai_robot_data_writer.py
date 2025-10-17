import json
import logging
import os
from datetime import datetime
from threading import Thread
from time import sleep
from pathlib import Path

from nebula.hivemind import DataBorg
from modules.ai_robot_visualiser import AI_visualiser
import config


class AIRobotDataWriter:

    def __init__(self, master_path):
        self.hivemind = DataBorg()

        # make all dirs for data logging
        self.ai_robot_path = Path(f"{master_path}/ai_robot")
        self.makenewdir(self.ai_robot_path)

        self.ai_robot_images = Path(f"{self.ai_robot_path}/images")
        self.makenewdir(self.ai_robot_images)

        self.hivemind = DataBorg()
        self.samplerate = config.samplerate

        self.data_file_path = Path(f"{self.ai_robot_path}/AI_Robot_{self.hivemind.session_date}.json")
        self.data_file = open(self.data_file_path, "a")
        self.data_file.write("[")


    def json_update(self):
        """
        Write a hiveming tic in the json file.
        """
        json_dict = {
            "date": datetime.now().isoformat(),
            "master_stream": self.hivemind.thought_train_stream,
            "mic_in": self.hivemind.mic_in,
            "rnd_poetry": self.hivemind.rnd_poetry,
            "audio2eda": self.hivemind.audio2eda,
            "flow2core": self.hivemind.flow2core,
            "core2flow": self.hivemind.core2flow,
            "audio2core": self.hivemind.audio2core,
            "audio2flow": self.hivemind.audio2flow,
            "flow2audio": self.hivemind.flow2audio,
            "eda2flow": self.hivemind.eda2flow,
            "design decision": self.hivemind.design_decision,
            "interrupt": self.hivemind.interrupted,
            "x": self.hivemind.current_robot_x_y_z[0],
            "y": self.hivemind.current_robot_x_y_z[1],
            "z": self.hivemind.current_robot_x_y_z[2],
        }
        json_object = json.dumps(json_dict)
        self.data_file.write(json_object)
        self.data_file.write(',\n')

    def terminate_data_writer(self):
        """
        Terminate the json writer and close file.
        """
        self.data_file.seek(self.data_file.tell() - 3, os.SEEK_SET)
        self.data_file.truncate()  # remove ",\n"
        self.data_file.write("]")
        self.data_file.close()
        # sleep(3)
        # self.process_and_plot()

    def main_loop(self):
        """
        Start the main thread for the writing manager.
        """
        writer_thread = Thread(target=self.writing_manager)
        writer_thread.start()

    def writing_manager(self):
        """
        Write realtime data from hivemind.
        """
        while self.hivemind.running:
            self.json_update()
            sleep(self.samplerate)
        logging.info("quitting data writer thread")
        self.terminate_data_writer()

    def process_and_plot(self):
        AI_visualiser(raw_file_path=self.data_file_path,
                      ai_robot_images_path=self.ai_robot_images)

    def makenewdir(self, path):
        try:
           path.mkdir(parents=True)
           # os.makedirs(path)
        except OSError:
            print(f"Path Error - unable to create Directory {path}")
