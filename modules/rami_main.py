import art
import time

import config
from modules.conducter import Conducter
from modules.ai_robot_data_writer import AIRobotDataWriter
from modules.biodata_data_writer import BiodataDataWriter
from nebula.hivemind import DataBorg
from nebula.nebula import Nebula



class Rami_Main:
    """
    Main script to start a single robot arm digital score work.
    Conducter calls the local interpreter for project specific functions. This
    communicates directly to the robot libraries.
    Nebula kick-starts the AI Factory for generating NNet data and affect
    flows.
    This script also controls the live mic audio analyser.
    Paramaters are to be modified in config.py.
    """
    def __init__(self, eda, master_path):
        art.tprint("RAMI")
        # Build initial dataclass filled with random numbers
        self.hivemind = DataBorg()

        # Init the AI factory (inherits AIFactory, Listener)
        nebula = Nebula(eda=eda)  #, speed=config.speed)

        # Init Conducter & Gesture management (controls XArm)
        robot = Conducter(speed=config.speed)

        # Init data writer
        if config.data_writer:
            aidw = AIRobotDataWriter(master_path)
            bdw = BiodataDataWriter(master_path)

        # Start Nebula AI Factory after conducter starts data moving
        nebula.endtime = time.time() + config.duration_of_piece
        self.hivemind.running = True
        robot.main_loop()
        nebula.main_loop()

        if config.data_writer:
            aidw.main_loop()
            bdw.main_loop()


# if __name__ == "__main__":
#     Rami_Main()
