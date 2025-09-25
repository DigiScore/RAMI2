import logging

import pandas as pd
import matplotlib.pyplot as plt
import config
import json

# example of data
# {"date": "2024-11-21T12:22:33.275471", "master_stream": " ", "mic_in": 4.482421875e-05, "rnd_poetry": 0.3010769531419044, "flow2core": 0.4151631397097578, "core2flow": 0.6335565511568154, "audio2core": 0.21971498619599966, "audio2flow": 0.6035721781171628, "flow2audio": 3.6552383224169296e-10, "eda2flow": 0.9998352258886954, "design decision": " ", "interrupt": false, "x": 0.423767, "y": 0.591036, "z": 0.28214708994709},

class AI_visualiser:

    def __init__(self, raw_file_path, ai_robot_images_path):
        # variables
        self.ai_robot_images_path = ai_robot_images_path
        self.figsize_xy = config.figsize_xy

        # load the data from the file
        df = pd.DataFrame(json.loads(open(raw_file_path).read()))

        # each row of x and y is a list; explode the values in the lists to separate rows
        df = df.explode(["date", "master_stream", "mic_in", "rnd_poetry", "audio2eda", "flow2core", "core2flow",
                         "audio2core", "audio2flow", "flow2audio", "eda2flow",
                         "design decision", "interrupt", "x", "y", "z"], ignore_index=True)

        # get date and time
        df["date"] = pd.to_datetime(df["date"])
        date = df["date"]

        # make master plot
        fig, ax = plt.subplots(6, figsize=self.figsize_xy)

        # plot audio
        ax[0].plot(date, df["mic_in"], label="mic in")
        ax[0].set_title("Human Audio")
        ax[0].set_ylabel("Amplitude")
        ax[0].set_xlabel("Time")

        # plot AI factory
        ax[1].set_title("AI Factory")
        ax[1].set_ylabel("Amplitude")
        ax[1].set_xlabel("Time")
        # ax[1].plot(date, df["rnd_poetry"])
        ax[1].plot(date, df["audio2eda"], label="audio2eda")
        ax[1].plot(date, df["flow2core"], label="flow2core")
        ax[1].plot(date, df["core2flow"], label="core2flow")
        ax[1].plot(date, df["audio2core"], label="audio2core")
        ax[1].plot(date, df["audio2flow"], label="audio2flow")
        ax[1].plot(date, df["flow2audio"], label="flow2audio")
        ax[1].plot(date, df["eda2flow"], label="eda2flow")
        ax[1].legend(shadow=True, fancybox=True)

        # plot AI gesture manager/ focus
        ax[2].set_title("Gesture Manager/Focus")
        ax[2].set_ylabel("Attention Focus")
        ax[2].set_xlabel("Time")
        ax[2].plot(date, df["master_stream"])

        # plot Interrupt
        ax[3].set_title("Interrupt")
        ax[3].set_ylabel("True/ False")
        ax[3].set_xlabel("Time")
        ax[3].plot(date, df["interrupt"])

        # plot robot xyz
        ax[4].set_title("Robot xyz")
        ax[4].set_ylabel("Amplitude")
        ax[4].set_xlabel("Time")
        ax[4].plot(date, df["x"], label="x")
        ax[4].plot(date, df["y"], label="y")
        ax[4].plot(date, df["z"], label="z")
        ax[4].legend(shadow=True, fancybox=True)

        # plot design decisions
        ax[5].set_title("Design Decisions")
        ax[5].set_ylabel("Movement")
        ax[5].set_xlabel("Time")
        ax[5].plot(date, df["interrupt"])

        plt.savefig(f"{self.ai_robot_images_path}/ai_plot")

if __name__ == "__main__":
    test = AI_visualiser('../data/2025_09_25_0957/IMPROV2_block_1_performance_1_mode_AI/ai_robot/AI_Robot_2025_09_25_0957.json',
                         '../data/2025_09_25_0957/IMPROV2_block_1_performance_1_mode_AI/ai_robot/images')
