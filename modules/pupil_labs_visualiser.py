# Script for processing of Pupil Player output
# 
# Laura Bishop, laura.bishop@imv.uio.no
#


import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import pandas as pd
import matplotlib.pyplot as plt
import config


class PupilLabsVisualiser:

    def __init__(self, pupillabs_path):

        # Change this to your local working directory
        # os.chdir("/Volumes/Current/Active_projects/RAMI/Data/Proof-of-concept/PupilLabs")

        # Subfolder within working directory containing Pupil Lab csv exports
        self.path_data = f"{pupillabs_path}/raw"

        # Subfolder to receive exported plots
        self.path_plots = f"{pupillabs_path}/images"

        # Subfolder to receive exported filtered data
        self.path_filtered = f"{pupillabs_path}/filtered"

        # These parameters are used for the filtering function & can be changed if needed
        self.choose_resample = True  # Do you want to downsample to 60 Hz? True or False
        self.outlier_value = 2  # Used for defining outlier velocities
        self.low_threshold = 2  # Used for cutting off values that are too far below trial mean
        self.high_threshold = 3  # Used for cutting off values that are too far above trial mean
        self.sg_w = 19  # Used in Savitzky-Golay filter (window size); reduce for less smoothing
        self.s_omit = 0
        self.e_omit = 100000
        self.figsize_xy = config.figsize_xy

    def filtering(self, eye_data):
        # Resample down to 60 Hz using a linear interpolation
        if self.choose_resample == "T":
            trial = pd.DataFrame({
                'newtime': np.arange(eye_data['newtime'].min(), eye_data['newtime'].max(), 1/60)
            })
            interp_func = interp1d(eye_data['newtime'], eye_data['diameter_3d'], bounds_error=False)
            trial['diameter_3d'] = interp_func(trial['newtime'])
        else:
            trial = eye_data.copy()

        # Calculate cutoff velocity value
        trial['xmm_vel'] = np.append([np.nan], np.diff(trial['diameter_3d']) / np.append([np.nan], np.diff(trial['newtime'])))
        out_bound = self.outlier_value * np.nanstd(trial['xmm_vel'])

        # Create new variable & omit instances where diameter = 0
        trial['xmm_p'] = trial['diameter_3d']
        trial['xmm_p'] = np.where(trial['xmm_p'] < 0.5, np.nan, trial['xmm_p'])

        # Create new variable & omit extreme velocities
        trial['xmm_pp'] = np.where(np.abs(trial['xmm_vel']) > out_bound, np.nan, trial['xmm_p'])

        # Calculate lowest allowed value & remove values below this threshold
        low_bound = np.nanmean(trial['xmm_pp']) - self.low_threshold * np.nanstd(trial['xmm_pp'])
        trial['xmm_ppp'] = np.where(trial['xmm_pp'] < low_bound, np.nan, trial['xmm_pp'])

        # Calculate highest allowed value & remove values above this threshold
        high_bound = np.nanmean(trial['xmm_pp']) + self.high_threshold * np.nanstd(trial['xmm_pp'])
        trial['xmm_ppp'] = np.where(trial['xmm_pp'] > high_bound, np.nan, trial['xmm_ppp'])

        # Savitzky-Golay filter
        trial['xmm_sg'] = np.nan
        trial['xmm_sg'] = savgol_filter(trial['xmm_ppp'], self.sg_w, 3, mode='interp')

        # Interpolation
        interp_func = interp1d(trial['newtime'], trial['xmm_sg'], bounds_error=False)
        trial['xmm_int'] = interp_func(trial['newtime'])

        return trial

    def main(self):
        #########################
        # Load data
        #########################

        # List the files that are currently in path.data and choose one to process -----------
        pupils_list = pd.DataFrame(list(filter(lambda x: "pupil_positions.csv" in x,
                                                [os.path.join(root, file) for root, dirs, files in os.walk(self.path_data) for file in files])))
        pupils_list.columns = ["files"]
        # print(pupils_list)

        test_file_name = pupils_list.files[1]  # Choose a csv file from your list to process
        file_name = "JB-Imp1"

        #########################
        # Process data
        #########################

        # Uncomment lines below & lines 163-165 if you want to loop through files in pupils.list
        # all_output = pd.DataFrame({'eye0': [0], 'eye1': [0], 'file': [None]})
        # for test_file_name in pupils_list['files']:

        player_output = pd.read_csv(f"{self.path_data}{test_file_name}", sep=",")  # Load csv file
        # print(player_output.head())  # Check that the data look ok...
        player_3d = player_output[player_output['method'] != "2d c++"]

        eye0 = player_3d[player_3d['eye_id'] == 0]  # Isolate data for each eye
        eye1 = player_3d[player_3d['eye_id'] == 1]

        # Left eye (eye = 0)
        eye0_ts = eye0.groupby('pupil_timestamp')['diameter_3d'].mean().reset_index()  # Collapse over timestamp values
        eye0_ts['newtime'] = (eye0_ts['pupil_timestamp'] - eye0_ts['pupil_timestamp'].iloc[0]).round(3)  # Create new timestamp starting at 0
        eye0_conf = eye0.groupby('pupil_timestamp')['confidence'].mean().reset_index()
        eye0_ts['confidence'] = eye0_conf['confidence']

        # Right eye (eye = 1)
        eye1_ts = eye1.groupby('pupil_timestamp')['diameter_3d'].mean().reset_index()  # Collapse over timestamp values
        eye1_ts['newtime'] = (eye1_ts['pupil_timestamp'] - eye1_ts['pupil_timestamp'].iloc[0]).round(3)  # Create new timestamp starting at 0
        eye1_conf = eye1.groupby('pupil_timestamp')['confidence'].mean().reset_index()
        eye1_ts['confidence'] = eye1_conf['confidence']

        # Omit starts and ends of trials
        eye0_true = eye0_ts[(eye0_ts['newtime'] > self.s_omit) & (eye0_ts['newtime'] < self.e_omit)]
        eye1_true = eye1_ts[(eye1_ts['newtime'] > self.s_omit) & (eye1_ts['newtime'] < self.e_omit)]

        # Create preliminary plots of raw data. Black lines are raw data; red lines are raw data only where
        # Pupil Labs' measured confidence is > .6
        plt.figure(figsize=self.figsize_xy)
        plt.subplot(2, 1, 1)
        plt.plot(eye0_true['newtime'], eye0_true['diameter_3d'], label='Raw Data')
        plt.plot(eye0_true[eye0_true['confidence'] > 0.6]['newtime'],
                 eye0_true[eye0_true['confidence'] > 0.6]['diameter_3d'], color='red', label='High Confidence')
        plt.title("Eye 0")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(eye1_true['newtime'], eye1_true['diameter_3d'], label='Raw Data')
        plt.plot(eye1_true[eye1_true['confidence'] > 0.6]['newtime'],
                 eye1_true[eye1_true['confidence'] > 0.6]['diameter_3d'], color='red', label='High Confidence')
        plt.title("Eye 1")
        plt.legend()
        plt.savefig(f"{self.path_plots}{file_name}-confidence.pdf")
        plt.close()

        # Run filtering function
        filtered_eye0 = self.filtering(eye0_true)
        filtered_eye1 = self.filtering(eye1_true)

        # Write filtered data to file
        filtered_eye0.to_csv(f"{self.path_filtered}{file_name}-Eye0.txt", index=False, sep='\t')
        filtered_eye1.to_csv(f"{self.path_filtered}{file_name}-Eye1.txt", index=False, sep='\t')

        ### Plot raw data + filtered data
        plt.figure(figsize=self.figsize_xy)
        plt.subplot(2, 1, 1)
        plt.plot(filtered_eye0['newtime'], filtered_eye0['diameter_3d'], label='Raw Data')
        plt.plot(filtered_eye0['newtime'], filtered_eye0['xmm_int'], color='red', label='Filtered Data')
        plt.title("Eye 0")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(filtered_eye1['newtime'], filtered_eye1['diameter_3d'], label='Raw Data')
        plt.plot(filtered_eye1['newtime'], filtered_eye1['xmm_int'], color='red', label='Filtered Data')
        plt.title("Eye 1")
        plt.legend()
        plt.savefig(f"{self.path_plots}{file_name}-filtered.pdf")
        plt.close()

        ### If you want to zoom in on short intervals & see the plots in the console, use code below
        ### (shown here for eye 0 and random start/stop times)
        start = 100  # Start of interval in seconds
        end = 120  # End of interval in seconds
        sub_filtered_eye0 = filtered_eye0[(filtered_eye0['newtime'] >= start) &
                                           (filtered_eye0['newtime'] < end)]
        plt.plot(sub_filtered_eye0['newtime'], sub_filtered_eye0['diameter_3d'], label='Raw Data')
        plt.plot(sub_filtered_eye0['newtime'], sub_filtered_eye0['xmm_int'], color='red', label='Filtered Data')
        plt.title("Eye 0")
        plt.legend()
        plt.show()

        # }
        # all_output = all_output.iloc[1:, :]
        # all_output


        # Calculate an average pupil size per 1 seconds
        filtered_eye0['rtime'] = filtered_eye0['newtime'].round()
        ag = filtered_eye0.groupby('rtime')['xmm_int'].median().reset_index()

        plt.plot(filtered_eye0['newtime'], filtered_eye0['xmm_int'], linestyle='-')
        plt.plot(ag['rtime'], ag['xmm_int'], color='red', linewidth=2)
        plt.plot(ag['rtime'], ag['xmm_int'], color='red', linewidth=2)
        plt.show()
