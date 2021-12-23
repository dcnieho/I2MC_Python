# %%
"""
# I2MC Example
-----
"""

# %%
import os
import numpy as np
import pandas as pd
import regex as re
import matplotlib.pyplot as plt
import I2MC

# %%
"""
## Options
"""

# %%
logging = True

# %%
opt = dict()
# General variables for eye-tracking data
# maximum value of horizontal resolution in pixels
opt['xres'] = 1920.0
opt['yres'] = 1080.0  # maximum value of vertical resolution in pixels
# missing value for horizontal position in eye-tracking data (example data uses -xres). used throughout
# internal_helpers as signal for data loss
opt['missingx'] = -opt['xres']
# missing value for vertical position in eye-tracking data (example data uses -yres). used throughout
# internal_helpers as signal for data loss
opt['missingy'] = -opt['yres']
# sampling frequency of data (check that this value matches with values actually obtained from measurement!)
opt['freq'] = 300.0

# Variables for the calculation of visual angle
# These values are used to calculate noise measures (RMS and BCEA) of
# fixations. The may be left as is, but don't use the noise measures then.
# If either or both are empty, the noise measures are provided in pixels
# instead of degrees.
opt['scrSz'] = [50.9174, 28.6411]  # screen size in cm
opt['disttoscreen'] = 65.0  # distance to screen in cm.

# %%
"""
### Optional Options
"""

# %%
# STEFFEN INTERPOLATION
# max duration (s) of missing values for interpolation to occur
opt['windowtimeInterp'] = 0.1
# amount of data (number of samples) at edges needed for interpolation
opt['edgeSampInterp'] = 2
# maximum displacement during missing for interpolation to be possible
opt['maxdisp'] = opt['xres'] * 0.2 * np.sqrt(2)

# # K-MEANS CLUSTERING
# time window (s) over which to calculate 2-means clustering (choose value so that max. 1 saccade can occur)
opt['windowtime'] = 0.2
# time window shift (s) for each iteration. Use zero for sample by sample processing
opt['steptime'] = 0.02
# maximum number of errors allowed in k-means clustering procedure before proceeding to next file
opt['maxerrors'] = 100
opt['downsamples'] = [2.0, 5.0, 10.0]
# use chebychev filter when down sampling? 1: yes, 0: no. requires signal processing toolbox. is what matlab's
# down sampling internal_helpers do, but could cause trouble (ringing) with the hard edges in eye-movement data
opt['downsampFilter'] = 0

# # FIXATION DETERMINATION
# number of standard deviations above mean k-means weights will be used as fixation cutoff
opt['cutoffstd'] = 2.0
# number of MAD away from median fixation duration. Will be used to walk forward at fixation starts and backward at
# fixation ends to refine their placement and stop algorithm from eating into saccades
opt['onoffsetThresh'] = 3.0
# maximum Euclidean distance in pixels between fixations for merging
opt['maxMergeDist'] = 30.0
# maximum time in ms between fixations for merging
opt['maxMergeTime'] = 30.0
# minimum fixation duration after merging, fixations with shorter duration are removed from output
opt['minFixDur'] = 40.0

# %%
"""
## Folders
"""

# %%
folders = dict()
# folder in which data is stored (each folder in folders.data is considered 1 subject)
folders['data'] = './example data'
# folder for output (will use structure in folders.data for saving output)
folders['output'] = './output'

# Check if output directory exists, if not create it
if not os.path.isdir(folders['output']):
    os.mkdir(folders['output'])
fold = list(os.walk(folders['data']))
all_folders = [f[0] for f in fold[1:]]
number_of_folders = len(all_folders)

# Get all files
all_files = [f[2] for f in fold[1:]]
number_of_files = [len(f) for f in all_files]


# Write the final fixation output file
df_fixation = pd.DataFrame(columns=['FixStart', 'FixEnd', 'FixDur', 'XPos', 'YPos',
                                    'FlankedByDataLoss', 'Fraction Interpolated', 'WeightCutoff', 'RMSxy', 'BCEA',
                                    'FixRangeX', 'FixRangeY', 'Participant', 'Trial'])

# %%
"""
## Start The Algorithm
"""

# %%
for folder_idx, folder in enumerate(all_folders):
    if logging:
        print('Processing folder {} of {}'.format(folder_idx + 1, number_of_folders))

    for file_idx, file in enumerate(all_files[folder_idx]):
        if logging:
            print('\tProcessing file {} of {}'.format(file_idx + 1, number_of_files[folder_idx]))

        # get the current file name
        file_name = os.path.join(folder, file)
        # load the data
        if logging:
            print('\t\tLoading data...')

        df_eyetracking_data = pd.read_csv(file_name, sep='\t', header=None)
        # keep colum 7, 8, 13, 20, 21, 26, 27
        df_eyetracking_data = df_eyetracking_data.iloc[:, [7, 8, 13, 20, 21, 26, 27]]
        # set first row to be the column names of the dataframe
        df_eyetracking_data.columns = df_eyetracking_data.iloc[0]
        # remove the first row
        df_eyetracking_data = df_eyetracking_data.iloc[1:]
        # reset the index
        df_eyetracking_data = df_eyetracking_data.reset_index(drop=True)
        # update the columns to display coordinates in pixels
        df_eyetracking_data["LGazePos2dx"] = df_eyetracking_data["LGazePos2dx"].astype(float) * opt['xres']
        df_eyetracking_data["LGazePos2dy"] = df_eyetracking_data["LGazePos2dy"].astype(float) * opt['yres']
        df_eyetracking_data["RGazePos2dx"] = df_eyetracking_data["RGazePos2dx"].astype(float) * opt['xres']
        df_eyetracking_data["RGazePos2dy"] = df_eyetracking_data["RGazePos2dy"].astype(float) * opt['yres']
        df_eyetracking_data["LValidity"] = df_eyetracking_data["LValidity"].astype(int)
        df_eyetracking_data["RValidity"] = df_eyetracking_data["RValidity"].astype(int)
        df_eyetracking_data["RelTimestamp"] = df_eyetracking_data["RelTimestamp"].astype(float)

        # sometimes we have weird peaks where one sample is (very) far outside the
        # monitor. Here, count as missing any data that is more than one monitor
        # distance outside the monitor.

        df_eyetracking_data["l_miss_x"] = df_eyetracking_data.apply(lambda row: row["LGazePos2dx"] < -opt['xres'] or row['LGazePos2dx'] >= 2*opt["xres"], axis=1)
        df_eyetracking_data["l_miss_y"] = df_eyetracking_data.apply(lambda row: row["LGazePos2dy"] < -opt['yres'] or row['LGazePos2dy'] >= 2*opt["yres"], axis=1)
        df_eyetracking_data["r_miss_x"] = df_eyetracking_data.apply(lambda row: row["RGazePos2dx"] < -opt['xres'] or row['RGazePos2dx'] >= 2*opt["xres"], axis=1)
        df_eyetracking_data["r_miss_y"] = df_eyetracking_data.apply(lambda row: row["RGazePos2dy"] < -opt['yres'] or row['RGazePos2dy'] >= 2*opt["yres"], axis=1)

        df_eyetracking_data["l_miss"] = df_eyetracking_data.apply(lambda row: row["l_miss_x"] or row["l_miss_y"] or row["LValidity"] > 1, axis=1)
        df_eyetracking_data["r_miss"] = df_eyetracking_data.apply(lambda row: row["r_miss_x"] or row["r_miss_y"] or row["RValidity"] > 1, axis=1)

        df_eyetracking_data.loc[df_eyetracking_data["l_miss"], "LGazePos2dx"] = opt["missingx"]
        df_eyetracking_data.loc[df_eyetracking_data["l_miss"], "LGazePos2dy"] = opt["missingy"]
        df_eyetracking_data.loc[df_eyetracking_data["r_miss"], "RGazePos2dx"] = opt["missingx"]
        df_eyetracking_data.loc[df_eyetracking_data["r_miss"], "RGazePos2dy"] = opt["missingy"]

        # drop miss columns
        df_eyetracking_data = df_eyetracking_data.drop(columns=["l_miss_x", "l_miss_y", "r_miss_x", "r_miss_y", "l_miss", "r_miss"])

        # rename columns
        df_eyetracking_data.rename(columns={'LGazePos2dx': 'L_X', 'LGazePos2dy': 'L_Y', 'RGazePos2dx': 'R_X', 'RGazePos2dy': 'R_Y', 'RelTimestamp': 'time'}, inplace=True)

        if len(df_eyetracking_data) == 0:
            if logging:
                print('\t\tNo data found in file {}'.format(file_name))
            continue

        # run fixation detection
        if logging:
            print('\t\tRunning fixation detection...')
        try:
            fix, data, par = I2MC.I2MC(df_eyetracking_data, opt, logging_offset="\t\t\t")
        except Exception as e:
            print('\t\tError in file {}: {}'.format(file_name, e))
            continue

        if not fix:
            if logging:
                print('\t\tFixation calculation had some Problem with file {}'.format(file_name))
            continue


        # save the data
        for t in range(len(fix['start'])):
            df_tmp = pd.DataFrame([[fix['startT'][t], fix['endT'][t], fix['dur'][t], fix['xpos'][t], fix['ypos'][t],
                                    fix['flankdataloss'][t], fix['fracinterped'][t],
                                    fix['cutoff'], fix['RMSxy'][t], fix['BCEA'][t],
                                    fix['fixRangeX'][t], fix['fixRangeY'][t],
                                    re.findall(r'\d+', folder)[-1], file.split(".")[0]]], columns=df_fixation.columns)
            df_fixation = df_fixation.append(df_tmp, ignore_index=True)

        # plot the data
        outFold = folders['output'] + os.sep + (folder.split(os.sep)[-1])
        if not os.path.exists(outFold):
            os.makedirs(outFold)
        saveFile = outFold + os.sep + os.path.splitext(file)[0] + '.png'
        f = I2MC.plot.plot_i2mc(df_eyetracking_data, fix, [opt['xres'], opt['yres']])
        # save figure and close
        print('\t\tSaving image to: ' + saveFile)
        f.savefig(saveFile)
        plt.close(f)
        break
    break

# %%
