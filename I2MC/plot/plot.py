import math

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
import numpy as np


def plot_i2mc(data, fix, res=None):
    """
    Plots the results of the I2MC function
    """

    if res is None:
        res = [1920, 1080]

    time = data['time'].array
    Xdat = np.array([])
    Ydat = np.array([])
    klr = []
    if 'L_X' in data.keys():
        Xdat = data['L_X'].array
        Ydat = data['L_Y'].array
        klr.append('g')
    if 'R_X' in data.keys():
        if len(Xdat) == 0:
            Xdat = data['R_X'].array
            Ydat = data['R_Y'].array
        else:
            Xdat = np.vstack([Xdat, data['R_X'].array])
            Ydat = np.vstack([Ydat, data['R_Y'].array])
        klr.append('r')
    if 'average_X' in data.keys() and 'L_X' not in data.keys() and 'R_X' not in data.keys():
        if len(Xdat) == 0:
            Xdat = data['average_X'].array
            Ydat = data['average_Y'].array
        else:
            Xdat = np.vstack([Xdat, data['average_X'].array])
            Ydat = np.vstack([Ydat, data['average_Y'].array])
        klr.append('b')

        # Plot settings
    myfontsize = 10
    myLabelSize = 12
    traceLW = 0.5
    fixLWax1 = res[0] / 100
    fixLWax2 = res[1] / 100

    font = {'size': myfontsize}
    matplotlib.rc('font', **font)

    # plot layout
    f = plt.figure(figsize=(10, 6), dpi=300)
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_ylabel('Horizontal position (pixels)', size=myLabelSize)
    ax1.set_xlim([0, time[-1]])
    ax1.set_ylim([0, res[0]])

    # Plot x position
    if len(Xdat.shape) > 1:
        for p in range(Xdat.shape[0]):
            ax1.plot(time, Xdat[p, :], klr[p] + '-', linewidth=traceLW)
    else:
        ax1.plot(time, Xdat, klr[0] + '-', linewidth=traceLW)

    # Plot Y posiiton
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Vertical position (pixels)', size=myLabelSize)
    ax2.set_ylim([0, res[1]])
    ax2.invert_yaxis()
    if len(Xdat.shape) > 1:
        for p in range(Ydat.shape[0]):
            ax2.plot(time, Ydat[p, :], klr[p] + '-', linewidth=traceLW)
    else:
        ax2.plot(time, Ydat, klr[0] + '-', linewidth=traceLW)

    # add fixations, but adds a shaded area instead of line
    for b in range(len(fix['startT'])):
        ax1.add_patch(patches.Rectangle((fix['startT'][b], fix['xpos'][b] - (fixLWax1 / 2)),
                                        fix['endT'][b] - fix['startT'][b],
                                        abs(fixLWax1), fill=True, alpha=0.8, color='k',
                                        linewidth=0, zorder=3))
        ax2.add_patch(patches.Rectangle((fix['startT'][b], fix['ypos'][b] - (fixLWax2 / 2)),
                                        fix['endT'][b] - fix['startT'][b],
                                        abs(fixLWax2), fill=True, alpha=0.8, color='k',
                                        linewidth=0, zorder=3))

    return f
