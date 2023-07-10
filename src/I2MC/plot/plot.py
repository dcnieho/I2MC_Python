# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:57:23 2019

@author: Jonathan van Leeuwen, Diederick Niehorster
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

def data_and_fixations(data, fix, fix_as_line=True, unit='pixels', res=None):
    """
    Plots the results of the I2MC function
    fix_as_line: if true, fixations are drawn as lines, if
    false as shaded areas
    If the res parameter (screen resolution) is not provided,
    a 1920x1080 pix screen is assumed
    """

    if res is None:
        res = [1920, 1080]

    if isinstance(data,dict):
        # for backward compatibility, convert to pd.DataFrame
        data = pd.DataFrame.from_dict(data)

    time = data['time'].array
    Xdat = np.array([])
    Ydat = np.array([])
    klr  = []
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

    font = {'size': myfontsize}
    matplotlib.rc('font', **font)

    ## plot layout
    f   = plt.figure(figsize=(10, 6), dpi=300)
    ax1 = plt.subplot(2,1,1)
    ax1.set_ylabel('Horizontal position ({})'.format(unit), size = myLabelSize)
    ax1.set_xlim([time[1], time[-1]])
    if isinstance(res[0],list):
        ax1.set_ylim([res[0][0], res[0][1]])
    else:
        ax1.set_ylim([0, res[0]])

    ax2 = plt.subplot(2,1,2, sharex=ax1)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Vertical position ({})'.format(unit), size = myLabelSize)
    if isinstance(res[1],list):
        ax2.set_ylim([res[1][0], res[1][1]])
    else:
        ax2.set_ylim([0, res[1]])

    ### Plot X position
    Xdatp = np.atleast_2d(Xdat)
    for p in range(Xdatp.shape[0]):
        ax1.plot(np.array(time),Xdatp[p,:],klr[p]+'-', linewidth = traceLW)

    ### Plot Y posiiton
    Ydatp = np.atleast_2d(Ydat)
    for p in range(Ydatp.shape[0]):
        ax2.plot(np.array(time),Ydatp[p,:],klr[p]+'-', linewidth = traceLW)

    # add fixations
    if fix_as_line:
        fixLW = 2
        for b in range(len(fix['startT'])):
            ax1.plot([fix['startT'][b], fix['endT'][b]], [fix['xpos'][b], fix['xpos'][b]],'k-', linewidth = fixLW)
        for b in range(len(fix['startT'])):
            ax2.plot([fix['startT'][b], fix['endT'][b]], [fix['ypos'][b], fix['ypos'][b]],'k-', linewidth = fixLW)
    else:
        for b in range(len(fix['startT'])):
            ax1.add_patch(patches.Rectangle((fix['startT'][b], 0),
                                            fix['endT'][b] - fix['startT'][b],
                                            res[0], fill=True, color='0.7',
                                            linewidth=0))
            ax2.add_patch(patches.Rectangle((fix['startT'][b], 0),
                                            fix['endT'][b] - fix['startT'][b],
                                            res[1], fill=True, color='0.7',
                                            linewidth=0))

    return f

