# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:54:00 2019

@author: Jonathan van Leeuwen, Diederick Niehorster
"""

import pandas as pd
import numpy as np
import math
import scipy
import scipy.signal
from scipy.cluster.vq import vq, _vq
from scipy.spatial.distance import cdist
import copy
import warnings

# =============================================================================
# Helper functions
# =============================================================================
def is_number(s):
    try:
        np.array(s, dtype=float)
        return True
    except ValueError:
        return False

def check_numeric(k, v):
    if not is_number(v):
        raise ValueError('The value of "{}" is invalid. Expected input to be a number. Instead its type was {}.'.format(k, type(v)))

def check_scalar(k, v):
    if not np.ndim(v) == 0:
        raise ValueError('The value of "{}" is invalid. Expected input to be a scalar.'.format(k))

def check_vector_2(k, v):
    if not np.size(v) == 2:
        raise ValueError('The value of "{}" is invalid. Expected input to be a 2-element array.'.format(k))

def check_int(k, v):
    if np.sum(np.array(v) % 1) != 0:
        raise ValueError('The value of "{}" is invalid. Expected input to be an integer or list of integers.'.format(k))

def check_fun(k, d, s):
    if k not in d.keys():
        raise ValueError('I2MCfunc: "{}" must be specified using the "{}" option key, but it cannot be found'.format(s, k))
    if not is_number(d[k]):
        raise ValueError('I2MCfunc: "{}" must be set as a number using the "{}" option'.format(s, k))

def angle_to_pixels(angle, screenDist, screenW, screenXY):
    """
    Calculate the number of pixels which equals a specified angle in visual
    degrees, given parameters. Calculates the pixels based on the width of
    the screen. If the pixels are not square, a separate conversion needs
    to be done with the height of the screen.\n
    "angleToPixelsWH" returns pixels for width and height.

    Parameters
    ----------
    angle : float or int
        The angle to convert in visual degrees
    screenDist : float or int
        Viewing distance in cm
    screenW : float or int
        The width of the screen in cm
    screenXY : tuple, ints
        The resolution of the screen (width - x, height - y), pixels

    Returns
    -------
    pix : float
        The number of pixels which corresponds to the visual degree in angle,
        horizontally

    Examples
    --------
    >>> pix = angleToPixels(1, 75, 47.5, (1920,1080))
    >>> pix
    52.912377341863817
    """
    pixSize = screenW / float(screenXY[0])
    angle = np.radians(angle / 2.0)
    cmOnScreen = np.tan(angle) * float(screenDist)
    pix = (cmOnScreen / pixSize) * 2

    return pix

def get_missing(L_X, R_X, missing_x, L_Y, R_Y, missing_y):
    """
    Gets missing data and returns missing data for left, right and average

    Parameters
    ----------
    L_X : np.array
        Left eye X gaze position data
    R_X : np.array
        Right eye X gaze position data
    missing_x : scalar
        The value reflecting mising values for X coordinates in the dataset
    L_Y : np.array
        Left eye Y gaze position data
    R_Y : np.array
        Right eye Y gaze position data
    missing_y : scalar
        The value reflecting mising values for Y coordinates in the dataset

    Returns
    -------
    qLMiss : np.array - Boolean
        Boolean indicating missing samples for the left eye
    qRMiss : np.array - Boolean
        Boolean indicating missing samples for the right eye
    qBMiss : np.array - Boolean
        Boolean indicating missing samples for both eyes
    """

    # Get where the missing is

    # Left eye
    qLMissX = np.logical_or(L_X == missing_x, np.isnan(L_X))
    qLMissY = np.logical_or(L_Y == missing_y, np.isnan(L_Y))
    qLMiss = np.logical_and(qLMissX, qLMissY)

    # Right
    qRMissX = np.logical_or(R_X == missing_x, np.isnan(R_X))
    qRMissY = np.logical_or(R_Y == missing_y, np.isnan(R_Y))
    qRMiss = np.logical_and(qRMissX, qRMissY)

    # Both eyes
    qBMiss = np.logical_and(qLMiss, qRMiss)

    return qLMiss, qRMiss, qBMiss


def average_eyes(L_X, R_X, missing_x, L_Y, R_Y, missing_y):
    """
    Averages data from two eyes. Take one eye if only one was found.

    Parameters
    ----------
    L_X : np.array
        Left eye X gaze position data
    R_X : np.array
        Right eye X gaze position data
    missing_x : scalar
        The value reflecting mising values for X coordinates in the dataset
    L_Y : np.array
        Left eye Y gaze position data
    R_Y : np.array
        Right eye Y gaze position data
    missing_y : scalar
        The value reflecting mising values for Y coordinates in the dataset

    Returns
    -------
    xpos   : np.array
        The average Y gaze position
    ypos   : np.array
        The average X gaze position
    qBMiss : np.array - Boolean
        Boolean indicating missing samples for both eyes
    qLMiss : np.array - Boolean
        Boolean indicating missing samples for the left eye
    qRMiss : np.array - Boolean
        Boolean indicating missing samples for the right eye
    """

    xpos = np.zeros(len(L_X))
    ypos = np.zeros(len(L_Y))

    # get missing
    qLMiss, qRMiss, qBMiss = get_missing(L_X, R_X, missing_x, L_Y, R_Y, missing_y)

    q = np.logical_and(np.invert(qLMiss), np.invert(qRMiss))
    xpos[q] = (L_X[q] + R_X[q]) / 2.
    ypos[q] = (L_Y[q] + R_Y[q]) / 2.

    q =  np.logical_and(qLMiss, np.invert(qRMiss))
    xpos[q] = R_X[q]
    ypos[q] = R_Y[q]

    q = np.logical_and(np.invert(qLMiss), qRMiss)
    xpos[q] = L_X[q]
    ypos[q] = L_Y[q]

    xpos[qBMiss] = np.nan
    ypos[qBMiss] = np.nan

    return xpos, ypos, qBMiss, qLMiss, qRMiss

def bool2bounds(b):
    """
    Finds all contiguous sections of true in a boolean

    Parameters
    ----------
    data : np.array - Boolean (or convertible to boolean)
        A 1D array containing stretches of True and False

    Returns
    -------
    on : np.array
        The array contains the indices where each stretch of True starts
    off : np.array
        The array contains the indices where each stretch of True ends

    Example
    --------
    >>> import numpy as np
    >>> b = np.array([1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0])
    >>> on, off = bool2bounds(b)
    >>> print(on)
    [0 4 8]
    >>> print(off)
    [0 6 9]
    """
    b = np.array(np.array(b, dtype = bool), dtype=int)
    b = np.pad(b, (1, 1), 'constant', constant_values=(0, 0))
    D = np.diff(b)
    on  = np.array(np.where(D == 1)[0], dtype=int)
    off = np.array(np.where(D == -1)[0] -1, dtype=int)
    return on, off

# =============================================================================
# Interpolation functions
# =============================================================================
def find_interp_wins(xpos, ypos, missing, window_time, edge_samples, freq, max_disp):
    """
    Description

    Parameters
    ----------
    xpos : np.array
        X gaze position
    ypos : type
        Y gaze position
    missing : type
        Description
    window_time : float
        Duration of window to interpolate over (ms)
    edge_samples : int
        Number of samples at window edge used for interpolating
    freq : float
        Measurement frequency
    max_disp : float
        maximum dispersion in position signal (i.e. if signal is in pixels, provide maxdisp in pixels)

    Returns
    -------
    miss_start : np.array
        Array containing indices where each interval to be interpolated starts
    miss_end : np.array
        Array containing indices where each interval to be interpolated ends
    """

    # get indices of where missing intervals start and end
    miss_start, miss_end = bool2bounds(missing)
    data_start, data_end = bool2bounds(np.invert(missing))

    # Determine windowsamples
    window_samples = round(window_time/(1./freq))

    # for each candidate, check if have enough valid data at edges to execute
    # interpolation. If not, see if merging with adjacent missing is possible
    # we don't throw out anything we can't deal with yet, we do that below.
    # this is just some preprocessing
    k=0  #was K=1 in matlab
    while k<len(miss_start)-1:
        # skip if too long
        if miss_end[k]-miss_start[k]+1 > window_samples:
            k = k+1
            continue

        # skip if not enough data at left edge
        if np.sum(data_end == miss_start[k]-1) > 0:
            datk = int(np.argwhere(data_end==miss_start[k]-1))
            if data_end[datk]-data_start[datk]+1 < edge_samples:
                k = k+1
                continue

        # if not enough data at right edge, merge with next. Having not enough
        # on right edge of this one, means not having enough at left edge of
        # next. So both will be excluded always if we don't do anything. So we
        # can just merge without further checks. Its ok if it then grows too
        # long, as we'll just end up excluding that too below, which is what
        # would have happened if we didn't do anything here
        datk = np.argwhere(data_start==miss_end[k]+1)
        if len(datk) > 0:
            datk = int(datk)
            if data_end[datk]-data_start[datk]+1 < edge_samples:
                miss_end   = np.delete(miss_end  , k)
                miss_start = np.delete(miss_start, k+1)

                # don't advance k so we check this one again and grow it further if
                # needed
                continue

        # nothing left to do, continue to next
        k = k+1

    # mark intervals that are too long to be deleted (only delete later so that
    # below checks can use all missing on and offsets)
    miss_dur = miss_end - miss_start + 1
    qRemove = miss_dur>window_samples

    # for each candidate, check if have enough valid data at edges to execute
    # interpolation and check displacement during missing wasn't too large.
    # Mark for later removal as multiple missing close together may otherwise
    # be wrongly allowed
    for p in range(len(miss_start)):
        # check enough valid data at edges
        # missing too close to beginning of data
        # previous missing too close
        # missing too close to end of data
        # next missing too close
        if miss_start[p]<edge_samples+1 or \
            (p>0 and miss_end[p-1] > miss_start[p]-edge_samples-1) or \
            miss_end[p]>len(xpos)-1-edge_samples or \
            (p<len(miss_start)-1 and miss_start[p+1] < miss_end[p]+edge_samples+1):
            qRemove[p] = True
            continue

        # check displacement, per missing interval
        # we want to check per bit of missing, even if multiple bits got merged
        # this as single data points can still anchor where the interpolation
        # goes and we thus need to check distance per bit, not over the whole
        # merged bit
        idx = np.arange(miss_start[p],miss_end[p]+1, dtype = int)
        on,off = bool2bounds(np.isnan(xpos[idx]))
        for q in range(len(on)):
            lesamps = np.array(on[q] -np.arange(edge_samples)+miss_start[p]-1, dtype=int)
            resamps = np.array(off[q]+np.arange(edge_samples)+miss_start[p]+1, dtype=int)
            displacement = np.hypot(np.nanmedian(xpos[resamps])-np.nanmedian(xpos[lesamps]),
                                    np.nanmedian(ypos[resamps])-np.nanmedian(ypos[lesamps]))
            if displacement > max_disp:
                qRemove[p] = True
                break

        if qRemove[p]:
            continue

    # Remove the missing clusters which cannot be interpolated
    qRemove    = np.where(qRemove)[0]
    miss_start = np.delete(miss_start, qRemove)
    miss_end   = np.delete(miss_end  , qRemove)

    return miss_start, miss_end

def windowed_interpolate(xpos, ypos, missing, miss_start, miss_end, edge_samples):
    """
    Interpolates the missing data, and removes areas which are not allowed
    to be interpolated

    Parameters
    ----------
    xpos : np.array
        X gaze positions
    ypos : type
        Y gaze positions
    missing : np.array
        Boolean vector indicating missing samples
    miss_start : np.array
        Array containing indices where each interval to be interpolated starts
    miss_end : np.array
        Array containing indices where each interval to be interpolated ends
    edge_samples : int
        Number of samples at window edge used for interpolating

    Returns
    -------
    xi : np.array
        Interpolated X gaze position
    yi : np.array
        Interpolated Y gaze position
    new_missing : np.array
        Updated boolean vector indicating missing samples after interpolation
    """
    new_missing = copy.deepcopy(missing)

    # Do the interpolating
    for p in range(len(miss_start)):
        # make vector of all samples in this window
        out_win = np.arange(miss_start[p],miss_end[p]+1)

        # get edge samples: where no missing data was observed
        # also get samples in window where data was observed
        out_win_not_missing = np.invert(new_missing[out_win])
        valid_samps = np.concatenate((out_win[0]+np.arange(-edge_samples,0),
                                      out_win[out_win_not_missing],
                                      out_win[-1]+np.arange(1,edge_samples+1)))

        # get valid values: where no missing data was observed
        valid_x     = xpos[valid_samps]
        valid_y     = ypos[valid_samps]

        # do Steffen interpolation, update xpos, ypos
        xpos[out_win]= steffen_interp(valid_samps,valid_x,out_win)
        ypos[out_win]= steffen_interp(valid_samps,valid_y,out_win)

        # update missing: hole is now plugged
        new_missing[out_win] = False

    return xpos, ypos, new_missing

# =============================================================================
# interpolator
# =============================================================================
def steffen_interp(x, y, xi):
    # STEFFEN 1-D Steffen interpolation
    #    steffenInterp[X,Y,XI] interpolates to find YI, the values of the
    #    underlying function Y at the points in the array XI, using
    #    the method of Steffen.  X and Y must be vectors of length N.
    #
    #    Steffen's method is based on a third-order polynomial. The
    #    slope at each grid point is calculated in a way to guarantee
    #    a monotonic behavior of the interpolating function. The
    #    curve is smooth up to the first derivative.

    # Joe Henning - Summer 2014
    # edited DC Niehorster - Summer 2015

    # M. Steffen
    # A Simple Method for Monotonic Interpolation in One Dimension
    # Astron. Astrophys. 239, 443-450 [1990]
    n = len(x)

    # calculate slopes
    yp = np.zeros(n)

    # first point
    h1 = x[1] - x[0]
    h2 = x[2] - x[1]
    s1 = (y[1] - y[0])/h1
    s2 = (y[2] - y[1])/h2
    p1 = s1*(1 + h1/(h1 + h2)) - s2*h1/(h1 + h2)
    if p1*s1 <= 0:
        yp[0] = 0
    elif np.abs(p1) > 2*np.abs(s1):
        yp[0] = 2*s1
    else:
        yp[0] = p1

    # inner points
    for i in range(1,n-1):
        hi = x[i+1] - x[i]
        him1 = x[i] - x[i-1]
        si = (y[i+1] - y[i])/hi
        sim1 = (y[i] - y[i-1])/him1
        pi = (sim1*hi + si*him1)/(him1 + hi)

        if sim1*si <= 0:
            yp[i] = 0
        elif (np.abs(pi) > 2*np.abs(sim1)) or (np.abs(pi) > 2*np.abs(si)):
            a = np.sign(sim1)
            yp[i] = 2*a*np.min([np.abs(sim1),np.abs(si)])
        else:
            yp[i] = pi

    # last point
    hnm1 = x[n-1] - x[n-2]
    hnm2 = x[n-2] - x[n-3]
    snm1 = (y[n-1] - y[n-2])/hnm1
    snm2 = (y[n-2] - y[n-3])/hnm2
    pn = snm1*(1 + hnm1/(hnm1 + hnm2)) - snm2*hnm1/(hnm1 + hnm2)
    if pn*snm1 <= 0:
        yp[n-1] = 0
    elif np.abs(pn) > 2*np.abs(snm1):
        yp[n-1] = 2*snm1
    else:
        yp[n-1] = pn

    yi = np.zeros(xi.size)
    for i in range(len(xi)):
        # Find the right place in the table by means of a bisection.
        # do this instead of search with find as the below now somehow gets
        # better optimized by matlab's JIT [runs twice as fast].
        klo = 1
        khi = n
        while khi-klo > 1:
            k = int(np.fix((khi+klo)/2.0))
            if x[k] > xi[i]:
                khi = k
            else:
                klo = k

        # check if requested output is in input, so we can just copy
        if xi[i]==x[klo]:
            yi[i] = y[klo]
            continue
        elif xi[i]==x[khi]:
            yi[i] = y[khi]
            continue

        h = x[khi] - x[klo]
        s = (y[khi] - y[klo])/h

        a = (yp[klo] + yp[khi] - 2*s)/h/h
        b = (3*s - 2*yp[klo] - yp[khi])/h
        c = yp[klo]
        d = y[klo]

        t = xi[i] - x[klo]
        # Use Horner's scheme for efficient evaluation of polynomials
        # y = a*t*t*t + b*t*t + c*t + d
        yi[i] = d + t*(c + t*(b + t*a))

    return yi

# =============================================================================
# Clustering functions
# =============================================================================
class NotConvergedError(Exception):
    pass

def kmeans2(data):
    # n points in p dimensional space
    n = data.shape[0]

    max_iterations = 100

    ## initialize using kmeans++ method.
    # code taken and slightly edited from scipy.cluster.vq
    dims = data.shape[1] if len(data.shape) > 1 else 1
    C = np.ndarray((2, dims))

    # first cluster
    C[0, :] = data[np.random.randint(data.shape[0])]

    # second cluster
    D = cdist(C[:1,:], data, metric='sqeuclidean').min(axis=0)
    probs = D/D.sum()
    cumprobs = probs.cumsum()
    r = np.random.rand()
    C[1, :] = data[np.searchsorted(cumprobs, r)]

    # Compute the distance from every point to each cluster centroid and the
    # initial assignment of points to clusters
    D = cdist(C, data, metric='sqeuclidean')
    # Compute the nearest neighbor for each obs using the current code book
    label = vq(data, C)[0]
    # Update the code book by computing centroids
    C = _vq.update_cluster_means(data, label, 2)[0]
    m = np.bincount(label)

    ## Begin phase one:  batch reassignments
    #-----------------------------------------------------
    # Every point moved, every cluster will need an update
    prevtotsumD = math.inf
    iter = 0
    prev_label = None
    while True:
        iter += 1
        # Calculate the new cluster centroids and counts, and update the
        # distance from every point to those new cluster centroids
        Clast = C
        mlast = m
        D = cdist(C, data, metric='sqeuclidean')

        # Deal with clusters that have just lost all their members
        if np.any(m==0):
            i = np.argwhere(m==0)
            d = D[[label],[range(n)]]   # use newly updated distances

            # Find the point furthest away from its current cluster.
            # Take that point out of its cluster and use it to create
            # a new singleton cluster to replace the empty one.
            lonely = np.argmax(d)
            cFrom = label[lonely]    # taking from this cluster
            if m[cFrom] < 2:
                # In the very unusual event that the cluster had only
                # one member, pick any other non-singleton point.
                cFrom = np.argwhere(m>1)[0]
                lonely = np.argwhere(label==cFrom)[0]
            label[lonely] = i

            # Update clusters from which points are taken
            C = _vq.update_cluster_means(data, label, 2)[0]
            m = np.bincount(label)
            D = cdist(C, data, metric='sqeuclidean')

        # Compute the total sum of distances for the current configuration.
        totsumD = np.sum(D[[label],[range(n)]])
        # Test for a cycle: if objective is not decreased, back out
        # the last step and move on to the single update phase
        if prevtotsumD <= totsumD:
            label = prev_label
            C = Clast
            m = mlast
            iter -= 1
            break
        if iter >= max_iterations:
            break

        # Determine closest cluster for each point and reassign points to clusters
        prev_label = label
        prevtotsumD = totsumD
        new_label = vq(data, C)[0]

        # Determine which points moved
        moved = new_label != prev_label
        if np.any(moved):
            # Resolve ties in favor of not moving
            moved[np.bitwise_and(moved, D[0,:]==D[1,:])] = False
        if not np.any(moved):
            break
        label = new_label
        # update centers
        C = _vq.update_cluster_means(data, label, 2)[0]
        m = np.bincount(label)


    #------------------------------------------------------------------
    # Begin phase two:  single reassignments
    #------------------------------------------------------------------
    last_moved = -1
    converged = False
    while iter < max_iterations:
        # Calculate distances to each cluster from each point, and the
        # potential change in total sum of errors for adding or removing
        # each point from each cluster.  Clusters that have not changed
        # membership need not be updated.
        #
        # Singleton clusters are a special case for the sum of dists
        # calculation. Removing their only point is never best, so the
        # reassignment criterion had better guarantee that a singleton
        # point will stay in its own cluster. Happily, we get
        # Del(i,idx(i)) == 0 automatically for them.
        Del = cdist(C, data, metric='sqeuclidean')
        mbrs = label==0
        sgn = 1 - 2*mbrs    # -1 for members, 1 for nonmembers
        if m[0] == 1:
            sgn[mbrs] = 0   # prevent divide-by-zero for singleton mbrs
        Del[0,:] = (m[0] / (m[0] + sgn)) * Del[0,:]
        # same for cluster 2
        sgn = -sgn          # -1 for members, 1 for nonmembers
        if m[1] == 1:
            sgn[np.invert(mbrs)] = 0    # prevent divide-by-zero for singleton mbrs
        Del[1,:] = (m[1] / (m[1] + sgn)) * Del[1,:]

        # Determine best possible move, if any, for each point.  Next we
        # will pick one from those that actually did move.
        prev_label = label
        new_label = (Del[1,:]<Del[0,:]).astype('int')
        moved = np.argwhere(prev_label != new_label)
        if moved.size>0:
            # Resolve ties in favor of not moving
            moved = np.delete(moved,(Del[0,moved]==Del[1,moved]).flatten(),None)
        if moved.size==0:
            converged = True
            break

        # Pick the next move in cyclic order
        moved = (np.min((moved - last_moved % n) + last_moved) % n)

        # If we've gone once through all the points, that's an iteration
        if moved <= last_moved:
            iter = iter + 1
            if iter >= max_iterations:
                break
        last_moved = moved

        olbl = label[moved]
        nlbl = new_label[moved]
        totsumD = totsumD + Del[nlbl,moved] - Del[olbl,moved]

        # Update the cluster index vector, and the old and new cluster
        # counts and centroids
        label[moved] = nlbl
        m[nlbl] += 1
        m[olbl] -= 1
        C[nlbl,:] = C[nlbl,:] + (data[moved,:] - C[nlbl,:]) / m[nlbl]
        C[olbl,:] = C[olbl,:] - (data[moved,:] - C[olbl,:]) / m[olbl]

    #------------------------------------------------------------------
    if not converged:
        raise NotConvergedError('Failed to converge after %d iterations.' % iter)

    return label, C

def two_cluster_weighting(xpos, ypos, missing, downsamples, downsamp_filter, cheby_order, window_time, step_time, freq, max_errors, logging, logging_offset):
    """
    Description

    Parameters
    ----------
    xpos : type
        Description
    ypos : type
        Description
    missing : type
        Description
    downsamples : type
        Description
    downsamp_filter : type
        Description
    cheby_order : type
        Description
    window_time : type
        Description
    step_time : type
        Description
    freq : type
        Description
    max_errors : type
        Description

    Returns
    -------
    finalweights : np.array
        Vector of 2-means clustering weights (one weight for each sample), the higher, the more likely a saccade happened
    stopped : Boolean
        Indicates if stopped because of too many errors encountered (True), or completed successfully (False)
    """
    # calculate number of samples of the moving window
    num_samples = int(window_time/(1./freq))
    step_size  = np.max([1,int(step_time/(1./freq))])

    # create empty weights vector
    total_weights = np.zeros(len(xpos))
    total_weights[missing] = np.nan
    num_tests = np.zeros(len(xpos))

    # stopped is always zero, unless maxiterations is exceeded. this
    # indicates that file could not be analysed after trying for x iterations
    stopped = False
    num_errors = 0

    # Number of downsamples
    nd = len(downsamples)

    # Downsample
    if downsamp_filter:
        # filter signal. Follow the lead of decimate(), which first runs a
        # Chebychev filter as specified below
        rp = .05 # passband ripple in dB
        b = [[] for i in range(nd)]
        a = [[] for i in range(nd)]
        for p in range(nd):
            b[p],a[p] = scipy.signal.cheby1(cheby_order, rp, .8/downsamples[p])


    # idx for downsamples
    idxs = []
    for i in range(nd):
        idxs.append(np.arange(num_samples,0,-downsamples[i],dtype=int)[::-1] - 1)

    # see where are missing in this data, for better running over the data
    # below.
    on,off = bool2bounds(missing)
    if on.size > 0:
        #  merge intervals smaller than nrsamples long
        merge = np.argwhere((on[1:] - off[:-1])-1 < num_samples).flatten()
        for p in merge[::-1]:
            off[p] = off[p+1]
            off = np.delete(off, p+1)
            on = np.delete(on, p+1)

        # check if intervals at data start and end are large enough
        if on[0]<num_samples+1:
            # not enough data point before first missing, so exclude them all
            on[0]=0

        if off[-1]>(len(xpos)-1-num_samples):
            # not enough data points after last missing, so exclude them all
            off[-1]=len(xpos)-1

        # start at first non-missing sample if trial starts with missing (or
        # excluded because too short) data
        if on[0]==0:
            i=off[0]+1 # start at first non-missing
        else:
            i=0
    else:
        i=0

    eind = i+num_samples
    while eind<=(len(xpos)-1):
        # check if max errors is crossed
        if num_errors > max_errors:
            if logging:
                print(logging_offset + 'Too many empty clusters encountered, aborting file. \n')
            stopped = True
            final_weights = np.nan
            return final_weights, stopped

        # select data portion of nrsamples
        idx = range(i,eind)
        ll_d = [[] for p in range(nd+1)]
        IDL_d = [[] for p in range(nd+1)]
        ll_d[0] = np.vstack([xpos[idx], ypos[idx]])

        # Filter the bit of data we're about to downsample. Then we simply need
        # to select each nth sample where n is the integer factor by which
        # number of samples is reduced. select samples such that they are till
        # end of window
        for p in range(nd):
            if downsamp_filter:
                ll_d[p+1] = scipy.signal.filtfilt(b[p],a[p],ll_d[0], padlen=3*(max(len(b[p]),len(a[p]))-1)) # adjust default padlen to match matlab's filtfilt
                ll_d[p+1] = ll_d[p+1][:,idxs[p]]
            else:
                ll_d[p+1] = ll_d[0][:,idxs[p]]

        # do 2-means clustering
        try:
            for p in range(nd+1):
                IDL_d[p] = kmeans2(ll_d[p].T)[0]
        except NotConvergedError as e:
            if logging:
                print(logging_offset + str(e))
            num_errors += 1
        except Exception as e:
            if logging:
                print(logging_offset + 'Unknown error encountered at sample {}.\n'.format(i))
            raise e

        # detect switches and weight of switch (= 1/number of switches in
        # portion)
        switches  = [[] for p in range(nd+1)]
        switchesw = np.empty(nd+1)
        for p in range(nd+1):
            switches[p] = np.abs(np.diff(IDL_d[p]))
            switchesw[p]= 1./np.sum(switches[p])

        # get nearest samples of switch and add weight
        weighted = np.hstack([switches[0]*switchesw[0],0])
        for p in range(nd):
            j = idxs[p][np.argwhere(switches[p+1]).flatten()]
            for o in range(int(downsamples[p])):
                weighted[j+o] = weighted[j+o] + switchesw[p+1]

        # add to totalweights
        total_weights[idx] = total_weights[idx] + weighted
        # record how many times each sample was tested
        num_tests[idx] = num_tests[idx] + 1

        # update i
        i += step_size
        eind += step_size
        missing_on = np.logical_and(on>=i, on<=eind)
        missing_off = np.logical_and(off>=i, off<=eind)
        qWhichMiss = np.logical_or(missing_on, missing_off)
        if np.sum(qWhichMiss) > 0:
            # we have some missing in this window. we don't process windows
            # with missing. Move back if we just skipped some samples, or else
            # skip whole missing and place start of window and first next
            # non-missing.
            if on[qWhichMiss][0] == (eind-step_size):
                # continue at first non-missing
                i = off[qWhichMiss][0]+1
            else:
                # we skipped some points, move window back so that we analyze
                # up to first next missing point
                i = on[qWhichMiss][0]-num_samples
            eind = i+num_samples

        if eind>len(xpos)-1 and eind-step_size<len(xpos)-1:
            # we just exceeded data bound, but previous eind was before end of
            # data: we have some unprocessed samples. retreat just enough so we
            # process those end samples once
            d = eind-len(xpos)+1
            eind = eind-d
            i = i-d


    # create final weights
    np.seterr(invalid='ignore')
    final_weights = total_weights/num_tests
    np.seterr(invalid='warn')

    return final_weights, stopped

# =============================================================================
# Fixation detection functions
# =============================================================================
def get_fixations(final_weights, timestamp, xpos, ypos, missing, par):
    """
    Description

    Parameters
    ----------
    finalweights : type
        weighting from 2-means clustering procedure
    timestamp : np.array
        Timestamp from eyetracker (should be in ms!)
    xpos : np.array
        Horizontal coordinates from Eyetracker
    ypos : np.array
        Vertical coordinates from Eyetracker
    missing : np.array
        Vector containing the booleans for mising values
    par : Dictionary containing the following keys and values
        cutoffstd : float
            Number of std above mean clustering-weight to use as fixation cutoff
        onoffsetThresh : float
            Threshold (x*MAD of fixation) for walking forward/back for saccade off- and onsets
        maxMergeDist : float
            Maximum Euclidean distance in pixels between fixations for merging
        maxMergeTime : float
            Maximum time in ms between fixations for merging
        minFixDur : Float
            Minimum duration allowed for fiation


    Returns
    -------
    fix : Dictionary containing the following keys and values
        cutoff : float
            Cutoff used for fixation detection
        start : np.array
            Vector with fixation start indices
        end : np.array
            Vector with fixation end indices
        startT : np.array
            Vector with fixation start times
        endT : np.array
            Vector with fixation end times
        dur : type
            Vector with fixation durations
        xpos : np.array
            Vector with fixation median horizontal position (one value for each fixation in trial)
        ypos : np.array
            Vector with fixation median vertical position (one value for each fixation in trial)
        flankdataloss : bool
            Boolean with 1 for when fixation is flanked by data loss, 0 if not flanked by data loss
        fracinterped : float
            Fraction of data loss/interpolated data

    Examples
    --------
    >>> fix = getFixations(finalweights,data['time'],xpos,ypos,missing,par)
    >>> fix
        {'cutoff': 0.1355980099309374,
         'dur': array([366.599, 773.2  , 239.964, 236.608, 299.877, 126.637]),
         'end': array([111, 349, 433, 508, 600, 643]),
         'endT': array([373.284, 1166.525, 1446.462, 1696.398, 2002.993, 2146.306]),
         'flankdataloss': array([1., 0., 0., 0., 0., 0.]),
         'fracinterped': array([0.06363636, 0.        , 0.        , 0.        , 0.        ,
                0.        ]),
         'start': array([  2, 118, 362, 438, 511, 606]),
         'startT': array([   6.685,  393.325, 1206.498, 1459.79 , 1703.116, 2019.669]),
         'xpos': array([ 945.936,  781.056, 1349.184, 1243.92 , 1290.048, 1522.176]),
         'ypos': array([486.216, 404.838, 416.664, 373.005, 383.562, 311.904])}
    """
    ### Extract the required parameters
    cutoffstd = par['cutoffstd']
    onoffsetThresh = par['onoffsetThresh']
    maxMergeDist = par['maxMergeDist']
    maxMergeTime = par['maxMergeTime']
    minFixDur = par['minFixDur']

    ### first determine cutoff for finalweights
    cutoff = np.nanmean(final_weights) + cutoffstd*np.nanstd(final_weights,ddof=1)

    ### get boolean of fixations
    fixbool = final_weights < cutoff

    ### get indices of where fixations start and end
    fixstart, fixend = bool2bounds(fixbool)

    ### for each fixation start, walk forward until recorded position is below
    # a threshold of lambda*MAD away from median fixation position.
    # same for each fixation end, but walk backward
    for p in range(len(fixstart)):
        xFix = xpos[fixstart[p]:fixend[p]+1]
        yFix = ypos[fixstart[p]:fixend[p]+1]
        xmedThis = np.nanmedian(xFix)
        ymedThis = np.nanmedian(yFix)

        # MAD = median(abs(x_i-median({x}))). For the 2D version, I'm using
        # median 2D distance of a point from the median fixation position. Not
        # exactly MAD, but makes more sense to me for 2D than city block,
        # especially given that we use 2D distance in our walk here
        MAD = np.nanmedian(np.hypot(xFix-xmedThis, yFix-ymedThis))
        thresh = MAD*onoffsetThresh

        # walk until distance less than threshold away from median fixation
        # position. No walking occurs when we're already below threshold.
        i = fixstart[p]
        if i>0:  # don't walk when fixation starting at start of data
            while np.hypot(xpos[i]-xmedThis,ypos[i]-ymedThis)>thresh:
                i = i+1
            fixstart[p] = i

        # and now fixation end.
        i = fixend[p]
        if i<len(xpos): # don't walk when fixation ending at end of data
            while np.hypot(xpos[i]-xmedThis,ypos[i]-ymedThis)>thresh:
                i = i-1
            fixend[p] = i

    ### get start time, end time,
    starttime = timestamp[fixstart]
    endtime = timestamp[fixend]

    ### loop over all fixation candidates in trial, see if should be merged
    for p in range(1,len(starttime))[::-1]:
        # get median coordinates of fixation
        xmedThis = np.median(xpos[fixstart[p]:fixend[p]+1])
        ymedThis = np.median(ypos[fixstart[p]:fixend[p]+1])
        xmedPrev = np.median(xpos[fixstart[p-1]:fixend[p-1]+1])
        ymedPrev = np.median(ypos[fixstart[p-1]:fixend[p-1]+1])

        # check if fixations close enough in time and space and thus qualify
        # for merging
        # The interval between the two fixations is calculated correctly (see
        # notes about fixation duration below), i checked this carefully. (Both
        # start and end of the interval are shifted by one sample in time, but
        # assuming practically constant sample interval, thats not an issue.)
        if starttime[p]-endtime[p-1] < maxMergeTime and \
            np.hypot(xmedThis-xmedPrev,ymedThis-ymedPrev) < maxMergeDist:
            # merge
            fixend[p-1] = fixend[p]
            endtime[p-1]= endtime[p]
            # delete merged fixation
            fixstart = np.delete(fixstart, p)
            fixend = np.delete(fixend, p)
            starttime = np.delete(starttime, p)
            endtime = np.delete(endtime, p)

    ### beginning and end of fixation must be real data, not interpolated.
    # If interpolated, those bit(s) at the edge(s) are excluded from the
    # fixation. First throw out fixations that are all missing/interpolated
    for p in range(len(starttime))[::-1]:
        miss = missing[fixstart[p]:fixend[p]+1]
        if np.sum(miss) == len(miss):
            fixstart = np.delete(fixstart, p)
            fixend = np.delete(fixend, p)
            starttime = np.delete(starttime, p)
            endtime = np.delete(endtime, p)

    # then check edges and shrink if needed
    for p in range(len(starttime)):
        if missing[fixstart[p]]:
            fixstart[p] = fixstart[p] + np.argmax(np.invert(missing[fixstart[p]:fixend[p]+1]))
            starttime[p]= timestamp[fixstart[p]]
        if missing[fixend[p]]:
            fixend[p] = fixend[p] - (np.argmax(np.invert(missing[fixstart[p]:fixend[p]+1][::-1]))+1)
            endtime[p] = timestamp[fixend[p]]

    ### adjust fixation end time
    # since our fixation markers are inclusive, endT is at the end of the sample,
    # not at its start. At this stage in the code, endtime simply reflects the
    # timestamp of the end sample. But such a timestamp is for the beginning of
    # the sample, not for its end. To solve this, as end time we need to take the
    # timestamp of the sample that is one past the last sample of the fixation.
    # this is also important for calculating fixation duration. Without this
    # adjustment, we would always underestimate fixation duration by one sample
    # because you're in practice counting to the beginning of the sample, not
    # the end of it.

    # 1. determine what the duration of each end sample was
    nextSamp = np.min(np.vstack([fixend+1,np.zeros(len(fixend),dtype=int)+len(timestamp)-1]),axis=0) # make sure we don't run off the end of the data
    extratime = timestamp[nextSamp]-timestamp[fixend]

    # if last fixation ends at end of data, we need to determine how long that
    # sample is and add that to the end time. Here we simply guess it as the
    # duration of previous sample
    if not len(fixend)==0 and fixend[-1]==len(timestamp): # first check if there are fixations in the first place, or we'll index into non-existing data
        extratime[-1] = np.diff(timestamp[-3:-1])

    # now add the duration of the sample to each fixation end time, so that they are
    # correct
    endtime = endtime+extratime

    ### calculate fixation duration
    fixdur = endtime-starttime

    ### check if any fixations are too short
    qTooShort = np.argwhere(fixdur<minFixDur)
    if len(qTooShort) > 0:
        fixstart = np.delete(fixstart, qTooShort)
        fixend = np.delete(fixend, qTooShort)
        starttime = np.delete(starttime, qTooShort)
        endtime = np.delete(endtime, qTooShort)
        fixdur = np.delete(fixdur, qTooShort)

    ### process fixations, get other info about them
    xmedian = np.zeros(fixstart.shape) # vector for median
    ymedian = np.zeros(fixstart.shape)  # vector for median
    flankdataloss = np.zeros(fixstart.shape) # vector for whether fixation is flanked by data loss
    fracinterped = np.zeros(fixstart.shape) # vector for fraction interpolated
    for a in range(len(fixstart)):
        idxs = range(fixstart[a],fixend[a]+1)
        # get data during fixation
        xposf = xpos[idxs]
        yposf = ypos[idxs]
        # for all calculations below we'll only use data that is not
        # interpolated, so only real data
        qMiss = missing[idxs]

        # get median coordinates of fixation
        xmedian[a] = np.median(xposf[np.invert(qMiss)])
        ymedian[a] = np.median(yposf[np.invert(qMiss)])

        # determine whether fixation is flanked by period of data loss
        flankdataloss[a] = (fixstart[a]>0 and missing[fixstart[a]-1]) or (fixend[a]<len(xpos)-1 and missing[fixend[a]+1])

        # fraction of data loss during fixation that has been (does not count
        # data that is still lost)
        fracinterped[a] = np.sum(np.invert(np.isnan(xposf[qMiss])))/(fixend[a]-fixstart[a]+1)

    # store all the results in a dictionary
    fix = {}
    fix['cutoff'] = cutoff
    fix['start'] = fixstart
    fix['end'] = fixend
    fix['startT'] = np.array(starttime)
    fix['endT'] = np.array(endtime)
    fix['dur'] = np.array(fixdur)
    fix['xpos'] = xmedian
    fix['ypos'] = ymedian
    fix['flankdataloss'] = flankdataloss
    fix['fracinterped'] = fracinterped
    return fix

def get_fix_stats(xpos, ypos, missing, fix, pix_per_deg = None):
    """
    Description

    Parameters
    ----------
    xpos : np.array
        X gaze positions
    ypos : np.array
        Y gaze positions
    missing : np.array - Boolean
        Vector containing the booleans indicating missing samples (originally, before interpolation!)
    fix : Dictionary containing the following keys and values
        fstart : np.array
            fixation start indices
        fend : np.array
            fixation end indices
    pixperdeg : float
        Number of pixels per visual degree. Output in degrees if provided, in pixels otherwise


    Returns
    -------
    fix : the fix input dictionary with the following added keys and values
        RMSxy : float
            RMS of fixation (precision)
        BCEA : float
            BCEA of fixation (precision)
        rangeX : float
            max(xpos) - min(xpos) of fixation
        rangeY : float
            max(ypos) - min(ypos) of fixation

    Examples
    --------
    >>> fix = getFixStats(xpos,ypos,missing,fix,pixperdeg)
    >>> fix
        {'BCEA': array([0.23148877, 0.23681681, 0.24498942, 0.1571361 , 0.20109245,
            0.23703843]),
     'RMSxy': array([0.2979522 , 0.23306149, 0.27712236, 0.26264146, 0.28913117,
            0.23147076]),
     'cutoff': 0.1355980099309374,
     'dur': array([366.599, 773.2  , 239.964, 236.608, 299.877, 126.637]),
     'end': array([111, 349, 433, 508, 600, 643]),
     'endT': array([373.284, 1166.525, 1446.462, 1696.398, 2002.993, 2146.306]),
     'fixRangeX': array([0.41066299, 0.99860672, 0.66199772, 0.49593727, 0.64628929,
            0.81010568]),
     'fixRangeY': array([1.58921528, 1.03885955, 1.10576059, 0.94040142, 1.21936613,
            0.91263117]),
     'flankdataloss': array([1., 0., 0., 0., 0., 0.]),
     'fracinterped': array([0.06363636, 0.        , 0.        , 0.        , 0.        ,
            0.        ]),
     'start': array([  2, 118, 362, 438, 511, 606]),
     'startT': array([   6.685,  393.325, 1206.498, 1459.79 , 1703.116, 2019.669]),
     'xpos': array([ 945.936,  781.056, 1349.184, 1243.92 , 1290.048, 1522.176]),
     'ypos': array([486.216, 404.838, 416.664, 373.005, 383.562, 311.904])}
    """

    # Extract the required parameters
    fstart = fix['start']
    fend = fix['end']

    # vectors for precision measures
    RMSxy = np.zeros(fstart.shape)
    BCEA  = np.zeros(fstart.shape)
    rangeX = np.zeros(fstart.shape)
    rangeY = np.zeros(fstart.shape)

    for a in range(len(fstart)):
        idxs = range(fstart[a],fend[a]+1)
        # get data during fixation
        xposf = xpos[idxs]
        yposf = ypos[idxs]
        # for all calculations below we'll only use data that is not
        # interpolated, so only real data
        qMiss = missing[idxs]

        ### calculate RMS
        # since its done with diff, don't just exclude missing and treat
        # resulting as one continuous vector. replace missing with nan first,
        # use left-over values
        # Difference x position
        xdif = xposf.copy()
        xdif[qMiss] = np.nan
        xdif = np.diff(xdif)**2
        xdif = xdif[np.invert(np.isnan(xdif))]
        # Difference y position
        ydif = yposf.copy()
        ydif[qMiss] = np.nan
        ydif = np.diff(ydif)**2
        ydif = ydif[np.invert(np.isnan(ydif))]
        # Distance and RMS measure
        c = xdif + ydif # 2D sample-to-sample displacement value in pixels
        RMSxy[a] = np.sqrt(np.mean(c))
        if pix_per_deg is not None:
            RMSxy[a] = RMSxy[a]/pix_per_deg # value in degrees visual angle

        ### calculate BCEA (Crossland and Rubin 2002 Optometry and Vision Science)
        stdx = np.std(xposf[np.invert(qMiss)],ddof=1)
        stdy = np.std(yposf[np.invert(qMiss)],ddof=1)
        if pix_per_deg is not None:
            # value in degrees visual angle
            stdx = stdx/pix_per_deg
            stdy = stdy/pix_per_deg

        if len(yposf[np.invert(qMiss)])<2:
            BCEA[a] = np.nan
        else:
            xx = np.corrcoef(xposf[np.invert(qMiss)],yposf[np.invert(qMiss)])
            rho = xx[0,1]
            P = 0.68 # cumulative probability of area under the multivariate normal
            k = np.log(1./(1-P))
            BCEA[a] = 2*k*np.pi*stdx*stdy*np.sqrt(1-rho**2)

        ### calculate max-min of fixation
        if np.sum(qMiss) == len(qMiss):
            rangeX[a] = np.nan
            rangeY[a] = np.nan
        else:
            rangeX[a] = (np.max(xposf[np.invert(qMiss)]) - np.min(xposf[np.invert(qMiss)]))
            rangeY[a] = (np.max(yposf[np.invert(qMiss)]) - np.min(yposf[np.invert(qMiss)]))

        if pix_per_deg is not None:
            # value in degrees visual angle
            rangeX[a] = rangeX[a]/pix_per_deg
            rangeY[a] = rangeY[a]/pix_per_deg

    # Add results to fixation dictionary
    fix['RMSxy'] = RMSxy
    fix['BCEA'] = BCEA
    fix['fixRangeX'] = rangeX
    fix['fixRangeY'] = rangeY

    return fix

# =============================================================================
# =============================================================================
# # The actual I2MC pipeline function
# =============================================================================
# =============================================================================
def I2MC(gazeData, options = None, logging=True, logging_offset=""):
    """
    Parameters
    ----------
    @param gazeData: a dataframe containing the gaze data
        the dataframe should contain the following column:
            time        - time of the gaze sample (ms)
        and the dataframe should furthermore contain at least some of the
        following columns of eye data (either L, or R, or both L and R, or average):
            L_X         - left eye x position
            L_Y         - left eye y position
            R_X         - right eye x position
            R_Y         - right eye y position
            average_X   - average x position
            average_Y   - average y position
    @param options: a dictionary containing the options for the I2MC analysis
        the dictionary should contain the following keys:
            x_res        - x resolution of the screen in pixels
            y_res        - y resolution of the screen in pixels
            freq         - frequency of the Eyetracker in Hz
            missing_x    - value indicating data loss
            missing_y    - value indicating data loss
    @param logging: boolean indicating whether to log the results
    @param logging_offset: offset before every logging message
    Returns
    -------
    @return: false if the analysis was not successful, otherwise a dictionary
        containing the results of the analysis.
        The Dictionary contains the following keys:
            cutoff          -
            start           -
            end             -
            startT          -
            endT            -
            dur             -
            xpos            -
            ypos            -
            flankdataloss   -
            fracinterped    -
            RMSxy           -
            BCEA            -
            fixRangeX       -
            fixRangeY       -
    """

    # set defaults
    if options is None:
        options = {}
    opt  = options.copy()

    if isinstance(gazeData,dict):
        # for backward compatibility, convert to pd.DataFrame
        data = pd.DataFrame.from_dict(gazeData)
    else:
        data = copy.deepcopy(gazeData)

    par  = {}

    # Check required parameters and extract
    check_fun('xres',     opt, 'horizontal screen resolution')
    check_fun('yres',     opt, 'vertical screen resolution')
    check_fun('freq',     opt, 'tracker sampling rate')
    check_fun('missingx', opt, 'value indicating data loss for horizontal position')
    check_fun('missingy', opt, 'value indicating data loss for vertical position')
    par['xres']             = opt.pop('xres')
    par['yres']             = opt.pop('yres')
    par['freq']             = opt.pop('freq')
    par['missingx']         = opt.pop('missingx')
    par['missingy']         = opt.pop('missingy')

    # parameters for the calculation of angular measures. if either or both are
    # None, the noise measures are provided in pixels instead of degrees
    par['scrSz']            = opt.pop('scrSz', None )           # screen size (e.g. in cm). Optional, specify if want fixation statistics in deg
    par['disttoscreen']     = opt.pop('disttoscreen', None)     # screen distance (in same unit as size). Optional, specify if want fixation statistics in deg

    #parameters with defaults:
    # CUBIC SPLINE INTERPOLATION
    par['windowtimeInterp'] = opt.pop('windowtimeInterp', .1)   # max duration (s) of missing values for interpolation to occur
    par['edgeSampInterp']   = opt.pop('edgeSampInterp', 2)      # amount of data (number of samples) at edges needed for interpolation
    par['maxdisp']          = opt.pop('maxdisp', None)          # maximum displacement during missing for interpolation to be possible. Default set below if needed

    # K-MEANS CLUSTERING
    par['windowtime']       = opt.pop('windowtime', .2)         # time window (s) over which to calculate 2-means clustering (choose value so that max. 1 saccade can occur)
    par['steptime']         = opt.pop('steptime', .02)          # time window shift (s) for each iteration. Use zero for sample by sample processing
    par['downsamples']      = opt.pop('downsamples', [2, 5, 10]) # downsample levels (can be empty)
    par['downsampFilter']   = opt.pop('downsampFilter', True)   # use chebychev filter when downsampling? True: yes, False: no. requires signal processing toolbox. is what matlab's downsampling functions do, but could cause trouble (ringing) with the hard edges in eye-movement data
    par['chebyOrder']       = opt.pop('chebyOrder', 8.)         # order of cheby1 Chebyshev downsampling filter, default is normally ok, as long as there are 25 or more samples in the window (you may have less if your data is of low sampling rate or your window is small
    par['maxerrors']        = opt.pop('maxerrors', 100.)        # maximum number of errors allowed in k-means clustering procedure before proceeding to next file
    # FIXATION DETERMINATION
    par['cutoffstd']        = opt.pop('cutoffstd', 2.)          # number of standard deviations above mean k-means weights will be used as fixation cutoff
    par['onoffsetThresh']   = opt.pop('onoffsetThresh', 3.)     # number of MAD away from median fixation duration. Will be used to walk forward at fixation starts and backward at fixation ends to refine their placement and stop algorithm from eating into saccades
    par['maxMergeDist']     = opt.pop('maxMergeDist', 30.)      # maximum Euclidean distance in pixels between fixations for merging
    par['maxMergeTime']     = opt.pop('maxMergeTime', 30.)      # maximum time in ms between fixations for merging
    par['minFixDur']        = opt.pop('minFixDur', 40.)         # minimum fixation duration (ms) after merging, fixations with shorter duration are removed from output

    # Development parameters, change these to False when not developing
    par['skip_inputhandling']  = opt.pop('skip_inputhandling', False)

    for key in opt:
        assert False, 'Key "{}" not recognized'.format(key)

    # =============================================================================
    # # Input handeling and checking
    # =============================================================================
    ## loop over input
    if not par['skip_inputhandling']:
        for key, value in par.items():
            if key in ['xres','yres','freq','missingx','missingy','windowtimeInterp','maxdisp','windowtime',
                       'steptime','cutoffstd','onoffsetThresh','maxMergeDist','maxMergeTime','minFixDur']:
                check_numeric(key,value)
                check_scalar(key,value)
            elif key == 'disttoscreen':
                if value is not None:   # may be None (its an optional parameter)
                    check_numeric(key,value)
                    check_scalar(key,value)
            elif key in ['downsampFilter','chebyOrder','maxerrors','edgeSampInterp']:
                check_int(key,value)
                check_scalar(key,value)
            elif key == 'scrSz':
                if value is not None:   # may be None (its an optional parameter)
                    check_numeric(key,value)
                    check_vector_2(key,value)
            elif key == 'downsamples':
                check_int(key,value)
            else:
                if type(key) != str:
                    raise ValueError('Key "{}" not recognized'.format(key))

    # set defaults
    if par['maxdisp'] is None:
        par['maxdisp'] = par['xres']*0.2*np.sqrt(2)

    # check filter
    if par['downsampFilter']:
        nSampRequired = np.max([1,3*par['chebyOrder']])+1  # nSampRequired = max(1,3*(nfilt-1))+1, where nfilt = chebyOrder+1
        nSampInWin = round(par['windowtime']/(1./par['freq']))
        if nSampInWin < nSampRequired:
            raise ValueError('I2MC: Filter parameters requested with the setting "chebyOrder" ' +
                             'will not work for the sampling frequency of your data. Please lower ' +
                             '"chebyOrder", or set the setting "downsampFilter" to False')

    assert np.sum(par['freq']%np.array(par['downsamples'])) ==0,'I2MCfunc: Some of your downsample levels are not divisors of your sampling frequency. Change the option "downsamples"'

    # setup visual angle conversion
    pix_per_deg = None
    if par['scrSz'] is not None and par['disttoscreen'] is not None:
        pix_per_deg = angle_to_pixels(1, par['disttoscreen'], par['scrSz'][0], [par['xres'], par['yres']])

    # =============================================================================
    # Determine missing values and determine X and Y gaze pos
    # =============================================================================
    # deal with monocular data, or create average over two eyes
    if 'L_X' in data.keys() and 'R_X' not in data.keys():
        xpos = data['L_X'].array
        ypos = data['L_Y'].array
        # Check for missing values
        missing_x = np.logical_or(np.isnan(xpos), xpos == par['missingx'])
        missing_y = np.logical_or(np.isnan(ypos), ypos == par['missingy'])
        missing = np.logical_or(missing_x, missing_y)
        data['left_missing'] = missing
        q2Eyes = False

    elif 'R_X' in data.keys() and 'L_X' not in data.keys():
        xpos = data['R_X'].array
        ypos = data['R_Y'].array
        # Check for missing values
        missing_x = np.logical_or(np.isnan(xpos), xpos == par['missingx'])
        missing_y = np.logical_or(np.isnan(ypos), ypos == par['missingy'])
        missing = np.logical_or(missing_x, missing_y)
        data['right_missing'] = missing
        q2Eyes = False

    elif 'average_X' in data.keys():
        xpos = data['average_X'].array
        ypos = data['average_Y'].array
        missing_x = np.logical_or(np.isnan(xpos), xpos == par['missingx'])
        missing_y = np.logical_or(np.isnan(ypos), ypos == par['missingy'])
        missing = np.logical_or(missing_x, missing_y)
        data['average_missing'] = missing
        q2Eyes = 'R_X' in data.keys() and 'L_X' in data.keys()
        if q2Eyes:
            # we have left and right and average already provided, but we need
            # to get missing in the individual eye signals
            llmiss, rrmiss, bothmiss = get_missing(data['L_X'], data['R_X'], par['missingx'], data['L_Y'], data['R_Y'], par['missingy'])
            data['left_missing']  = llmiss
            data['right_missing'] = rrmiss

    else: # we have left and right, average them
        data['average_X'], data['average_Y'], missing, llmiss, rrmiss = average_eyes(data['L_X'].array, data['R_X'].array, par['missingx'], data['L_Y'].array, data['R_Y'].array, par['missingy'])
        xpos = data['average_X'].array
        ypos = data['average_Y'].array
        data['average_missing'] = missing
        data['left_missing']  = llmiss
        data['right_missing'] = rrmiss
        q2Eyes = True

    # =============================================================================
    # INTERPOLATION
    # =============================================================================
    # get interpolation windows for average and individual eye signals
    if logging:
        print(logging_offset + 'I2MC: Searching for valid interpolation windows')
    missStart,missEnd = find_interp_wins(xpos, ypos, missing, par['windowtimeInterp'],
                                         par['edgeSampInterp'], par['freq'], par['maxdisp'])
    if q2Eyes:
        llmissStart,llmissEnd = find_interp_wins(data['L_X'].array, data['L_Y'].array, llmiss, par['windowtimeInterp'],
                                                 par['edgeSampInterp'], par['freq'], par['maxdisp'])
        rrmissStart,rrmissEnd = find_interp_wins(data['R_X'].array, data['R_Y'].array, rrmiss, par['windowtimeInterp'],
                                                 par['edgeSampInterp'], par['freq'], par['maxdisp'])
    
    # Use Steffen interpolation and replace values
    if logging:
        print(logging_offset + 'I2MC: Replace interpolation windows with Steffen interpolation')
    xpos, ypos, missingn = windowed_interpolate(xpos, ypos, missing, missStart, missEnd, par['edgeSampInterp'])
    if q2Eyes:
        llx, lly,llmissn = windowed_interpolate(data['L_X'].array, data['L_Y'].array, data['left_missing'].array,
                                                llmissStart, llmissEnd, par['edgeSampInterp'])
        rrx, rry,rrmissn = windowed_interpolate(data['R_X'].array, data['R_Y'].array, data['right_missing'].array,
                                                rrmissStart, rrmissEnd, par['edgeSampInterp'])

    # =============================================================================
    # 2-MEANS CLUSTERING
    # =============================================================================
    ## CALCULATE 2-MEANS CLUSTERING FOR SINGLE EYE
    if not q2Eyes:
        # get kmeans-clustering for averaged signal
        if logging:
            print(logging_offset + 'I2MC: 2-Means clustering started for averaged signal')
        data['finalweights'], stopped = two_cluster_weighting(xpos, ypos, missingn, par['downsamples'],
                                                              par['downsampFilter'], par['chebyOrder'],
                                                              par['windowtime'], par['steptime'],par['freq'],
                                                              par['maxerrors'], logging, logging_offset)

        # check whether clustering succeeded
        if stopped:
            warnings.warn('I2MC: Clustering stopped after exceeding max errors, continuing to next file \n')
            return False, None, None

    ## CALCULATE 2-MEANS CLUSTERING FOR SEPARATE EYES
    elif q2Eyes:
        # get kmeans-clustering for left eye signal
        if logging:
            print(logging_offset + 'I2MC: 2-Means clustering started for left eye signal')
        finalweights_left, stopped = two_cluster_weighting(llx, lly, llmissn, par['downsamples'],
                                                           par['downsampFilter'], par['chebyOrder'],
                                                           par['windowtime'], par['steptime'], par['freq'],
                                                           par['maxerrors'], logging, logging_offset)

        # check whether clustering succeeded
        if stopped:
            warnings.warn('I2MC: Clustering stopped after exceeding max errors, continuing to next file \n')
            return False, None, None

        # get kmeans-clustering for right eye signal
        if logging:
            print(logging_offset + 'I2MC: 2-Means clustering started for right eye signal')
        finalweights_right, stopped = two_cluster_weighting(rrx, rry, rrmissn, par['downsamples'],
                                                            par['downsampFilter'],par['chebyOrder'],
                                                            par['windowtime'], par['steptime'], par['freq'],
                                                            par['maxerrors'], logging, logging_offset)

        # check whether clustering succeeded
        if stopped:
            warnings.warn('I2MC: Clustering stopped after exceeding max errors, continuing to next file')
            return False, None, None

        ## AVERAGE FINALWEIGHTS OVER COMBINED & SEPARATE EYES
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # ignore warnings from np.nanmean
            data['finalweights'] = np.nanmean(np.vstack([finalweights_left, finalweights_right]), axis=0)

    # =============================================================================
    #  DETERMINE FIXATIONS BASED ON FINALWEIGHTS_AVG
    # =============================================================================
    if logging:
        print(logging_offset + 'I2MC: Determining fixations based on clustering weight mean for averaged signal and separate eyes + {:.2f}*std'.format(par['cutoffstd']))
    fix = get_fixations(data['finalweights'].array, data['time'].array, xpos, ypos, missing, par)
    fix = get_fix_stats(xpos, ypos, missing, fix, pix_per_deg)

    return fix,data,par
