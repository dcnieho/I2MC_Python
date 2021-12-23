# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:54:00 2019

@author: Jonathan
"""

import copy

import warnings
import numpy as np
import scipy
import math
import scipy.signal
from scipy.cluster.vq import vq, _vq

# =============================================================================
# =============================================================================
# General stuff
# - Check all the internal_helpers help text (type, description etc)

# Functions (Interpolate)
# - Can we use my implementation? not steffen

# Functions (CLUSTERING)
# - twoClusterWeighting
#   - Downsampling
#   - missing values handling?
#   - Edges and going back for the latest samples

# Questions
# When getting median gaze position (fixation), the interpolated values are ignored.
#   Is this intended, then whats the point of interpolation, besides the clustering?
# Not all results are identical to matlab version
#   for instance, trial 1 pp2, fix5 and 6 have a slight overlap
#   pp2 trial 3,4 and 5 has a fixations that's to short compared to matlab (almost last fixation)


# =============================================================================
# Helper internal_helpers
# =============================================================================
from scipy.spatial.distance import cdist


def isNumber(s):
    try:
        np.array(s, dtype=float)
        return True
    except ValueError:
        return False


def checkNumeric(k, v):
    if not isNumber(v):
        raise ValueError('I2MC: The value of "{}" is invalid. Expected input to be a number.'.format(k))


def checkScalar(k, v):
    if not np.ndim(v) == 0:
        raise ValueError('I2MC: The value of "{}" is invalid. Expected input to be a scalar.'.format(k))


def check2Vector(k, v):
    if not np.size(v) == 2:
        raise ValueError('I2MC: The value of "{}" is invalid. Expected input to be a 2-element vector.'.format(k))


def checkIntegers(k, v):
    if np.sum(np.array(v) % 1) != 0:
        raise ValueError('I2MC: The value of "{}" is invalid. Expected input to be an integer or list of ints.'
                         .format(k))


def checkFun(k, d, _s):
    if k not in d.keys():
        raise ValueError('I2MC: "{}" must be specified using the key {}'.format(d, k))
    if not isNumber(d[k]):
        raise ValueError('I2MC: "{}" must be a number'.format(k))


def angleToPixels(angle, screenDist, screenW, screenXY):
    """
    Calculate the number of pixels which equals a specified angle in visual
    degrees, given parameters. Calculates the pixels based on the width of
    the screen. If the pixels are not square, a separate conversion needs
    to be done with the height of the screen.\n
    "angleToPixelsWH" returns pixels for width and height.

    Parameters
    ----------

    Returns
    -------

    """
    pixSize = screenW / float(screenXY[0])
    angle = np.radians(angle / 2.0)
    cmOnScreen = np.tan(angle) * float(screenDist)
    pix = (cmOnScreen / pixSize) * 2

    return pix


def getMissing(L_X, R_X, missing_x, L_Y, R_Y, missing_y):
    """
    Gets missing data and returns missing data for left, right and average

    Parameters
    ----------

    Returns
    -------


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


def averageEyes(L_X, R_X, missing_x, L_Y, R_Y, missing_y):
    """
    Averages data from two eyes. Take one eye if only one was found.

    Parameters
    ----------

    Returns
    -------

    """

    xpos = np.zeros(len(L_X))
    ypos = np.zeros(len(L_Y))

    # get missing
    qLMiss, qRMiss, qBMiss = getMissing(L_X, R_X, missing_x, L_Y, R_Y, missing_y)

    q = np.logical_and(np.invert(qLMiss), np.invert(qRMiss))
    xpos[q] = (L_X[q] + R_X[q]) / 2.
    ypos[q] = (L_Y[q] + R_Y[q]) / 2.

    q = np.logical_and(qLMiss, np.invert(qRMiss))
    xpos[q] = R_X[q]
    ypos[q] = R_Y[q]

    q = np.logical_and(np.invert(qLMiss), qRMiss)
    xpos[q] = L_X[q]
    ypos[q] = L_Y[q]

    xpos[qBMiss] = np.NAN
    ypos[qBMiss] = np.NAN

    return xpos, ypos, qBMiss, qLMiss, qRMiss


def bool2bounds(b):
    """
    Finds all contiguous sections of true in a boolean

    Parameters
    ----------


    Returns
    -------

    """
    b = np.array(np.array(b, dtype=np.bool_), dtype=int)
    b = np.pad(b, (1, 1), 'constant', constant_values=(0, 0))
    D = np.diff(b)
    on = np.array(np.where(D == 1)[0], dtype=int)
    off = np.array(np.where(D == -1)[0] - 1, dtype=int)
    return on, off


def getCluster(b):
    """
    Splits a np.array with True, False values into clusters. A cluster is
    defined as adjacent points with the same value, e.g. True or False.
    The output from this function is used to determine cluster sizes when
    running cluster statistics.

    Parameters
    ----------

    Returns
    -------

    """
    if b.dtype != 'bool':
        b = np.array(b, dtype=np.bool)
    clusters = np.split(b, np.where(np.diff(b) != 0)[0] + 1)
    idx = np.split(np.arange(len(b)), np.where(np.diff(b) != 0)[0] + 1)
    size = np.array([len(c) for c in idx])
    offC = np.array([np.sum(c) > 0 for c in clusters])
    onC = np.invert(offC)
    offCluster = [idx[i] for i in range(len(offC)) if offC[i]]
    onCluster = [idx[i] for i in range(len(onC)) if onC[i]]

    offSize = size[offC]
    onSize = size[onC]

    missStart = np.array([c[0] for c in offCluster], dtype=int)
    missEnd = np.array([c[-1] for c in offCluster], dtype=int)

    dataStart = np.array([c[0] for c in onCluster], dtype=int)
    dataEnd = np.array([c[-1] for c in onCluster], dtype=int)

    return missStart, missEnd, dataStart, dataEnd, onSize, offSize, onCluster, offCluster


# =============================================================================
# Interpolation internal_helpers
# =============================================================================
def findInterpWins(xpos, ypos, missing, window_time, edge_samples, freq, max_disp):
    """
    Description

    Parameters
    ----------

    Returns
    -------

    """
    # get indices of where missing intervals start and end
    missStart, missEnd = bool2bounds(missing)
    dataStart, dataEnd = bool2bounds(np.invert(missing))
    # missStart, missEnd, dataStart, dataEnd, onSize, offSize, onCluster, offCluster = getCluster(missing)

    # Determine window-samples
    window_samples = round(window_time / (1. / freq))

    # for each candidate, check if have enough valid data at edges to execute
    # interpolation. If not, see if merging with adjacent missing is possible
    # we don't throw out anything we can't deal with yet, we do that below.
    # this is just some preprocessing
    k = 0  # was K=1 in matlab
    while k < len(missStart) - 1:
        # skip if too long
        if missEnd[k] - missStart[k] + 1 > window_samples:
            k = k + 1
            continue

        # skip if not enough data at left edge
        if np.sum(dataEnd == missStart[k] - 1) > 0:
            datk = int(np.argwhere(dataEnd == missStart[k] - 1))
            if dataEnd[datk] - dataStart[datk] + 1 < edge_samples:
                k = k + 1
                continue

        # if not enough data at right edge, merge with next. Having not enough
        # on right edge of this one, means not having enough at left edge of
        # next. So both will be excluded always if we don't do anything. So we
        # can just merge without further checks. Its ok if it then grows too
        # long, as we'll just end up excluding that too below, which is what
        # would have happened if we didn't do anything here
        datk = np.argwhere(dataStart == missEnd[k] + 1)
        if len(datk) > 0:
            datk = int(datk)
            if dataEnd[datk] - dataStart[datk] + 1 < edge_samples:
                missEnd = np.delete(missEnd, k)
                missStart = np.delete(missStart, k + 1)

                # don't advance k so we check this one again and grow it further if
                # needed
                continue

        # nothing left to do, continue to next
        k = k + 1

    # mark intervals that are too long to be deleted (only delete later so that
    # below checks can use all missing on and offsets)
    missDur = missEnd - missStart + 1
    qRemove = missDur > window_samples

    # for each candidate, check if have enough valid data at edges to execute
    # interpolation and check displacement during missing wasn't too large.
    # Mark for later removal as multiple missing close together may otherwise
    # be wrongly allowed
    for p in range(len(missStart)):
        # check enough valid data at edges
        # missing too close to beginning of data
        # previous missing too close
        # missing too close to end of data
        # next missing too close
        if missStart[p] < edge_samples + 1 or \
                (p > 0 and missEnd[p - 1] > missStart[p] - edge_samples - 1) or \
                missEnd[p] > len(xpos) - 1 - edge_samples or \
                (p < len(missStart) - 1 and missStart[p + 1] < missEnd[p] + edge_samples + 1):
            qRemove[p] = True
            continue

        # check displacement, per missing interval
        # we want to check per bit of missing, even if multiple bits got merged
        # this as single data points can still anchor where the interpolation
        # goes and we thus need to check distance per bit, not over the whole
        # merged bit
        idx = np.arange(missStart[p], missEnd[p] + 1, dtype=int)
        on, off = bool2bounds(np.isnan(xpos[idx]))
        for q in range(len(on)):
            le_samps = np.array(on[q] - np.arange(edge_samples) + missStart[p] - 1, dtype=int)
            re_samps = np.array(off[q] + np.arange(edge_samples) + missStart[p] + 1, dtype=int)
            displacement = np.hypot(np.nanmedian(xpos[re_samps]) - np.nanmedian(xpos[le_samps]),
                                    np.nanmedian(ypos[re_samps]) - np.nanmedian(ypos[le_samps]))
            if displacement > max_disp:
                qRemove[p] = True
                break

        if qRemove[p]:
            continue

    # Remove the missing clusters which cannot be interpolated
    qRemove = np.where(qRemove)[0]
    missStart = np.delete(missStart, qRemove)
    missEnd = np.delete(missEnd, qRemove)

    # update missing vector
    notAllowed = missing.copy()
    for s, e in zip(missStart, missEnd):
        notAllowed[range(s, e + 1)] = False

    return missStart, missEnd


def windowedInterpolate(xpos, ypos, missing, missStart, missEnd, edge_samples):
    """
    Interpolates the missing data, and removes areas which are not allowed
    to be interpolated

    Parameters
    ----------


    Returns
    -------


    """
    missing = copy.deepcopy(missing)

    # Do the interpolating
    for p in range(len(missStart)):
        # make vector of all samples in this window
        outWin = np.arange(missStart[p], missEnd[p] + 1)

        # get edge samples: where no missing data was observed
        # also get samples in window where data was observed
        outWinNotMissing = np.invert(missing[outWin])
        valid_samps = np.concatenate((outWin[0] + np.arange(-edge_samples, 0), outWin[outWinNotMissing],
                                      outWin[-1] + np.arange(1, edge_samples + 1)))

        # get valid values: where no missing data was observed
        valid_x = xpos[valid_samps]
        valid_y = ypos[valid_samps]

        # do Steffen interpolation, update xpos, ypos
        xpos[outWin] = steffenInterp(valid_samps, valid_x, outWin)
        ypos[outWin] = steffenInterp(valid_samps, valid_y, outWin)

        # update missing: hole is now plugged
        missing[outWin] = False

    # plot interpolated data before (TODO, we didn't update this...)
    # if dev:
    #     f, [ax1, ax2] = plt.subplots(2, 1)
    #     ax1.plot(newX, xi, 'k-')
    #     ax1.scatter(newX[notMissing], xpos[notMissing], s=2, color='r')
    #     ax1.scatter(newX[missing], xi[missing], s=25, color='b')
    #     ax2.plot(newX, yi, 'k-')
    #     ax2.scatter(newX[notMissing], ypos[notMissing], s=2, color='r')
    #     ax2.scatter(newX[missing], yi[missing], s=25, color='b')

    return xpos, ypos, missing


# =============================================================================
# interpolator
# =============================================================================
def steffenInterp(x, y, xi):
    """
     STEFFEN 1-D Steffen interpolation
        steffenInterp[X,Y,XI] interpolates to find YI, the values of the
        underlying function Y at the points in the array XI, using
        the method of Steffen.  X and Y must be vectors of length N.

        Steffen's method is based on a third-order polynomial.  The
        slope at each grid point is calculated in a way to guarantee
        a monotonic behavior of the interpolating function.  The
        curve is smooth up to the first derivative.


     M. Steffen
     A Simple Method for Monotonic Interpolation in One Dimension
     Astron. Astrophys. 239, 443-450 [1990]
    """

    n = len(x)

    # calculate slopes
    yp = np.zeros(n)

    # first point
    h1 = x[1] - x[0]
    h2 = x[2] - x[1]
    s1 = (y[1] - y[0]) / h1
    s2 = (y[2] - y[1]) / h2
    p1 = s1 * (1 + h1 / (h1 + h2)) - s2 * h1 / (h1 + h2)
    if p1 * s1 <= 0:
        yp[0] = 0
    elif np.abs(p1) > 2 * np.abs(s1):
        yp[0] = 2 * s1
    else:
        yp[0] = p1

    # inner points
    for i in range(1, n - 1):
        hi = x[i + 1] - x[i]
        him1 = x[i] - x[i - 1]
        si = (y[i + 1] - y[i]) / hi
        sim1 = (y[i] - y[i - 1]) / him1
        pi = (sim1 * hi + si * him1) / (him1 + hi)

        if sim1 * si <= 0:
            yp[i] = 0
        elif (np.abs(pi) > 2 * np.abs(sim1)) or (np.abs(pi) > 2 * np.abs(si)):
            a = np.sign(sim1)
            yp[i] = 2 * a * np.min([np.abs(sim1), np.abs(si)])
        else:
            yp[i] = pi

    # last point
    hnm1 = x[n - 1] - x[n - 2]
    hnm2 = x[n - 2] - x[n - 3]
    snm1 = (y[n - 1] - y[n - 2]) / hnm1
    snm2 = (y[n - 2] - y[n - 3]) / hnm2
    pn = snm1 * (1 + hnm1 / (hnm1 + hnm2)) - snm2 * hnm1 / (hnm1 + hnm2)
    if pn * snm1 <= 0:
        yp[n - 1] = 0
    elif np.abs(pn) > 2 * np.abs(snm1):
        yp[n - 1] = 2 * snm1
    else:
        yp[n - 1] = pn

    yi = np.zeros(xi.size)
    for i in range(len(xi)):
        # Find the right place in the table by means of a bisection.
        # do this instead of search with find as the below now somehow gets
        # better optimized by matlab's JIT [runs twice as fast].
        klo = 1
        khi = n
        while khi - klo > 1:
            k = int(np.fix((khi + klo) / 2.0))
            if x[k] > xi[i]:
                khi = k
            else:
                klo = k

        # check if requested output is in input, so we can just copy
        if xi[i] == x[klo]:
            yi[i] = y[klo]
            continue
        elif xi[i] == x[khi]:
            yi[i] = y[khi]
            continue

        h = x[khi] - x[klo]
        s = (y[khi] - y[klo]) / h

        a = (yp[klo] + yp[khi] - 2 * s) / h / h
        b = (3 * s - 2 * yp[klo] - yp[khi]) / h
        c = yp[klo]
        d = y[klo]

        t = xi[i] - x[klo]
        # Use Horner's scheme for efficient evaluation of polynomials
        # y = a*t*t*t + b*t*t + c*t + d
        yi[i] = d + t * (c + t * (b + t * a))

    return yi


# =============================================================================
# Clustering internal_helpers
# use scipy k means
# =============================================================================
def kmeans2(data):
    # n points in p dimensional space
    n = data.shape[0]

    max_iterations = 100

    # initialize using kmeans++ method.
    # code taken and slightly edited from scipy.cluster.vq
    dims = data.shape[1] if len(data.shape) > 1 else 1
    C = np.ndarray((2, dims))

    # first cluster
    C[0, :] = data[np.random.randint(data.shape[0])]

    # second cluster
    D = cdist(C[:1, :], data, metric='sqeuclidean').min(axis=0)
    probs = D / D.sum()
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

    # Begin phase one:  batch reassignments
    # -----------------------------------------------------
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
        if np.any(m == 0):
            i = np.argwhere(m == 0)
            d = D[[label], [range(n)]]  # use newly updated distances

            # Find the point furthest away from its current cluster.
            # Take that point out of its cluster and use it to create
            # a new singleton cluster to replace the empty one.
            lonely = np.argmax(d)
            cFrom = label[lonely]  # taking from this cluster
            if m[cFrom] < 2:
                # In the very unusual event that the cluster had only
                # one member, pick any other non-singleton point.
                cFrom = np.argwhere(m > 1)[0]
                lonely = np.argwhere(label == cFrom)[0]
            label[lonely] = i

            # Update clusters from which points are taken
            C = _vq.update_cluster_means(data, label, 2)[0]
            m = np.bincount(label)
            D = cdist(C, data, metric='sqeuclidean')

        # Compute the total sum of distances for the current configuration.
        totsumD = np.sum(D[[label], [range(n)]])
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
            moved[np.bitwise_and(moved, D[0, :] == D[1, :])] = False
        if not np.any(moved):
            break
        label = new_label
        # update centers
        C = _vq.update_cluster_means(data, label, 2)[0]
        m = np.bincount(label)

    # ------------------------------------------------------------------
    # Begin phase two:  single reassignments
    # ------------------------------------------------------------------
    last_moved = -1
    converged = False
    while iter < max_iterations:
        # Calculate distances to each cluster from each point, and the
        # potential change in total sum of errors for adding or removing
        # each point from each cluster.  Clusters that have not changed
        # membership need not be updated.

        # Singleton clusters are a special case for the sum of dists
        # calculation. Removing their only point is never best, so the
        # reassignment criterion had better guarantee that a singleton
        # point will stay in its own cluster. Happily, we get
        # Del(i,idx(i)) == 0 automatically for them.
        Del = cdist(C, data, metric='sqeuclidean')
        mbrs = label == 0
        sgn = 1 - 2 * mbrs  # -1 for members, 1 for nonmembers
        if m[0] == 1:
            sgn[mbrs] = 0  # prevent divide-by-zero for singleton mbrs
        Del[0, :] = (m[0] / (m[0] + sgn)) * Del[0, :]
        # same for cluster 2
        sgn = -sgn  # -1 for members, 1 for nonmembers
        if m[1] == 1:
            sgn[np.invert(mbrs)] = 0  # prevent divide-by-zero for singleton mbrs
        Del[1, :] = (m[1] / (m[1] + sgn)) * Del[1, :]

        # Determine best possible move, if any, for each point.  Next we
        # will pick one from those that actually did move.
        prev_label = label
        new_label = (Del[1, :] < Del[0, :]).astype('int')
        moved = np.argwhere(prev_label != new_label)
        if moved.size > 0:
            # Resolve ties in favor of not moving
            moved = np.delete(moved, (Del[0, moved] == Del[1, moved]).flatten(), None)
        if moved.size == 0:
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
        totsumD = totsumD + Del[nlbl, moved] - Del[olbl, moved]

        # Update the cluster index vector, and the old and new cluster
        # counts and centroids
        label[moved] = nlbl
        m[nlbl] += 1
        m[olbl] -= 1
        C[nlbl, :] = C[nlbl, :] + (data[moved, :] - C[nlbl, :]) / m[nlbl]
        C[olbl, :] = C[olbl, :] - (data[moved, :] - C[olbl, :]) / m[olbl]

    # ------------------------------------------------------------------
    if not converged:
        raise RuntimeError('Failed to converge after %d iterations.' % iter)

    return label, C


def twoClusterWeighting(xpos, ypos, missing, downsamples, downsamp_filter, cheby_order, windowtime, steptime, freq,
                        max_errors, logging, logging_offset):
    """
    Description

    Parameters
    ----------

    Returns
    -------

    """
    # calculate number of samples of the moving window
    nrsamples = int(windowtime / (1. / freq))
    stepsize = np.max([1, int(steptime / (1. / freq))])

    # create empty weights vector
    totalweights = np.zeros(len(xpos))
    totalweights[missing] = np.nan
    nrtests = np.zeros(len(xpos))

    # stopped is always zero, unless max iterations is exceeded. this
    # indicates that file could not be analysed after trying for x iterations
    stopped = False
    counterrors = 0

    # Number of downsamples
    nd = len(downsamples)

    # Downsample
    if downsamp_filter:
        # filter signal. Follow the lead of decimate(), which first runs a
        # Chebyshev filter as specified below
        # passband ripple in dB
        rp = .05
        b = [[] for _i in range(nd)]
        a = [[] for _i in range(nd)]
        for p in range(nd):
            b[p], a[p] = scipy.signal.cheby1(cheby_order, rp, .8 / downsamples[p])

    # idx for downsamples
    indices = []
    for i in range(nd):
        indices.append(np.arange(nrsamples, 0, -downsamples[i], dtype=int)[::-1] - 1)

    # see where are missing in this data, for better running over the data
    # below.
    on, off = bool2bounds(missing)
    if on.size > 0:
        #  merge intervals smaller than nrsamples long
        merge = np.argwhere((on[1:] - off[:-1]) - 1 < nrsamples).flatten()
        for p in merge[::-1]:
            off[p] = off[p + 1]
            off = np.delete(off, p + 1)
            on = np.delete(on, p + 1)

        # check if intervals at data start and end are large enough
        if on[0] < nrsamples + 1:
            # not enough data point before first missing, so exclude them all
            on[0] = 0

        if off[-1] > (len(xpos) - 1 - nrsamples):
            # not enough data points after last missing, so exclude them all
            off[-1] = len(xpos) - 1

        # start at first non-missing sample if trial starts with missing (or
        # excluded because too short) data
        if on[0] == 0:
            i = off[0] + 1  # start at first non-missing
        else:
            i = 0
    else:
        i = 0

    eind = i + nrsamples
    while eind <= (len(xpos) - 1):
        # check if max errors is crossed
        if counterrors > max_errors:
            if logging:
                print(logging_offset + 'I2MC: Too many empty clusters encountered, aborting file. \n')
            stopped = True
            final_weights = np.nan
            return final_weights, stopped

        # select data portion of nrsamples
        idx = range(i, eind)
        ll_d = [[] for _p in range(nd + 1)]
        IDL_d = [[] for _p in range(nd + 1)]
        ll_d[0] = np.vstack([xpos[idx], ypos[idx]])

        # Filter the bit of data we're about to downsample. Then we simply need
        # to select each nth sample where n is the integer factor by which
        # number of samples is reduced. select samples such that they are till
        # end of window
        for p in range(nd):
            if downsamp_filter:
                ll_d[p + 1] = scipy.signal.filtfilt(b[p], a[p], ll_d[0])
                ll_d[p + 1] = ll_d[p + 1][:, indices[p]]
            else:
                ll_d[p + 1] = ll_d[0][:, indices[p]]

        # do 2-means clustering
        try:
            for p in range(nd + 1):
                IDL_d[p] = kmeans2(ll_d[p].T)[0]
        except Exception as e:
            if logging:
                print(logging_offset + 'I2MC: Unknown error encountered at sample {}.\n'.format(i))
            raise e

        # detect switches and weight of switch (= 1/number of switches in
        # portion)
        switches = [[] for _p in range(nd + 1)]
        switchesw = [[] for _p in range(nd + 1)]
        for p in range(nd + 1):
            switches[p] = np.abs(np.diff(IDL_d[p]))
            switchesw[p] = 1. / np.sum(switches[p])

        # get nearest samples of switch and add weight
        weighted = np.hstack([switches[0] * switchesw[0], 0])
        for p in range(nd):
            j = np.array((np.argwhere(switches[p + 1]).flatten() + 1) * downsamples[p], dtype=int) - 1
            for o in range(int(downsamples[p])):
                weighted[j + o] = weighted[j + o] + switchesw[p + 1]

        # add to totalweights
        totalweights[idx] = totalweights[idx] + weighted
        # record how many times each sample was tested
        nrtests[idx] = nrtests[idx] + 1

        # update i
        i += stepsize
        eind += stepsize
        missingOn = np.logical_and(on >= i, on <= eind)
        missingOff = np.logical_and(off >= i, off <= eind)
        qWhichMiss = np.logical_or(missingOn, missingOff)
        if np.sum(qWhichMiss) > 0:
            # we have some missing in this window. we don't process windows
            # with missing. Move back if we just skipped some samples, or else
            # skip whole missing and place start of window and first next
            # non-missing.
            if on[qWhichMiss][0] == (eind - stepsize):
                # continue at first non-missing
                i = off[qWhichMiss][0] + 1
            else:
                # we skipped some points, move window back so that we analyze
                # up to first next missing point
                i = on[qWhichMiss][0] - nrsamples
            eind = i + nrsamples

        if eind > len(xpos) - 1 > eind - stepsize:
            # we just exceeded data bound, but previous eind was before end of
            # data: we have some unprocessed samples. retreat just enough so we
            # process those end samples once
            d = eind - len(xpos) + 1
            eind = eind - d
            i = i - d

    # create final weights
    np.seterr(invalid='ignore')
    final_weights = totalweights / nrtests
    np.seterr(invalid='warn')
    return final_weights, stopped


# =============================================================================
# Fixation detection internal_helpers
# =============================================================================
def getFixations(final_weights, timestamp, xpos, ypos, missing, par):
    """
    Description

    Parameters
    ----------

    Returns
    -------

    """
    # Extract the required parameters
    cutoffstd = par['cutoffstd']
    onoffsetThresh = par['onoffsetThresh']
    maxMergeDist = par['maxMergeDist']
    maxMergeTime = par['maxMergeTime']
    minFixDur = par['minFixDur']

    # first determine cutoff for final weights
    cutoff = np.nanmean(final_weights) + cutoffstd * np.nanstd(final_weights, ddof=1)

    # get boolean of fixations
    fixbool = final_weights < cutoff

    # get indices of where fixations start and end
    fixstart, fixend = bool2bounds(fixbool)

    # for each fixation start, walk forward until recorded position is below
    # a threshold of lambda*MAD away from median fixation position.
    # same for each fixation end, but walk backward
    for p in range(len(fixstart)):
        xFix = xpos[fixstart[p]:fixend[p] + 1]
        yFix = ypos[fixstart[p]:fixend[p] + 1]
        xmedThis = np.nanmedian(xFix)
        ymedThis = np.nanmedian(yFix)

        # MAD = median(abs(x_i-median({x}))). For the 2D version, I'm using
        # median 2D distance of a point from the median fixation position. Not
        # exactly MAD, but makes more sense to me for 2D than city block,
        # especially given that we use 2D distance in our walk here
        MAD = np.nanmedian(np.hypot(xFix - xmedThis, yFix - ymedThis))
        thresh = MAD * onoffsetThresh

        # walk until distance less than threshold away from median fixation
        # position. No walking occurs when we're already below threshold.
        i = fixstart[p]
        # don't walk when fixation starting at start of data
        if i > 0:
            while np.hypot(xpos[i] - xmedThis, ypos[i] - ymedThis) > thresh:
                i = i + 1
            fixstart[p] = i

        # and now fixation end.
        i = fixend[p]
        # don't walk when fixation ending at end of data
        if i < len(xpos):
            while np.hypot(xpos[i] - xmedThis, ypos[i] - ymedThis) > thresh:
                i = i - 1
            fixend[p] = i

    # get start time, end time,
    starttime = timestamp[fixstart]
    endtime = timestamp[fixend]

    # loop over all fixation candidates in trial, see if should be merged
    for p in range(1, len(starttime))[::-1]:
        # get median coordinates of fixation
        xmedThis = np.median(xpos[fixstart[p]:fixend[p] + 1])
        ymedThis = np.median(ypos[fixstart[p]:fixend[p] + 1])
        xmedPrev = np.median(xpos[fixstart[p - 1]:fixend[p - 1] + 1])
        ymedPrev = np.median(ypos[fixstart[p - 1]:fixend[p - 1] + 1])

        # check if fixations close enough in time and space and thus qualify
        # for merging
        # The interval between the two fixations is calculated correctly (see
        # notes about fixation duration below), i checked this carefully. (Both
        # start and end of the interval are shifted by one sample in time, but
        # assuming practically constant sample interval, that's not an issue.)
        if starttime[p] - endtime[p - 1] < maxMergeTime \
                and np.hypot(xmedThis - xmedPrev, ymedThis - ymedPrev) < maxMergeDist:
            # merge
            fixend[p - 1] = fixend[p]
            endtime[p - 1] = endtime[p]
            # delete merged fixation
            fixstart = np.delete(fixstart, p)
            fixend = np.delete(fixend, p)
            starttime = np.delete(starttime, p)
            endtime = np.delete(endtime, p)

    # beginning and end of fixation must be real data, not interpolated.
    # If interpolated, those bit(s) at the edge(s) are excluded from the
    # fixation. First throw out fixations that are all missing/interpolated
    for p in range(len(starttime))[::-1]:
        miss = missing[fixstart[p]:fixend[p] + 1]
        if np.sum(miss) == len(miss):
            fixstart = np.delete(fixstart, p)
            fixend = np.delete(fixend, p)
            starttime = np.delete(starttime, p)
            endtime = np.delete(endtime, p)

    # then check edges and shrink if needed
    for p in range(len(starttime)):
        if missing[fixstart[p]]:
            fixstart[p] = fixstart[p] + np.argmax(np.invert(missing[fixstart[p]:fixend[p] + 1]))
            starttime[p] = timestamp[fixstart[p]]
        if missing[fixend[p]]:
            fixend[p] = fixend[p] - (np.argmax(np.invert(missing[fixstart[p]:fixend[p] + 1][::-1])) + 1)
            endtime[p] = timestamp[fixend[p]]

    # calculate fixation duration
    # if you calculate fixation duration by means of time of last sample during
    # fixation minus time of first sample during fixation (our fixation markers
    # are inclusive), then you always underestimate fixation duration by one
    # sample because you're in practice counting to the beginning of the
    # sample, not the end of it. To solve this, as end time we need to take the
    # timestamp of the sample that is one past the last sample of the fixation.
    # so, first calculate fixation duration by simple timestamp subtraction.
    fixdur = endtime - starttime

    # then determine what duration of this last sample was
    # make sure we don't run off the end of the data
    next_samp = np.min(np.vstack([fixend + 1, np.zeros(len(fixend), dtype=int) + len(timestamp) - 1]), axis=0)
    extratime = timestamp[next_samp] - timestamp[fixend]

    # if last fixation ends at end of data, we need to determine how long that
    # sample is and add that to the end time. Here we simply guess it as the
    # duration of previous sample
    # first check if there are fixations in the first place, or we'll index into non-existing data
    if not len(fixend) == 0 and fixend[-1] == len(timestamp):
        extratime[-1] = np.diff(timestamp[-3:-1])

    # now add the duration of the end sample to fixation durations, so we have
    # correct fixation durations
    fixdur = fixdur + extratime

    # check if any fixations are too short
    qTooShort = np.argwhere(fixdur < minFixDur)
    if len(qTooShort) > 0:
        fixstart = np.delete(fixstart, qTooShort)
        fixend = np.delete(fixend, qTooShort)
        starttime = np.delete(starttime, qTooShort)
        endtime = np.delete(endtime, qTooShort)
        fixdur = np.delete(fixdur, qTooShort)

    # process fixations, get other info about them
    # vector for median
    xmedian = np.zeros(fixstart.shape)
    # vector for median
    ymedian = np.zeros(fixstart.shape)
    # vector for whether fixation is flanked by data loss
    flankdataloss = np.zeros(fixstart.shape)
    # vector for fraction interpolated
    fracinterped = np.zeros(fixstart.shape)
    for a in range(len(fixstart)):
        idxs = range(fixstart[a], fixend[a] + 1)
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
        flankdataloss[a] = (fixstart[a] > 0 and missing[fixstart[a] - 1]) or (
                fixend[a] < len(xpos) - 1 and missing[fixend[a] + 1])

        # fraction of data loss during fixation that has been (does not count
        # data that is still lost)
        fracinterped[a] = np.sum(np.invert(np.isnan(xposf[qMiss]))) / (fixend[a] - fixstart[a] + 1)

    # store all the results in a dictionary
    fix = dict()
    fix['cutoff'] = cutoff
    fix['start'] = fixstart
    fix['end'] = fixend
    fix['startT'] = starttime
    fix['endT'] = endtime
    fix['dur'] = fixdur
    fix['xpos'] = xmedian
    fix['ypos'] = ymedian
    fix['flankdataloss'] = flankdataloss
    fix['fracinterped'] = fracinterped
    return fix


def getFixStats(xpos, ypos, missing, pixel_per_deg=None, fix=None):
    """
    Description

    Parameters
    ----------

    Returns
    -------

    """
    # Extract the required parameters
    if fix is None:
        fix = {}
    fstart = fix['start']
    fend = fix['end']

    # vectors for precision measures
    RMSxy = np.zeros(fstart.shape)
    BCEA = np.zeros(fstart.shape)
    rangeX = np.zeros(fstart.shape)
    rangeY = np.zeros(fstart.shape)

    for a in range(len(fstart)):
        idxs = range(fstart[a], fend[a] + 1)
        # get data during fixation
        xposf = xpos[idxs]
        yposf = ypos[idxs]
        # for all calculations below we'll only use data that is not
        # interpolated, so only real data
        qMiss = missing[idxs]

        # calculate RMS
        # since its done with diff, don't just exclude missing and treat
        # resulting as one continuous vector. replace missing with nan first,
        # use left-over values
        # Difference x position
        xdif = xposf.copy()
        xdif[qMiss] = np.nan
        xdif = np.diff(xdif) ** 2
        xdif = xdif[np.invert(np.isnan(xdif))]
        # Difference y position
        ydif = yposf.copy()
        ydif[qMiss] = np.nan
        ydif = np.diff(ydif) ** 2
        ydif = ydif[np.invert(np.isnan(ydif))]
        # Distance and RMS measure
        c = xdif + ydif  # 2D sample-to-sample displacement value in pixels
        RMSxy[a] = np.sqrt(np.mean(c))
        if pixel_per_deg is not None:
            # value in degrees visual angle
            RMSxy[a] = RMSxy[a] / pixel_per_deg

            # calculate BCEA (Crossland and Rubin 2002 Optometry and Vision Science)
        stdx = np.std(xposf[np.invert(qMiss)], ddof=1)
        stdy = np.std(yposf[np.invert(qMiss)], ddof=1)
        if pixel_per_deg is not None:
            # value in degrees visual angle
            stdx = stdx / pixel_per_deg
            stdy = stdy / pixel_per_deg

        if len(yposf[np.invert(qMiss)]) < 2:
            BCEA[a] = np.nan
        else:
            xx = np.corrcoef(xposf[np.invert(qMiss)], yposf[np.invert(qMiss)])
            rho = xx[0, 1]
            # cumulative probability of area under the multivariate normal
            P = 0.68
            k = np.log(1. / (1 - P))
            BCEA[a] = 2 * k * np.pi * stdx * stdy * np.sqrt(1 - rho ** 2)

        # calculate max-min of fixation
        if np.sum(qMiss) == len(qMiss):
            rangeX[a] = np.nan
            rangeY[a] = np.nan
        else:
            rangeX[a] = (np.max(xposf[np.invert(qMiss)]) - np.min(xposf[np.invert(qMiss)]))
            rangeY[a] = (np.max(yposf[np.invert(qMiss)]) - np.min(yposf[np.invert(qMiss)]))

        if pixel_per_deg is not None:
            # value in degrees visual angle
            rangeX[a] = rangeX[a] / pixel_per_deg
            rangeY[a] = rangeY[a] / pixel_per_deg

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
def I2MC(gazeData, options=None, logging=True, logging_offset=""):
    """


    Parameters
    ----------
    @param gazeData: a dataframe containing the gaze data
    @param gazeData: a dataframe containing the gaze data
    @param options: a dictionary containing the options for the I2MC analysis
    @param logging: boolean indicating whether to log the results
    @param logging_offset: offset before every logging message

    Returns
    -------
    @return: a dataframe containing the I2MC analysis results (fixations)

    """
    # set defaults
    if options is None:
        options = {}
    data = gazeData.copy()
    opt = options.copy()
    par = {}

    # Check required parameters
    checkFun('xres', opt, 'horizontal screen resolution')
    checkFun('yres', opt, 'vertical screen resolution')
    checkFun('freq', opt, 'tracker sampling rate')
    checkFun('missingx', opt, 'value indicating data loss for horizontal position')
    checkFun('missingy', opt, 'value indicating data loss for vertical position')

    # required parameters:
    par['xres'] = opt.pop('xres')
    par['yres'] = opt.pop('yres')
    par['freq'] = opt.pop('freq')
    par['missingx'] = opt.pop('missingx')
    par['missingy'] = opt.pop('missingy')

    # screen size (e.g. in cm). Optional, specify if want fixation statistics in deg
    par['scrSz'] = opt.pop('scrSz', None)

    # screen distance (in same unit as size). Optional, specify if want fixation statistics in deg
    par['disttoscreen'] = opt.pop('disttoscreen', None)

    # parameters with defaults:
    # CUBIC SPLINE INTERPOLATION
    # max duration (s) of missing values for interpolation to occur
    par['windowtimeInterp'] = opt.pop('windowtimeInterp', .1)
    # amount of data (number of samples) at edges needed for interpolation
    par['edgeSampInterp'] = opt.pop('edgeSampInterp', 2)
    # maximum displacement during missing for interpolation to be possible. Default set below if needed
    par['maxdisp'] = opt.pop('maxdisp', None)

    # K-MEANS CLUSTERING
    # time window (s) over which to calculate 2-means clustering (choose value so that max. 1 saccade can occur)
    par['windowtime'] = opt.pop('windowtime', .2)
    # time window shift (s) for each iteration. Use zero for sample by sample processing
    par['steptime'] = opt.pop('steptime', .02)
    # downsample levels (can be empty)
    par['downsamples'] = opt.pop('downsamples', [2, 5, 10])
    # use chebyshev filter when downscaling?
    # True: yes,
    # False: no.
    # requires signal processing toolbox. is what matlab's downsampling internal_helpers do,
    # but could cause trouble (ringing) with the hard edges in
    # eye-movement data
    par['downsampFilter'] = opt.pop('downsampFilter', True)

    # order of cheby1 Chebyshev downsampling filter, default is normally ok, as long as there are 25 or more samples
    # in the window (you may have less if your data is of low sampling rate or your window is small
    par['chebyOrder'] = opt.pop('chebyOrder', 8)

    # maximum number of errors allowed in k-means clustering procedure before proceeding to next file
    par['maxerrors'] = opt.pop('maxerrors', 100)
    # FIXATION DETERMINATION
    # number of standard deviations above mean k-means weights will be used as fixation cutoff
    par['cutoffstd'] = opt.pop('cutoffstd', 2.)
    # number of MAD away from median fixation duration. Will be used to walk forward at fixation starts and backward
    # at fixation ends to refine their placement and stop algorithm from eating into saccades
    par['onoffsetThresh'] = opt.pop('onoffsetThresh', 3.)
    # maximum Euclidean distance in pixels between fixations for merging
    par['maxMergeDist'] = opt.pop('maxMergeDist', 30.)
    # maximum time in ms between fixations for merging
    par['maxMergeTime'] = opt.pop('maxMergeTime', 30.)
    # minimum fixation duration (ms) after merging, fixations with shorter duration are removed from output
    par['minFixDur'] = opt.pop('minFixDur', 40.)

    # Development parameters (plotting intermediate steps), Change these to False when not developing
    par['dev_interpolation'] = opt.pop('dev_interpolation', False)
    par['dev_cluster'] = opt.pop('dev_cluster', False)
    par['skip_inputhandeling'] = opt.pop('skip_inputhandeling', False)

    for key in opt:
        raise Exception('Key "{}" not recognized'.format(key))

    # =============================================================================
    # # Input handling and checking
    # =============================================================================
    # loop over input
    if not par['skip_inputhandeling']:
        for key, value in par.items():
            if key in ['xres', 'yres', 'freq', 'missingx', 'missingy', 'windowtimeInterp', 'maxdisp', 'windowtime',
                       'steptime', 'cutoffstd', 'onoffsetThresh', 'maxMergeDist', 'maxMergeTime', 'minFixDur']:
                checkNumeric(key, value)
                checkScalar(key, value)
            elif key == 'disttoscreen':
                if value is not None:  # may be none (its an optional parameter)
                    checkNumeric(key, value)
                    checkScalar(key, value)
            elif key in ['downsampFilter', 'chebyOrder', 'maxerrors', 'edgeSampInterp']:
                checkIntegers(key, value)
                checkScalar(key, value)
            elif key == 'scrSz':
                if value is not None:  # may be none (its an optional parameter)
                    checkNumeric(key, value)
                    check2Vector(key, value)
            elif key == 'downsamples':
                checkIntegers(key, value)
            else:
                if type(key) != str:
                    raise Exception('Key "{}" not recognized'.format(key))

    # set defaults
    if par['maxdisp'] is None:
        par['maxdisp'] = par['xres'] * 0.2 * np.sqrt(2)

    # check filter
    if 'downsampFilter' in par.keys():
        nSampRequired = np.max([1, 3 * par['chebyOrder']]) + 1
        nSampInWin = round(par['windowtime'] / (1. / par['freq']))
        if nSampInWin < nSampRequired:
            raise Exception('I2MC: Filter parameters requested with the setting "chebyOrder"' +
                            'will not work for the sampling frequency of your data. Please lower "chebyOrder",' +
                            'or set the setting "downsampFilter" to 0 ')

    # setup visual angle conversion
    pixel_per_deg = None
    if par['scrSz'] is not None and par['disttoscreen'] is not None:
        pixel_per_deg = angleToPixels(1, par['disttoscreen'], par['scrSz'][0], [par['xres'], par['yres']])

    # =============================================================================
    # Determine missing values and determine X and Y gaze pos
    # =============================================================================
    # deal with monocular data, or create average over two eyes
    if 'L_X' in data.keys() and 'R_X' not in data.keys():
        xpos = data['L_X'].array
        ypos = data['L_Y'].array
        # Check for missing values
        missingX = np.logical_or(np.isnan(xpos), xpos == par['missingx'])
        missingY = np.logical_or(np.isnan(ypos), ypos == par['missingy'])
        missing = np.logical_or(missingX, missingY)
        data['left_missing'] = missing
        q2Eyes = False

    elif 'R_X' in data.keys() and 'L_X' not in data.keys():
        xpos = data['R_X']
        ypos = data['R_Y']
        # Check for missing values
        missingX = np.logical_or(np.isnan(xpos), xpos == par['missingx'])
        missingY = np.logical_or(np.isnan(ypos), ypos == par['missingy'])
        missing = np.logical_or(missingX, missingY)
        data['right_missing'] = missing
        q2Eyes = False

    elif 'average_X' in data.keys():
        xpos = data['average_X']
        ypos = data['average_Y']
        missingX = np.logical_or(np.isnan(xpos), xpos == par['missingx'])
        missingY = np.logical_or(np.isnan(ypos), ypos == par['missingy'])
        missing = np.logical_or(missingX, missingY)
        data['average_missing'] = missing
        q2Eyes = 'R_X' in data.keys() and 'L_X' in data.keys()
        if q2Eyes:
            # we have left and right and average already provided, but we need
            # to get missing in the individual eye signals
            llmiss, rrmiss, bothmiss = getMissing(data['L_X'], data['R_X'], par['missingx'], data['L_Y'], data['R_Y'],
                                                  par['missingy'])
            data['left_missing'] = llmiss
            data['right_missing'] = rrmiss

    # we have left and right, average them
    else:
        data['average_X'], data['average_Y'], missing, llmiss, rrmiss = averageEyes(data['L_X'], data['R_X'],
                                                                                    par['missingx'], data['L_Y'],
                                                                                    data['R_Y'], par['missingy'])
        xpos = data['average_X']
        ypos = data['average_Y']
        data['average_missing'] = missing
        data['left_missing'] = llmiss
        data['right_missing'] = rrmiss
        q2Eyes = True

    xpos = xpos.array
    ypos = ypos.array
    missing = missing.array
    # =============================================================================
    # INTERPOLATION
    # =============================================================================
    # get interpolation windows for average and individual eye signals
    if logging:
        print(logging_offset + 'I2MC: Searching for valid interpolation windows')
    missStart, missEnd = findInterpWins(xpos, ypos, missing, par['windowtimeInterp'], par['edgeSampInterp'],
                                        par['freq'], par['maxdisp'])
    if q2Eyes:
        llmissStart, llmissEnd = findInterpWins(data['L_X'].array, data['L_Y'].array, llmiss, par['windowtimeInterp'],
                                                par['edgeSampInterp'], par['freq'], par['maxdisp'])
        rrmissStart, rrmissEnd = findInterpWins(data['R_X'].array, data['R_Y'].array, rrmiss, par['windowtimeInterp'],
                                                par['edgeSampInterp'], par['freq'], par['maxdisp'])

    # Use Steffen interpolation and replace values
    if logging:
        print(logging_offset + 'I2MC: Replace interpolation windows with Steffen interpolation')
    xpos, ypos, missingn = windowedInterpolate(xpos, ypos, missing, missStart, missEnd, par['edgeSampInterp'])
    if q2Eyes:
        llx, lly, llmissn = windowedInterpolate(data['L_X'].array, data['L_Y'].array, data['left_missing'].array,
                                                llmissStart, llmissEnd, par['edgeSampInterp'])
        rrx, rry, rrmissn = windowedInterpolate(data['R_X'].array, data['R_Y'].array, data['right_missing'].array,
                                                rrmissStart, rrmissEnd, par['edgeSampInterp'])

    # =============================================================================
    # 2-MEANS CLUSTERING
    # =============================================================================
    # CALCULATE 2-MEANS CLUSTERING FOR SINGLE EYE
    if not q2Eyes:
        # get kmeans-clustering for averaged signal
        if logging:
            print(logging_offset + 'I2MC: 2-Means clustering started for averaged signal')
        data['finalweights'], stopped = twoClusterWeighting(xpos, ypos, missingn, par['downsamples'],
                                                            par['downsampFilter'], par['chebyOrder'], par['windowtime'],
                                                            par['steptime'], par['freq'], par['maxerrors'],
                                                            logging, logging_offset)

        # check whether clustering succeeded
        if stopped:
            warnings.warn('I2MC: Clustering stopped early, probably due to too many errors')
            return False

    # CALCULATE 2-MEANS CLUSTERING FOR SEPARATE EYES
    elif q2Eyes:
        # get kmeans-clustering for left eye signal
        if logging:
            print(logging_offset + 'I2MC: 2-Means clustering started for left eye signal')
        finalweights_left, stopped = twoClusterWeighting(llx, lly, llmissn, par['downsamples'], par['downsampFilter'],
                                                         par['chebyOrder'], par['windowtime'], par['steptime'],
                                                         par['freq'], par['maxerrors'],
                                                         logging, logging_offset)

        # check whether clustering succeeded
        if stopped:
            if logging:
                print(logging_offset + 'I2MC: Clustering stopped after exceeding max errors, continuing to next file \n')
            raise Exception('Clustering stopped after exceeding max errors, continuing to next file')

        # get kmeans-clustering for right eye signal
        if logging:
            print(logging_offset + 'I2MC: 2-Means clustering started for right eye signal')
        finalweights_right, stopped = twoClusterWeighting(rrx, rry, rrmissn, par['downsamples'], par['downsampFilter'],
                                                          par['chebyOrder'], par['windowtime'], par['steptime'],
                                                          par['freq'], par['maxerrors'],
                                                          logging, logging_offset)

        # check whether clustering succeeded
        if stopped:
            if logging:
                print(logging_offset + 'I2MC: Clustering stopped after exceeding max errors, continuing to next file')
            raise Exception('Clustering stopped after exceeding max errors, continuing to next file')

        # AVERAGE FINALWEIGHTS OVER COMBINED & SEPARATE EYES
        # ignore all runtime warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data['finalweights'] = np.nanmean(np.vstack([finalweights_left, finalweights_right]), axis=0)

    # =============================================================================
    #  DETERMINE FIXATIONS BASED ON FINALWEIGHTS_AVG
    # =============================================================================
    if logging:
        print(logging_offset +
              'I2MC: Determining fixations based on clustering weight mean for averaged signal and separate eyes ' +
              '+ {:.2f}*std'.format(par['cutoffstd']))

    # finalweights to float
    data['finalweights'] = data['finalweights'].astype(float)
    # time to float
    data['time'] = data['time'].astype(float)

    fix = getFixations(data['finalweights'].array, data['time'].array, xpos, ypos, missing, par)
    fix = getFixStats(xpos, ypos, missing, pixel_per_deg, fix)



    return fix, data, par
