#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 14:41:08 2025

@author: justin
"""

import numpy as np

# Define the Gaussians and linear functions for each region
def gaussian(x, amp, cen, sigma):
    return amp * np.exp(-(x - cen)**2 / (2 * sigma**2))

def linear(x, slope, intercept):
    return slope * x + intercept

# Vectorized Gaussian computation
def sum_gaussians(x, params, num_gaussians):
    if num_gaussians == 0:
        return np.zeros_like(x)  # No Gaussians to compute
    amps, cens, sigmas = np.array(params[:num_gaussians]).T
    return np.sum(amps[:, None] * np.exp(-0.5 * ((x - cens[:, None]) / sigmas[:, None]) ** 2), axis=0)


def model_chooser(Nregions,Nlines):
    if Nregions == 1:
        if Nlines == 1:
            piecewise_model = piecewise_model_1region_1gaussian
        if Nlines == 2:
            piecewise_model = piecewise_model_1region_2gaussians
        if Nlines == 3:
            piecewise_model = piecewise_model_1region_3gaussians
        if Nlines == 4:
            piecewise_model = piecewise_model_1region_4gaussians           
        if Nlines == 5:
            piecewise_model = piecewise_model_1region_5gaussians     
        if Nlines == 6:
            piecewise_model = piecewise_model_1region_6gaussians                 

    if Nregions == 2:
        if Nlines == 1:
            pass
        if Nlines == 2:
            pass
        if Nlines == 3:
            piecewise_model = piecewise_model_2regions_3gaussians
            pass
        if Nlines == 4:
            pass
        if Nlines == 5:
            piecewise_model = piecewise_model_2regions_5gaussians
        if Nlines == 6:
            piecewise_model = piecewise_model_2regions_6gaussians
        if Nlines == 10:
            piecewise_model = piecewise_model_2regions_10gaussians

    if Nregions == 3:
        if Nlines == 1:
            pass
        if Nlines == 2:
            pass
        if Nlines == 3:
            pass
        if Nlines == 4:
            piecewise_model = piecewise_model_3regions_4gaussians
        if Nlines == 5:
            piecewise_model = piecewise_model_3regions_5gaussians
        if Nlines == 6:
            pass
        if Nlines == 10:
            pass
            
    return piecewise_model



def piecewise_model_1region_1gaussian(x, x1_start, x1_end,
                                        slope1, intercept1,
                                        amp1, cen1, sigma1,
                                        NR1):

    # Store Gaussian parameters
    gaussians = [(amp1, cen1, sigma1)]

    # Identify regions
    region1 = (x >= x1_start) & (x < x1_end)

    # Precompute Gaussians for each region
    R1 = slope1 * x[region1] + intercept1 + sum_gaussians(x[region1], gaussians[:NR1], NR1)

    # Assemble output
    y_fit = np.zeros_like(x)
    y_fit[region1] = R1

    return y_fit

def piecewise_model_1region_2gaussians(x, x1_start, x1_end,
                                        slope1, intercept1,
                                        amp1, cen1, sigma1,
                                        amp2, cen2, sigma2,
                                        NR1):

    # Store Gaussian parameters
    gaussians = [(amp1, cen1, sigma1),
                 (amp2, cen2, sigma2)]

    # Identify regions
    region1 = (x >= x1_start) & (x < x1_end)

    # Precompute Gaussians for each region
    R1 = slope1 * x[region1] + intercept1 + sum_gaussians(x[region1], gaussians[:NR1], NR1)

    # Assemble output
    y_fit = np.zeros_like(x)
    y_fit[region1] = R1

    return y_fit

def piecewise_model_1region_3gaussians(x, x1_start, x1_end,
                                        slope1, intercept1,
                                        amp1, cen1, sigma1,
                                        amp2, cen2, sigma2,
                                        amp3, cen3, sigma3,
                                        NR1):

    # Store Gaussian parameters
    gaussians = [(amp1, cen1, sigma1),
                 (amp2, cen2, sigma2),
                 (amp3, cen3, sigma3)]

    # Identify regions
    region1 = (x >= x1_start) & (x < x1_end)

    # Precompute Gaussians for each region
    R1 = slope1 * x[region1] + intercept1 + sum_gaussians(x[region1], gaussians[:NR1], NR1)

    # Assemble output
    y_fit = np.zeros_like(x)
    y_fit[region1] = R1

    return y_fit

def piecewise_model_1region_4gaussians(x, x1_start, x1_end,
                                        slope1, intercept1,
                                        amp1, cen1, sigma1,
                                        amp2, cen2, sigma2,
                                        amp3, cen3, sigma3,
                                        amp4, cen4, sigma4,
                                        NR1):

    # Store Gaussian parameters
    gaussians = [(amp1, cen1, sigma1),
                 (amp2, cen2, sigma2),
                 (amp3, cen3, sigma3),
                 (amp4, cen4, sigma4)
                 ]

    # Identify regions
    region1 = (x >= x1_start) & (x < x1_end)

    # Precompute Gaussians for each region
    R1 = slope1 * x[region1] + intercept1 + sum_gaussians(x[region1], gaussians[:NR1], NR1)

    # Assemble output
    y_fit = np.zeros_like(x)
    y_fit[region1] = R1

    return y_fit



def piecewise_model_1region_5gaussians(x, x1_start, x1_end,
                                        slope1, intercept1,
                                        amp1, cen1, sigma1,
                                        amp2, cen2, sigma2,
                                        amp3, cen3, sigma3,
                                        amp4, cen4, sigma4,
                                        amp5, cen5, sigma5,
                                        NR1):

    # Store Gaussian parameters
    gaussians = [(amp1, cen1, sigma1),
                 (amp2, cen2, sigma2),
                 (amp3, cen3, sigma3),
                 (amp4, cen4, sigma4),
                 (amp5, cen5, sigma5)
                 ]

    # Identify regions
    region1 = (x >= x1_start) & (x < x1_end)

    # Precompute Gaussians for each region
    R1 = slope1 * x[region1] + intercept1 + sum_gaussians(x[region1], gaussians[:NR1], NR1)

    # Assemble output
    y_fit = np.zeros_like(x)
    y_fit[region1] = R1

    return y_fit




def piecewise_model_1region_6gaussians(x, x1_start, x1_end,
                                        slope1, intercept1,
                                        amp1, cen1, sigma1,
                                        amp2, cen2, sigma2,
                                        amp3, cen3, sigma3,
                                        amp4, cen4, sigma4,
                                        amp5, cen5, sigma5,
                                        amp6, cen6, sigma6,
                                        NR1):

    # Store Gaussian parameters
    gaussians = [(amp1, cen1, sigma1),
                 (amp2, cen2, sigma2),
                 (amp3, cen3, sigma3),
                 (amp4, cen4, sigma4),
                 (amp5, cen5, sigma5),
                 (amp6, cen6, sigma6)
                 ]

    # Identify regions
    region1 = (x >= x1_start) & (x < x1_end)

    # Precompute Gaussians for each region
    R1 = slope1 * x[region1] + intercept1 + sum_gaussians(x[region1], gaussians[:NR1], NR1)

    # Assemble output
    y_fit = np.zeros_like(x)
    y_fit[region1] = R1

    return y_fit




#-----------------



def piecewise_model_2regions_3gaussians(x, x1_start, x1_end, x2_start, x2_end, x_int_1_start, x_int_1_end,
                                        slope1, intercept1, slope2, intercept2, slope_int_1, intercept_int_1,
                                        amp1, cen1, sigma1,
                                        amp2, cen2, sigma2,
                                        amp3, cen3, sigma3,
                                        NR1, NR2):

    # Store Gaussian parameters
    gaussians = [(amp1, cen1, sigma1), (amp2, cen2, sigma2), (amp3, cen3, sigma3)]

    # Identify regions
    region1 = (x >= x1_start) & (x < x1_end)
    region2 = (x >= x2_start) & (x < x2_end)
    intregion1 = (x >= x_int_1_start) & (x < x_int_1_end)

    # Precompute Gaussians for each region
    R1 = slope1 * x[region1] + intercept1 + sum_gaussians(x[region1], gaussians[:NR1], NR1)
    R2 = slope2 * x[region2] + intercept2 + sum_gaussians(x[region2], gaussians[NR1:], NR2)
    IR1 = slope_int_1 * x[intregion1] + intercept_int_1

    # Assemble output
    y_fit = np.zeros_like(x)
    y_fit[region1] = R1
    y_fit[region2] = R2
    y_fit[intregion1] = IR1

    return y_fit


def piecewise_model_2regions_5gaussians(x, x1_start, x1_end, x2_start, x2_end, x_int_1_start, x_int_1_end,
                                        slope1, intercept1, slope2, intercept2, slope_int_1, intercept_int_1,
                                        amp1, cen1, sigma1,
                                        amp2, cen2, sigma2,
                                        amp3, cen3, sigma3,
                                        amp4, cen4, sigma4,
                                        amp5, cen5, sigma5,
                                        NR1, NR2):

    # Store Gaussian parameters
    gaussians = [(amp1, cen1, sigma1), (amp2, cen2, sigma2), (amp3, cen3, sigma3), (amp4, cen4, sigma4), (amp5, cen5, sigma5)]

    # Identify regions
    region1 = (x >= x1_start) & (x < x1_end)
    region2 = (x >= x2_start) & (x < x2_end)
    intregion1 = (x >= x_int_1_start) & (x < x_int_1_end)

    # Precompute Gaussians for each region
    R1 = slope1 * x[region1] + intercept1 + sum_gaussians(x[region1], gaussians[:NR1], NR1)
    R2 = slope2 * x[region2] + intercept2 + sum_gaussians(x[region2], gaussians[NR1:], NR2)
    IR1 = slope_int_1 * x[intregion1] + intercept_int_1

    # Assemble output
    y_fit = np.zeros_like(x)
    y_fit[region1] = R1
    y_fit[region2] = R2
    y_fit[intregion1] = IR1

    return y_fit



def piecewise_model_2regions_6gaussians(x, x1_start, x1_end, x2_start, x2_end, x_int_1_start, x_int_1_end,
                                        slope1, intercept1, slope2, intercept2, slope_int_1, intercept_int_1,
                                        amp1, cen1, sigma1,
                                        amp2, cen2, sigma2,
                                        amp3, cen3, sigma3,
                                        amp4, cen4, sigma4,
                                        amp5, cen5, sigma5,
                                        amp6, cen6, sigma6,
                                        NR1, NR2):

    # Store Gaussian parameters
    gaussians = [(amp1, cen1, sigma1), 
                 (amp2, cen2, sigma2), 
                 (amp3, cen3, sigma3), 
                 (amp4, cen4, sigma4), 
                 (amp5, cen5, sigma5),
                 (amp6, cen6, sigma6)]

    # Identify regions
    region1 = (x >= x1_start) & (x < x1_end)
    region2 = (x >= x2_start) & (x < x2_end)
    intregion1 = (x >= x_int_1_start) & (x < x_int_1_end)

    # Precompute Gaussians for each region
    R1 = slope1 * x[region1] + intercept1 + sum_gaussians(x[region1], gaussians[:NR1], NR1)
    R2 = slope2 * x[region2] + intercept2 + sum_gaussians(x[region2], gaussians[NR1:], NR2)
    IR1 = slope_int_1 * x[intregion1] + intercept_int_1

    # Assemble output
    y_fit = np.zeros_like(x)
    y_fit[region1] = R1
    y_fit[region2] = R2
    y_fit[intregion1] = IR1

    return y_fit


def piecewise_model_2regions_10gaussians(x, x1_start, x1_end, x2_start, x2_end, x_int_1_start, x_int_1_end,
                                        slope1, intercept1, slope2, intercept2, slope_int_1, intercept_int_1,
                                        amp1, cen1, sigma1,
                                        amp2, cen2, sigma2,
                                        amp3, cen3, sigma3,
                                        amp4, cen4, sigma4,
                                        amp5, cen5, sigma5,
                                        amp6, cen6, sigma6,
                                        amp7, cen7, sigma7,
                                        amp8, cen8, sigma8,
                                        amp9, cen9, sigma9,
                                        amp10, cen10, sigma10,
                                        NR1, NR2):

    # Store Gaussian parameters
    gaussians = [(amp1, cen1, sigma1), 
                 (amp2, cen2, sigma2), 
                 (amp3, cen3, sigma3), 
                 (amp4, cen4, sigma4), 
                 (amp5, cen5, sigma5),
                 (amp6, cen6, sigma6),
                 (amp7, cen7, sigma7),
                 (amp8, cen8, sigma8),
                 (amp9, cen9, sigma9),
                 (amp10, cen10, sigma10)]

    # Identify regions
    region1 = (x >= x1_start) & (x < x1_end)
    region2 = (x >= x2_start) & (x < x2_end)
    intregion1 = (x >= x_int_1_start) & (x < x_int_1_end)

    # Precompute Gaussians for each region
    R1 = slope1 * x[region1] + intercept1 + sum_gaussians(x[region1], gaussians[:NR1], NR1)
    R2 = slope2 * x[region2] + intercept2 + sum_gaussians(x[region2], gaussians[NR1:], NR2)
    IR1 = slope_int_1 * x[intregion1] + intercept_int_1

    # Assemble output
    y_fit = np.zeros_like(x)
    y_fit[region1] = R1
    y_fit[region2] = R2
    y_fit[intregion1] = IR1

    return y_fit


def piecewise_model_3regions_4gaussians(x, 
                                        x1_start, x1_end, x2_start, x2_end, x3_start, x3_end, 
                                        slope1, intercept1, slope2, intercept2, slope3, intercept3,
                                        amp1, cen1, sigma1,
                                        amp2, cen2, sigma2,
                                        amp3, cen3, sigma3,
                                        amp4, cen4, sigma4,
                                        NR1, NR2, NR3):

    # Store Gaussian parameters
    gaussians = [(amp1, cen1, sigma1), (amp2, cen2, sigma2), (amp3, cen3, sigma3), 
                 (amp4, cen4, sigma4)]

    # Identify regions
    region1 = (x >= x1_start) & (x < x1_end)
    region2 = (x >= x2_start) & (x < x2_end)
    region3 = (x >= x3_start) & (x < x3_end)

    # Precompute Gaussians for each region
    R1 = slope1 * x[region1] + intercept1 + sum_gaussians(x[region1], gaussians[:NR1], NR1)
    R2 = slope2 * x[region2] + intercept2
    R3 = slope3 * x[region3] + intercept3 + sum_gaussians(x[region3], gaussians[NR1:], NR2)

    # Assemble output
    y_fit = np.zeros_like(x)
    y_fit[region1] = R1
    y_fit[region2] = R2
    y_fit[region3] = R3

    return y_fit



def piecewise_model_3regions_5gaussians(x, 
                                        x1_start, x1_end, x2_start, x2_end, x3_start, x3_end, 
                                        slope1, intercept1, slope2, intercept2, slope3, intercept3,
                                        amp1, cen1, sigma1,
                                        amp2, cen2, sigma2,
                                        amp3, cen3, sigma3,
                                        amp4, cen4, sigma4,
                                        amp5, cen5, sigma5,
                                        NR1, NR2, NR3):

    # Store Gaussian parameters
    gaussians = [(amp1, cen1, sigma1), (amp2, cen2, sigma2), (amp3, cen3, sigma3), 
                 (amp4, cen4, sigma4), (amp5, cen5, sigma5)]

    # Identify regions
    region1 = (x >= x1_start) & (x < x1_end)
    region2 = (x >= x2_start) & (x < x2_end)
    region3 = (x >= x3_start) & (x < x3_end)

    # Precompute Gaussians for each region
    R1 = slope1 * x[region1] + intercept1 + sum_gaussians(x[region1], gaussians[:NR1], NR1)
    R2 = slope2 * x[region2] + intercept2
    R3 = slope3 * x[region3] + intercept3 + sum_gaussians(x[region3], gaussians[NR1:], NR2)

    # Assemble output
    y_fit = np.zeros_like(x)
    y_fit[region1] = R1
    y_fit[region2] = R2
    y_fit[region3] = R3

    return y_fit


def piecewise_model_4regions_5gaussians(x, 
                                        x1_start, x1_end, x2_start, x2_end, x3_start, x3_end, x4_start, x4_end,
                                        slope1, intercept1, slope2, intercept2, slope3, intercept3, slope4, intercept4,
                                        amp1, cen1, sigma1,
                                        amp2, cen2, sigma2,
                                        amp3, cen3, sigma3,
                                        amp4, cen4, sigma4,
                                        amp5, cen5, sigma5,
                                        NR1, NR2, NR3, NR4):

    # Store Gaussian parameters
    gaussians = [(amp1, cen1, sigma1), (amp2, cen2, sigma2), (amp3, cen3, sigma3), 
                 (amp4, cen4, sigma4), (amp5, cen5, sigma5)]

    # Identify regions
    region1 = (x >= x1_start) & (x < x1_end)
    region2 = (x >= x2_start) & (x < x2_end)
    region3 = (x >= x3_start) & (x < x3_end)
    region4 = (x >= x4_start) & (x < x4_end)

    # Precompute Gaussians for each region
    R1 = slope1 * x[region1] + intercept1 + sum_gaussians(x[region1], gaussians[:NR1], NR1)
    R2 = slope2 * x[region2] + intercept2
    R3 = slope3 * x[region3] + intercept3 + sum_gaussians(x[region3], gaussians[NR1:], NR2)
    R4 = slope4 * x[region4] + intercept4 + sum_gaussians(x[region4], gaussians[NR2+NR3:], NR4)

    # Assemble output
    y_fit = np.zeros_like(x)
    y_fit[region1] = R1
    y_fit[region2] = R2
    y_fit[region3] = R3
    y_fit[region4] = R4

    return y_fit


def piecewise_model_5regions_5gaussians(x, 
                                        x1_start, x1_end, x2_start, x2_end, x3_start, x3_end, x4_start, x4_end, x5_start, x5_end,
                                        slope1, intercept1, slope2, intercept2, slope3, intercept3, slope4, intercept4, slope5, intercept5,
                                        amp1, cen1, sigma1,
                                        amp2, cen2, sigma2,
                                        amp3, cen3, sigma3,
                                        amp4, cen4, sigma4,
                                        amp5, cen5, sigma5,
                                        NR1, NR2, NR3, NR4, NR5):

    # Store Gaussian parameters
    gaussians = [(amp1, cen1, sigma1), (amp2, cen2, sigma2), (amp3, cen3, sigma3), 
                 (amp4, cen4, sigma4), (amp5, cen5, sigma5)]

    # Identify regions
    region1 = (x >= x1_start) & (x < x1_end)
    region2 = (x >= x2_start) & (x < x2_end)
    region3 = (x >= x3_start) & (x < x3_end)
    region4 = (x >= x4_start) & (x < x4_end)
    region5 = (x >= x5_start) & (x < x5_end)

    # Precompute Gaussians for each region
    R1 = slope1 * x[region1] + intercept1 + sum_gaussians(x[region1], gaussians[:NR1], NR1)
    R2 = slope2 * x[region2] + intercept2
    R3 = slope3 * x[region3] + intercept3 + sum_gaussians(x[region3], gaussians[NR1:], NR2)
    R4 = slope4 * x[region4] + intercept4 + sum_gaussians(x[region4], gaussians[NR2+NR3:], NR4)
    R5 = slope5 * x[region5] + intercept5 + sum_gaussians(x[region5], gaussians[NR3+NR4:], NR5)

    # Assemble output
    y_fit = np.zeros_like(x)
    y_fit[region1] = R1
    y_fit[region2] = R2
    y_fit[region3] = R3
    y_fit[region4] = R4
    y_fit[region5] = R5

    return y_fit

#---------



