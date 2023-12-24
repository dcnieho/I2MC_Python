[![Downloads](https://static.pepy.tech/badge/i2mc)](https://pepy.tech/project/i2mc)
[![Citation Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapi.juleskreuer.eu%2Fcitation-badge.php%3Fshield%26doi%3D10.3758%2Fs13428-016-0822-1&color=blue)](https://scholar.google.com/citations?view_op=view_citation&citation_for_view=uRUYoVgAAAAJ:KlAtU1dfN6UC)
[![PyPI Latest Release](https://img.shields.io/pypi/v/i2mc.svg)](https://pypi.org/project/i2mc/)
[![image](https://img.shields.io/pypi/pyversions/i2mc.svg)](https://pypi.org/project/i2mc/)
[![DOI](https://zenodo.org/badge/DOI/10.3758/s13428-016-0822-1.svg)](https://doi.org/10.3758/s13428-016-0822-1)
# Noise-robust fixation classification (I2MC) - Python implementation

## About I2MC

Original description of the MATLAB implementation from https://github.com/royhessels/I2MC:

The I2MC algorithm was designed to accomplish fixation classification in data across a wide range of noise levels and when periods of data loss may occur.

Cite as:
[Hessels, R.S., Niehorster, D.C., Kemner, C., & Hooge, I.T.C. (2017). Noise-robust fixation detection in eye-movement data - Identification by 2-means clustering (I2MC). Behavior Research Methods, 49(5): 1802–1823. doi: 10.3758/s13428-016-0822-1](https://link.springer.com/article/10.3758/s13428-016-0822-1)

For more information, questions, or to check whether we have updated to a better version, e-mail: royhessels@gmail.com / dcnieho@gmail.com. I2MC is available from www.github.com/royhessels/I2MC

Most parts of the I2MC algorithm are licensed under the Creative Commons Attribution 4.0 (CC BY 4.0) license. Some functions are under MIT license, and some may be under other licenses.

## About this implementation

Install using:
```
pip install I2MC
```
or
```
python -m pip install I2MC
```

This is a Python implementation of the I2MC algorithm (tested on version Python 3.8). For the original Matlab implementation see https://github.com/royhessels/I2MC

Most functions were initially ported to Python by Jonathan van Leeuwen. Diederick Niehorster and Roy Hessels updated a few functions to match the MATLAB version more closely and ported what remained.

Note that this Python implementation may be slightly different from the MATLAB version. Among the differences is a different implementation of Chebyshev filtering (the Python version uses `scipy.signal.cheby1()`, the MATLAB version uses `cheby1` from the signal processing toolbox).

The differences in output between the MATLAB and Python implementation of I2MC are visualized in Figure 1 and 2. As is visible from these Figures, slight differences in output between the MATLAB and Python implementations remain. As such, when using this Python implementation, please cite the original paper (see above) and mention that this Python implementation is used.


![](https://github.com/dcnieho/I2MC_Python/raw/master/Figure1.png)
*Figure 1.* Number of fixations, mean fixation duration and *SD* of fixation duration for the MATLAB and Python I2MC implementations using the RMS noise data set from the original I2MC paper. I2MC is the original published version of I2MC. I2MC2019 is a slightly modified version (the latest version from the original I2MC repository as of this writing [v2.0.3]). _python stands for the Python implementation in this repository. _nc stands for No Chebychev filtering. For more information on this specific analysis see Figure 3 in the original I2MC paper and the corresponding text.


![](https://github.com/dcnieho/I2MC_Python/raw/master/Figure2.png)
*Figure 2.* Classified fixations for episodes of example eye-tracking data using the MATLAB and Python I2MC implementations. I2MC is the original published version of I2MC. I2MC2019 is a slightly modified version (the latest version from the original I2MC repository as of this writing [v2.0.3]). _python stands from the Python implementation in this repository. _nc stands for No Chebychev filtering. For more information on this specific analysis see Figure 8 in the original I2MC paper and the corresponding text.


## Disclaimer

We take no responsibility for the correct working of this Python implementation of I2MC. We provide this solely as a service to the part of the (scientific) community that prefers Python over MATLAB. As we are not dedicated Python programmers, we welcome any bug reports, as well as improved implementations of I2MC.
