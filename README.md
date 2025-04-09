<p align="center">
  <img width="256" alt="HyperCube_logo" src="https://github.com/user-attachments/assets/7417ad66-b371-4065-84d8-e8392ddea246" />
</p>


<picture align="center">
  <source media="(prefers-color-scheme: dark)" width="256" srcset="https://github.com/user-attachments/assets/6a1518f7-834e-4cab-ab66-d5150d59dfec">
  <img width="256" alt="Shows an illustrated sun in light color mode and a moon with stars in dark color mode." src="https://github.com/user-attachments/assets/e5cf39ab-9f20-4a2d-bc5e-ac9f1de6a29d">
</picture>

HyperCube is a specialized Python3-based spectral fitting tool designed for analyzing integral field spectroscopic (IFS), or hyperspectral data, with a focus on emission lines from interstellar gas. The tool combines a user-friendly [PyQT5](https://github.com/PyQt5) GUI with the robust and flexible fitting capabilities of [lmfit](https://github.com/lmfit/lmfit-py), and is particularly well-suited for interactive and batch process spectral modeling of 3D spectral data.





## Installation
Installation and use of this tool has been tested on MacOS and Windows, it has not yet been tested on Linux operating systems. 

Clone the repository to a directory on your local machine where you have read/write/execute privileges. The tool was designed for quick and painless installation using `conda` environment management via the included environment file `hypercube.yml`. In a terminal, from your base conda environment, navigate to the new HyperCube directory and issue the following command:

```
conda env create -f hypercube.yml
```

Conda will install all of the required packages automatically. If not using conda, you can manually install the required packages (listed in hypercube.yml) via `pip`.

## Quick Start Guide
This guide walks you through a basic analysis of a Keck Cosmic Wave Imager (KCWI) data cube observation of the luminous infrared galaxy IRAS F23365+3604. The purpose of this guide is to familiarize you with the basic features and modes available to you when using HyperCube to fit 3D spectral data, it is not intended as a comprehensive introduction to every feature the tool offers.

From your new `hypercube` conda environment, open the tool via the following command:

```
python
```

## Acknowledging HyperCube
If you used HyperCube in your research, please consider acknowledging the use of the tool by including this text in your publications:

_This research has made use of HyperCube, the interactive analysis tool for integral field spectroscopic data, written by Justin A Kader._
