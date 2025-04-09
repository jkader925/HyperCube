<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" width="100%" srcset="https://github.com/user-attachments/assets/223adf80-8792-454e-ad99-94bbef751a5c">
    <img width="100%" alt="auto light/dark mode" src="https://github.com/user-attachments/assets/2e03646a-c6d5-444f-bcf6-bf43dfd4dd5c">
  </picture>
</div>


HyperCube is a python-based spectral fitting tool designed to make integral field spectroscopic (IFS), or hyperspectral data analysis more interactive and intuitive, while preserving automation and repeatability. The tool combines a user-friendly [PyQT5](https://github.com/PyQt5) GUI with the robust and flexible fitting capabilities of [lmfit](https://github.com/lmfit/lmfit-py), and is particularly well-suited for interactive and batch process spectral modeling of 3D spectral data.

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
python hypercube.py
```

This should launch the main application window. You can now load the IFS data using the `Open FITS` button on the bottom right of the application window and selecting the file `IRAS_F23365+3604.fits`. The main application (visualizer) window will now show to panels: on the left is the image viewer, which initially shows a white light image of the galaxy (from integrating the spectrum in each spectral pixel, or "spaxel"), on the right is a live spectrum viewer that updates as you move the cursor across the white light image. 

### Interacting with the Image Viewer Panel
As the cursor is moved around the image, an orange rectangle indicates the currently focused spaxel. You can lock the spaxel by pressing the `L` key. To unlock, move the cursor back to the image viewer panel and press `L` again.

### Interacting the the Spectrum Viewer Panel
The spectrum viewer panel shows the spectrum contained in the currently-selected spaxel. You can zoom into a portion of the spectrum by clicking and dragging across the spectrum. As you do, a grey rectangular region will indicate the range that will be zoomed to when you release click. The new horizontal (spectral) range reflects the one you selected, while the new vertical (signal or flux) range is auto-scaled to show the continuum and the peaks of any lines in that spectral window. Right-click anywhere on the spectrum to bring up a `reset zoom` button which can be clicked to set the spectrum viewer window to its original range.

## Acknowledging HyperCube
If you used HyperCube in your research, please consider acknowledging the use of the tool by including this text in your publications:

_This research has made use of HyperCube, the interactive analysis tool for integral field spectroscopic data, written by Justin A Kader._
