# empanada

## Overview

empanada (**EM** **Pan**optic **A**ny **D**imension **A**nnotation) is a tool for panoptic segmentation of organelles in 2D and 3D electron microscopy (EM) images.
Panoptic segmentation combines both instance and semantic segmentation enabling a holistic approach to image annotation
with a single deep learning model. To make model training and inference lightweight and broadly applicable to many EM
imaging modalities, empanada runs all expensive operations on 2D images or run length encoded versions of 3D volumes.

**Note: Development is active and breaking changes should be expected. Not all features are implemented.**

## Highlights

  - Train 2D panoptic segmentation models by customizing easy to read .yaml config files.
  - Get better models faster by using state-of-the-art CEM pretrained model weights to initialize training.
  - Run models on 2D images, videos, or isotropic and anisotropic volumes with orthoplane and stack inference, respectively.
  - Export GPU and quantized CPU trained models for use in napari with [empanada-napari](https://github.com/volume-em/empanada-napari).

## Installation

Install empanada with pip:

```shell
pip install empanada-dl
```

To install the latest development version of empanada directly from GitHub:

```shell
pip install git+https://github.com/volume-em/empanada.git
```
