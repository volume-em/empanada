.. empanada documentation master file, created by
   sphinx-quickstart on Wed Jan 12 11:00:52 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. comments

  .. image:: _static/empanada.png
      :width: 355px
      :alt: empanada

Overview
------------

Empanada (**EM** **Pan**\optic **A**\ny **D**\imension **A**\nnotation) is a tool for panoptic segmentation of organelles in electron microscopy (EM) images in 2D and 3D.
Panoptic segmentation combines both instance and semantic segmentation enabling a holistic approach to image annotation
with a single deep learning model. To make model training and inference lightweight and broadly applicable to many EM
imaging modalities, Empanada runs all expensive operations on 2D images or run length encoded versions of 3D volumes.


Highlights
==================

* Train 2D panoptic segmentation models by customizing easy to read .yaml config files.
* Get better models faster by using state-of-the-art CEM pretrained model weights to initialize training.
* Run models on 2D images, videos, or isotropic and anisotropic volumes with orthoplane and stack inference, respectively.
* Export GPU and quantized CPU trained models for use in Napari with empanada-napari.

Installation
==================

Install Empanada from PyPI::

    $ pip install empanada-dl

To install the latest development version of Empanada, you can use pip from GitHub::

    $ pip install git+https://github.com/volume-em/empanada.git

Contents
==================

.. toctree::
    :maxdepth: 2

    tutorial
    empanada-napari


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
