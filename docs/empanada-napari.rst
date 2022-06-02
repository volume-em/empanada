empanada-napari v0.2.0
************************

The empanada-napari plugin is built to democratize deep learning image segmentation
for researchers in electron microscopy (EM). It ships with MitoNet, a generalist model 
for the instance segmentation of mitochondria. There are also 
tools to quickly build and annotate training datasets, train generic panoptic segmentation 
models, finetune existing models, and scalably run inference on 2D or 3D data. To make 
segmentation model training faster and more robust, CEM pre-trained weights are used by 
default. These weights were trained using an unsupervised learning algorithm on over 
1.5 million EM images from hundreds of unique EM datasets making them remarkably general. 

.. image:: _static/demo.gif
    :width: 1000px
    :align: center
    :alt: 2D inference demo

Contents
----------

.. toctree::
    :maxdepth: 2

    plugin/install
    plugin/modules
    plugin/best-practice
    plugin/tutorials

