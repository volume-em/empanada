empanada-napari
-----------------

.. image:: _static/demo.gif
    :width: 1000px
    :align: center
    :alt: 2D inference demo

Installation
==============

Napari is still considered alpha phase software and may not install correctly on the
first attempt, if that happens please open an issue `with us here <https://github.com/volume-em/empanada-napari/issues>`_.
Or reach out to the napari developers directly `here <https://github.com/napari/napari/issues>`_.

.. note::

  **Only Python 3.7, 3.8, 3.9 are supported, 3.10 and later are not.**

1. If not already installed, you can `install miniconda here <https://docs.conda.io/en/latest/miniconda.html>`_.

2. Download the correct installer for your OS (Mac, Linux, Windows).

3. After installing `conda`, open a new terminal or command prompt window.

4. Verify conda installed correctly with::

    $ conda --help

  .. note::
      If you get a "conda not found error" the most likely cause is that the path wasn't updated correctly. Try restarting
      the terminal or command prompt window. If that doesn't work then
      see `fixing conda path on Mac/Linux <https://stackoverflow.com/questions/35246386/conda-command-not-found>`_
      or `fixing conda path on Windows <https://stackoverflow.com/questions/44597662/conda-command-is-not-recognized-on-windows-10>`_.

5. If you've previously installed and used conda, it's recommended (but optional) to create a new virtual environment in order to avoid dependency conflicts::

    $ conda create -y -n empanada -c conda-forge python=3.9
    $ conda activate empanada

6. Install napari with pip::

    $ python -m pip install "napari[all]"

7. To verify installation, run::

    $ napari

For alternative and more detailed installation instructions, see the
`official napari installation tutorial <https://napari.org/tutorials/fundamentals/installation>`_.

From here the easiest way to install empanada-napari is directly in napari.

1. From the “Plugins” menu, select “Install/Uninstall Plugins...”.

.. image:: _static/plugin-menu.png
  :align: center
  :width: 200px
  :alt: Napari Plugin menu

2. In the resulting window that opens, where it says “Install by name/URL”, type "empanada-napari".

.. image:: _static/plugin-install-dialog.png
  :align: center
  :width: 500px
  :alt: Plugin installation dialog

3. Click the “Install” button next to the input bar.

If installation was successful you should see `empanada-napari` in the Plugins menu. If you don't
see it, restart napari.

If you still don't see it, try installing the plugin with pip::

	$ pip install empanada-napari

Modules Overview
===================

Model Inference
^^^^^^^^^^^^^^^^^

`2D Inference (Parameter Testing)`_: Runs model inference on 2D images. Supports batch mode for
predicting segmentations on a series of unrelated images or can be used to segment arbitrary 2D slices
from volumetric data.

`3D Inference`_: Implements stack and ortho-plane inference functionality for volumetric datasets.

Model Training
^^^^^^^^^^^^^^^^^

`Pick training patches`_: Automatically picks patches of data to annotate from 2D or
3D images. Also gives the option for uses for manually select ROIs using placed points.

`Store training dataset`_: Stores training patch segmentations in the correct format
expected for model finetuning and training.

`Finetune a model`_: Allows users to finetune any registered model on a specialized
segmentation dataset.

`Train a model`_: Train models from scratch for arbitrary panoptic segmentation tasks.
Optionally, initialize training from CEM pre-trained weights for faster convergence
and greater robustness.

`Register new model`_: Make a new model accessible in all other training and inference
modules. Models can be registered from .pth files or from web URLs. Useful for
sharing models locally or over the internet.

Proofreading Tools
^^^^^^^^^^^^^^^^^^^^

`Merge and Delete labels`_: Allows the selection of multiple instances and merges them all to
the same label and allows the removal of selected labels, respectively.

`Split labels`_: Allows the placement of multiple markers for distance watershed-based instance splitting.

`Jump to label`_: Given a label ID, moves the napari viewer to the first 2D slice where an object appears.

2D Inference (Parameter Testing)
==================================

.. image:: _static/inference_2d.png
  :align: center
  :width: 500px
  :alt: Dialog for the 2D inference and parameter testing module.

Results
^^^^^^^^^^^^^

Returns a 2D labels layer in the napari viewer.

Parameters
^^^^^^^^^^^^^

**image layer:** The napari image layer on which to run model inference.

**Model:** Model to use for inference.

**Image Downsampling:** Downsampling factor to apply to the input image before running
model inference. The returned segmentation will be interpolated to the original
image size using the Point Rend module.

**Segmentation Confidence Thr:** The minimum confidence required for a pixel to
be classified as foreground. This only applies for binary segmentation.

**Center Confidence Thr:** The minimum intensity of a peak in the centers heatmap
for it to be considered a true object center.

**Centers Min Distance:** The minimum distance allowed between centers in pixels.

**Fine boundaries:** Whether to run Panoptic DeepLab postprocessing at 0.25x the
input image resolution. Can correct some segmentation errors at the cost of 4x
more GPU/CPU memory during postprocessing.

**Semantic Only:** Whether to skip panoptic postprocessing and return only a semantic
segmentation.

**Max objects per class:** The maximum number of objects that are allowed for any one
of the classes being segmented by the model.

**Batch Mode:** If checked, the selected model will be run independently on each
xy slice in a stack of images. This can be used, for example, to run inference on
all images in a folder by loading them with the "Open Folder..." option.

**Use GPU:** Whether to use system GPU for running inference. If no GPU is detected
on the workstation, then this parameter is ignored.

**Use quantized model:** Whether to use a quantized version of the segmentation model.
The quantized model only runs on CPU but uses ~4x less memory and runs 20-50% faster (depending
on the model architecture). Results may be 1-2% worse than using the non-quantized version.

See `Inference Best Practices`_ below for more usage notes.

3D Inference
==================================

.. image:: _static/inference_3d.png
  :align: center
  :width: 500px
  :alt: Dialog for the 3D inference module.

Results
^^^^^^^^^^^^^

Returns a 3D labels layer in the napari viewer for each segmentation class and,
optionally, panoptic segmentation stacks.

General Parameters
^^^^^^^^^^^^^^^^^^^^^^

**image layer:** The napari image layer on which to run model inference.

**model:** Model to use for inference.

**Zarr Directory (optional):** Path at which to store segmentation results in zarr
format. Writing results to disk can help avoid out-of-memory issues when running
inference on large volumes. Napari natively supports reading zarr files.

**Use GPU:** Whether to use system GPU for running inference. The box will be
check by default if a GPU is found on your system. If no GPU is detected, then
this parameter is ignored.

**Use quantized model:** Whether to use a quantized version of the segmentation model.
The quantized model only runs on CPU but uses ~4x less memory and runs 20-50% faster (depending
on the model architecture). Results may be 1-2% worse than using the non-quantized version.

**Multi GPU:** If the workstation is equipped with more than 1 GPU, inference
can be distributed across them. See note in `Inference Best Practices`_.

2D Parameters
^^^^^^^^^^^^^^^^

**Image Downsampling:** Downsampling factor to apply to the input image before running
model inference. The returned segmentation will be interpolated to the original
image size using the Point Rend module.

**Segmentation Confidence Thr:** The minimum confidence required for a pixel to
be classified as foreground. This only applies for binary segmentation.

**Center Confidence Thr:** The minimum intensity of a peak in the centers heatmap
for it to be considered a true object center.

**Centers Min Distance:** The minimum distance allowed between centers in pixels.

**Fine boundaries:** Whether to run Panoptic DeepLab postprocessing at 0.25x the
input image resolution. Can correct some segmentation errors at the cost of 4x
more GPU/CPU memory.

**Semantic Only:** Whether to skip panoptic postprocessing and return only a semantic
segmentation.

Stack Parameters
^^^^^^^^^^^^^^^^^^^

**Median Filter Size:** Number of image slices over which to apply a median filter
to semantic segmentation probabilities.

**Min Size (Voxels):** The smallest size object that's allowed in the final
segmentation as measured in voxels.

**Min Box Extent:** The minimum bounding box dimension that's allowed for an
object in the final segmentation. (Filters out big "pancakes").

**Max objects per class in 3D:** The maximum number of objects that are allowed for any one
of the classes being segmented by the model within a volume.

**Inference plane:** Plane from which to extract and segmentat slices. Choice of xy, xz, or yz.

Ortho-plane Parameters (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Run ortho-plane:** Whether to run ortho-plane inference. If unchecked, inference
will only be run on slices from the Inference plane chosen above.

**Return xy, xz, yz stacks:** Whether to return the panoptic segmentation stacks created
during inference on each plane. If unchecked, only the per-class consensus volumes
will be returned.

**Voxel Vote Thr Out of 3:** Number of stacks from ortho-plane inference in which a voxel
must be labeled in order to end up in the consensus segmentation.

**Permit detections found in 1 stack into consensus:** Whether to allow objects
that appear in only a single stack (for example an object only segmented in xy)
through to the ortho-plane consensus segmentation.

Inference Best Practices
============================

Depending on the size of the volume, 3D inference can take some time even with a GPU,
therefore it's highly recommended to test out inference parameters beforehand using the
2D inference module. The 2D inference module will run inference on whatever image slice
the viewer is pointed at in napari. This means that parameters can be tested on xy, xz and yz
slices by flipping the volume and scrolling through the stack. If results appear substantially
better on slices from a particular plane, then use that plane as the **Inference plane** for
3D inference. Similarly, if results on xy slices are good but results on xz and yz slices are poor,
then don't use ortho-plane inference.

.. note::

  When running the 2D inference module on images of a given size for the first
  time, results can be slow. After inference is run twice on a particular size it will
  be much faster. This is because pytorch is performing optimization in the background to
  make the model faster on your systems hardware.

.. note::

  When using Multi-GPU inference there's some overhead associated with the
  creation of a process group. This overhead can make Multi-GPU inference slower
  on small volumes. Therefore, we don't recommend using it unless working with datasets
  larger than 1 GB. However, in such cases, the inference speed-up is nearly linear
  (segmentation will be 4x faster with 4 GPUs than with 1). As more GPUs are added,
  inference starts to become CPU bound (i.e., the segmentations are being created
  faster on GPU than they can be processed on CPU). There can be a long delay between
  inference and backward matching as the CPU processes work to catch up.

We've found that models can give considerably different results based on the nanometer
resolution of the input image and model inference is faster for smaller input images.
Ideally you'll want to find and use the biggest **Image Downsampling** factor that still gives
satisfactory results.

Tweaking the **Segmentation Confidence Threshold** is often just a proxy for erosion and dilation of labels.
Because ortho-plane inference averages segmentations from 3 views using a lower confidence
threshold is sometimes beneficial: try 0.3 instead of 0.5.

The **Center Confidence Thr** and **Centers Min Distance** parameters both control how split up
instances will be in 2D. Raising the confidence threshold will result in fewer object centers
and therefore fewer instances in the segmentation. Increasing the minimum distance
will filter out centers that are too close together. This can help if you notice
that long objects are being oversplit.

Lastly for 2D parameters, the **Fine boundaries** option may be useful if the borders between instances
are too "blocky". This comes at the cost of 4x more memory usage during postprocessing though, so use it wisely.

The most important 3D parameter is the **Median Filter Size**. This smooths out the stacked
segmentations. The best kernel size is typically a function of voxel size. Lower-resolution
volumes (>20 nm) that have relatively more change between consecutive slices usually benefit from a smaller
kernel size like 3. Higher-resolution volumes (<10 nm) have much less change across slices and a kernel
size of 7 or 9 can work well.

The **Min Size** and **Min Extent** parameters filter out small objects and segmentation "pancakes". The
optimal size is strongly data-dependent. As a rough estimate, try drawing a bounding box around a small
object that you see. Divide the volume of the box by 2 to get the approximate volume of a sphere that
would fit inside that box. Pick some number a few hundred voxels below that threshold as your min size.
Likewise, the min extent should be a few increments less than the smallest dimension of the bounding box.

The **Voxel Vote Thr Out of 3** and **Permit detections found in 1 stack into consensus** are options
for when there are too many false negatives after ortho-plane segmentation. Decreasing the voxel
vote threshold to 1 will fill in more voxels but should not increase the number of false positive detections
very much. This is because the voxel vote threshold only affects detections that were picked up in more than 1 of the
inference stacks. **Permit detections found in 1 stack into consensus**, on the other hand, can increase false positives because
it will allow detections picked up by just a single stack into the consensus segmentation (what a well named parameter!).

When running ortho-plane inference it's recommended to also **Return xy, xz, yz stacks**
segmentations. In some cases, inference results are better on just a single plane (i.e., xz)
than they are in the consensus. Returning the intermediate panoptic results for each stack
will help you to decide whether that applies to your dataset or not.


Pick training patches
==================================

.. image:: _static/pick_patches.png
  :align: center
  :width: 500px
  :alt: Dialog for the patch picking module.

Results
^^^^^^^^^^^^^

If the image to pick patches from is 3D, returns a set of flipbooks with five
images in each along with a corresponding labels layer of the same size. If the
image is instead 2D or a 2D stack, returns a set of patches and a labels layer
of matching size.

.. note::

  When flipbooks are returned, it's assumed that the middle image in each will
  be annotated. For example, in a flipbook with five images, only the third image
  should be segmented.

Parameters
^^^^^^^^^^^^^

**image layer:** The napari image layer from which to sample patches.

**points layer:** Optional. The napari points layer containing fiducial points
centered at ROIs to pick for annotation.

**Number of patches for annotation:** Number of patches to pick for annotation.
By default, patches are chosen randomly. If the points layer was given but has
fewer points than this number, the remainder will be made up with random patches.
Overwritten if **Pick all points** (below) is selected.

**Patch size in pixels:** The desired pixel size for chosen patches. All patches
are square.

**Multiscale image level:** If the image layer is a multiscale image, select the
resolution level from which to sample. It's assumed that images in each level
were downsampled by 2x.

**Pick all points:** If checked, patches will be created from all points in
the given points layer, regardless of the **Number of patches for annotation**
that was set.

**Pick from xy, xz, or yz:** If checked, patches will be arbitrarily selected from
any of the principle planes. Only select this option for nearly isotropic voxel
3D datasets.

**Image is 2D stack:** If checked, treats the image layer as a stack of unrelated
2D images. For example, check this box when picking patches from a directory
of 2D images that were loaded with the "Open Folder..." option.

See `Training Best Practices`_ below for more usage notes.

Store training dataset
==================================

.. image:: _static/store_training_dataset.png
  :align: center
  :width: 500px
  :alt: Dialog for the dataset saving module.

Results
^^^^^^^^^^^^^

Creates or appends data to a directory with the structure expected for
model finetuning and training. If the image and labels layers are
flipbooks, only the middle image in each flipbook is saved.

Parameters
^^^^^^^^^^^^^

**image layer:** The napari image layer for annotated patches or flipbooks.

**labels layer:** The napari labels layer for annotated patches or flipbooks.

**Save directory:** Directory in which to save the dataset. A subdirectory
with the given **Dataset name** (below) will be created.

**Dataset name:** Name of the dataset directory to create. If the dataset already
exists, the new data will be appended.

See `Training Best Practices`_ below for more usage notes.

Finetune a model
==================================

.. image:: _static/finetune_model.png
  :align: center
  :width: 500px
  :alt: Dialog for the model finetuning module.

Results
^^^^^^^^^^^^^

Saves and registers a .pth torchscript model that has been finetuned on
the provided data. Also saves a .yaml config with parameters necessary for
additional finetuning.

Parameters
^^^^^^^^^^^^^

**Model name, no spaces:** Name of the finetuned model as it will appear in the
other empanada modules after finetuning.

**Train directory:** Training directory for finetuning. Must conform to the
standard directory structure specified for empanada (as for example is created
by the `Store training dataset`_ module).

**Validation directory (optional):** Validation directory. Must conform to the
standard directory structure specified for empanada. Can be the same as **Train directory**.

**Model directory:** Directory in which to save the finetuned model definition
and config file. The directory will be created if it doesn't exist already.

**Model to finetune:** Empanada model to finetune.

**Finetunable layers:** Layers to unfreeze in the model encoder during
finetuning.

**Iterations:** Number of iterations to finetune the model.

**Custom config (optional):** Use a custom config file to set other training
hyperparameters. `See here for a template to modify <https://github.com/volume-em/empanada-napari/blob/train/custom_configs/custom_finetuning.yaml>`_.

Train a model
==================================

.. image:: _static/train_model.png
  :align: center
  :width: 500px
  :alt: Dialog for the model training module.

Results
^^^^^^^^^^^^^

Saves and registers a .pth torchscript model that has been trained on
the provided data. Also saves a .yaml config with parameters necessary for
additional finetuning.

Parameters
^^^^^^^^^^^^^

**Model name, no spaces:** Name of the model as it will appear in the
other empanada modules after training.

**Train directory:** Training directory for finetuning. Must conform to the
standard directory structure specified for empanada (as for example is created
by the `Store training dataset`_ module).

**Validation directory (optional):** Validation directory. Must conform to the
standard directory structure specified for empanada. Can be the same as **Train directory**.

**Model directory:** Directory in which to save the trained model definition,
weights, and config file. The directory will be created if it doesn't exist already.

**Dataset labels:** List of labels in the training dataset. Each line is a comma separated list of three
items without spaces: <class_id>,<class_name>,<class_type>. Class IDs must be integers, class names
can be anything, class types must be either 'semantic' or 'instance'.

**Label divisor:** For mutliclass segmentation, the label divisor that was used
to offset the labels for each class.

**Model architecture:** The model architecture to use for training.

**Use CEM pretrained weights:** If checked the model encoder will be initialized
with the latest CEM weights. (CEM weights are created by self-supervised training
on the very large and heterogeneous CEM dataset).

**Finetunable layers:** Layers to unfreeze in the model encoder during
training. Ignored if **Use CEM pretrained weights** isn't checked.

**Iterations:** Number of iterations to train the model.

**Custom config (optional):** Use a custom config file to set other model and training
hyperparameters. `See here for a template to modify <https://github.com/volume-em/empanada-napari/blob/train/custom_configs/custom_training_pdl.yaml>`_.

Register new model
====================

.. image:: _static/register_new_model.png
  :align: center
  :width: 500px
  :alt: Dialog for the register new model module.

Results
^^^^^^^^^^^^^

Adds a new model to choose in inference and training modules.

Parameters
^^^^^^^^^^^^^^^^

**Model name:** Name to use for this model throughout the other plugin modules.

**Model config file:** Config file for the model as created in the Finetuning and Training
modules or by exporting from the empanada library.

**Model file (optional):** Path or URL to the torchscript .pth file defining the model. If the path/url
given in the config file is correct this is unnecessary.

**Quantized model file (optional):** Path or URL to the quantized torchscript .pth file defining the model.
If the path/url given in the config file is correct this is unnecessary.

.. note::

  If the 2D or 3D Inference module have already been opened, then registered models may not
  appear in the available models list. Open and close the relevant module or restart napari.

.. note::

  Removing models is manual. Simply delete or move the config file from `~/.empanada/configs`.

Merge and Delete labels
=============================

.. image:: _static/merge_labels.png
  :align: center
  :width: 500px
  :alt: Dialog for the merge and delete labels module.

Results
^^^^^^^^^^^^^

In-place merges or deletes selected labels from a labels layer.

Parameters
^^^^^^^^^^^^^^^^

The parameters for the Merge and Delete labels modules are the same.

**labels layer:** The napari labels layer for which to apply operations.

**points layers:** The napari points layer used to select instances for merging/deletion.

Split labels
=============================

.. image:: _static/split_labels.png
  :align: center
  :width: 500px
  :alt: Dialog for the split labels module.

Results
^^^^^^^^^^^^^

In-place splits the selected label in the labels layer.

Parameters
^^^^^^^^^^^^^^^^
.. note::

  Only one instance can be split at a time. All points aside from the first one will
  be ignored and deleted unless **Use points as markers** (below) is checked.

**labels layer:** The napari labels layer for which to apply operations.

**points layers:** The napari points layer used to select instance for splitting.

**Minimum Distance:** Minimum distance from the boundary of the instance for
a pixel/voxel to be included in a watershed marker.

**Use points as markers:** If checked, minimum distance will be ignored and the
watershed transform will treat each point as a marker.


Jump to label
=============================

.. image:: _static/jump_label.png
  :align: center
  :width: 500px
  :alt: Dialog for the jump to label module.

Results
^^^^^^^^^^^^^

Moves the napari viewer to the first slice showing the given label ID, if found.

Parameters
^^^^^^^^^^^^^^^^

**labels layer:** The napari labels layer in which to find the label.

**Label ID:** Integer ID for the label to jump the viewer to.
