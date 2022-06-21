.. _best-practice:

Best Practices
-----------------

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

Finetuning Best Practices
============================

.. note::
  
  empanada-napari is best for finetuning models quickly and for users with limited
  coding experience. The finetuning module was designed to be simple and eliminate
  the complexities of setting hyperparameters. For deep experimentation, it's strongly
  recommended to use the scripts provided by empanada instead of napari. Those scripts
  have built in support for tracking runs in MLflow and support significantly more
  features like data augmentations, optimizers, learning rate schedules, etc.

Finetuning modifies the parameters in a pre-trained model with new data. In empanada, a finetuned
model can't be given new capabilities. That means that the **MitoNet_v1** model can only be tweaked
to better segment mitochondria in new data; it can't be used to segment a different organelle like
nuclei. New tasks call for training new models: `Training Best Practices`_.

How much training data and finetuning are needed depends on how well existing models segment your
data. For example, if the **MitoNet_v1** model appears to be about 30% accurate in terms of intersection-over-union,
relatively brief finetuning of 100-200 iterations with 16 labeled patches should be sufficient to get satisfactory results 
(though it depends on how large and diverse the data you need to segment is). When working with large volumes that have
multiple distinct "contexts" (e.g., different cell types) try running inference on 2D slices in all of those areas.
If the model only performs poorly in a subset, then select training example by dropping points and using the 
:ref:`plugin/modules:Pick training patches` module.

Although by default it's required that you have 16 labeled images in order to finetune a model there are ways to skirt
that requirement. One way is to use a **Custom config** with batch size set to a number smaller than 16 (see note).
You could also duplicate images and masks after they've been saved with :ref:`plugin/modules:Store training dataset` until
there are at least 16. Finally, if finetuning one of the **MitoNet** models you could supplment your annotations with
relevant data from `CEM-MitoLab <https://www.ebi.ac.uk/empiar/EMPIAR-11037/>`_. With the metadata spreadsheet it's
possible to pick subdirectories with images that are related to the biological context for which you'd like to
finetune the model.

.. note::
  
  16 may seem arbitrary but it's based on experience. For the given model architectures and training
  hyperparameters, a batch size of 16 is verified to be safe. Going lower may make the BatchNorm parameters in the model
  unstable, going higher is perfectly fine.

Finally, setting the **Finetunable layers** appropriately can make finetuning faster and more effective. The layers available
to finetune are all in the encoder of the network, commonly a ResNet50 (decoder parameters are always trained). 
With the layer set to *none* training will be fastest but may underfit. If the model was already pretty good on your 
data then this shouldn't be much of a concern. Conversely, setting the layer to *all* will make training slowing and
may lead to overfitting. That won't be a concern if your data is pretty homogeneous, but could be a problem otherwise.
As a rule of thumb, start with *none*. If the validation metrics/inference results aren't as good as you'd like try
*all*. Using *stage4* is actually the best choice between the two extremes. 


Training Best Practices
============================

Panoptic segmentation is a powerful framework that allows segmentation of arbitrary
combinations of instance and semantic classes. This is especially relevant for EM 
data in which some organelles like mitochondria should have individual instances segmentated while 
others like ER only make sense in the context of semantic segmentation.

.. note::

  If you already have a labeled dataset (like CEM-MitoLab). The only requirement to use 
  finetuning or training is that you put the images into the correct directory structure.
  That structure is:

  *name_of_training_dataset*
  \
   *name_of_2D_image_or_3D_volume*
   \
    *images*
    \
     image1.tiff

     image2.tiff
    *masks*
    \
     image1.tiff   

     image2.tiff

  There can be multiple *name_of_2D_image_or_3D_volume* subdirectories. Each must have a subdirectory called images 
  and another called masks. Corresponding image and mask .tiff files must have identical names but reside in the 
  appropriate folder.

Instructions for labeling data correctly can be found in the :ref:`plugin/tutorials:Training a panoptic segmentation model`
tutorial. There are only two available model architectures to choose from: **PanopticDeepLab** and **PanopticBiFPN** 
(these are the architectures behind MitoNet_v1 and MitoNet_v1_mini, respectively). Both models predict the same 
targets: a semantic segmentation, an instance center heatmap, and xy offset vectors from each pixel to an 
associated object center (see `here <https://arxiv.org/abs/1911.10194>`_ for details). The key difference is 
in the number of parameters. PanopticBiFPN has about 50% fewer (29 million compared to 55 million).
In resource constrained compute environments, always opt for PanopticBiFPN. While smaller it's still a 
strong architecture based on the popular `EfficientDet <https://arxiv.org/abs/1911.09070>`_.

By default both models use ResNet50 as the encoder. If you choose to use a **Custom config** it's 
also possible to choose from smaller ResNet models or RegNets. The disadvantage is that none of these 
encoders can take advantage of CEM pretrained weights. CEM pretraining makes training a model much 
faster and has been shown to increase robustness and generalization (see our recent work with 
`CEM1.5M <https://www.biorxiv.org/content/10.1101/2022.03.17.484806>`_ and 
`CEM500K <https://elifesciences.org/articles/65894>`_). It's strongly recommended to always 
leave the **Use CEM pretrained weights** box checked.

Similar to best practices for finetuning, the **Finetunable layers** parameter can control 
the degree of over/underfitting that occurs. Setting this field to *all* generally yields
the best results in our experience. When compute resources are very constrained, training 
a PanopticBiFPN with the finetunable layer set to *none* is the best choice. It only 
requires the training of about 3 million parameters. 

For training a new model the number of **Iterations** required is highly task dependent. As a 
rule of thumb, models that predict more classes and have more training data or parameters will need 
more iterations. 500-1,000 iterations is a great range to start with.

The **Patch size in pixels** controls the size of random crops used during model training. This should be 
set to be the same size or smaller than the patch size chosen for the training data. If using PanopticDeepLab 
the patch size must be divisible by 16. For PanopticBiFPN it must be divisible by 128. Larger patch sizes are 
typically better for segmentation models, but we've found that most organelles can be well captured with patches 
of 256.