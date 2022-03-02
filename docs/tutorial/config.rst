Configuration
-------------

Empanada uses YAML to define training and inference parameters.

Training
=========

There are four sections to the training config: ``DATASET``, ``MODEL``, ``TRAIN`` and ``EVAL``.

The ``DATASET`` section gives a name to the training dataset, names the classes that will be segmented
and lists the ids of classes that require instance segmentation.

An example for instance segmentation of a single class:

.. code-block:: yaml

    DATASET:
      # name of dataset tracked by MLFLOW (see Logging in docs)
      dataset_name: "Mitochondria"
      # name of the segmentation class
      class_names: [ "mito" ]
      # only 1 segmentation class, id == 1
      labels: [ 1 ]
      # the 1 segmentation class has instances
      thing_list: [ 1 ]
      # pixel mean and std in dataset images
      norms: { mean: 0.508979, std: 0.148561 }

An example for true panoptic segmentation:

.. code-block:: yaml

    DATASET:
      dataset_name: "Organelles"
      # name of the segmentation class
      class_names: [ "er", "nuclei", "vesicles", "mito" ]
      # the four segmentation classes
      labels: [ 1, 2, 3, 4 ]
      # ER is semantic, all the others are instance
      thing_list: [ 2, 3, 4 ]
      norms: { mean: 0.577841, std: 0.114811 }

The ``MODEL`` section provides arguments to the model class. Any model arg or kwarg can be
added to the config file and passed to the model during training.

.. code-block:: yaml

    MODEL:
      # name of the model architecture
      arch: "PanopticDeepLabPR"
      # the encoder network to use
      encoder: "resnet50"
      # kwargs for this particular model architecture
      num_classes: 1
      stage4_stride: 16
      decoder_channels: 256
      low_level_stages: [ 1 ]
      low_level_channels_project: [ 32 ]
      atrous_rates: [ 2, 4, 6 ]
      aspp_channels: null
      aspp_dropout: 0.5
      ins_decoder: True
      ins_ratio: 0.5

The ``TRAIN`` section defines all parameters used to train the model, it's pretty extensive to
allow for maximum flexibility.

.. code-block:: yaml

    TRAIN:
      # name of the run stored in MLFLOW under "dataset_name"
      run_name: "Panoptic DeepLab PR Baseline" 

      # Directories containing images and masks for the segmentation
      # task. Multiple directories can be appended to the main train_dir
      train_dir: "mitochondria_data"
      additional_train_dirs: [ "new_mitochondria_data" ]

      # directory to save models to
      # and epochs between saving
      model_dir: "pdlpr_baselines/" 
      save_freq: 1

      # path to .pth file for resuming training
      resume: null 

      # encoder pretrained weights
      encoder_pretraining: "cem1.5m_swav_resnet50_200ep_balanced.pth.tar"
      # pretrained weights for the entire model, useful for finetuning
      whole_pretraining: null
      # layers in the encoder to finetune, choice of 
      # "all", "none", "stage1", "stage2", "stage3", "stage4"
      finetune_layer: "all"

      # the learning rate schedule, 
      # see torch.optim.lr_scheduler
      lr_schedule: "OneCycleLR"
      schedule_params:
        max_lr: 0.003
        epochs: 30
        steps_per_epoch: 339
        pct_start: 0.3

      # automatic mixed precision
      amp: True

      # setup the optimizer, see torch.optim
      optimizer: "AdamW"
      optimizer_params:
        weight_decay: 0.1

      # parameters to pass to the PDL loss function, see empanada.losses
      criterion_params:
        ce_weight: 1
        mse_weight: 200
        l1_weight: 0.01
        top_k_percent: 0.2
        confidence_loss: False
        cl_weight: 0.1
        pr_weight: 1

      # training performance metric to track, see empanada.metrics
      print_freq: 50
      metrics: [ "IoU" ]
      metric_params:
        topk: 1

      # dataset and dataloader parameters, see empanada.data
      batch_size: 64
      dataset_class: "SingleClassInstanceDataset"
      weight_gamma: 0.3
      workers: 8

      # augmentations to apply to images, 
      # nearly any in albumentations are supported
      augmentations:
        # aug name and kwargs
        - { aug: "RandomScale", scale_limit: [ -0.9, 1 ]} 
        - { aug: "PadIfNeeded", min_height: 256, min_width: 256, border_mode: 0 }
        - { aug: "RandomCrop", height: 256, width: 256}
        - { aug: "Rotate", limit: 180, border_mode: 0 }
        - { aug: "RandomBrightnessContrast", brightness_limit: 0.3, contrast_limit: 0.3 }
        - { aug: "HorizontalFlip" }
        - { aug: "VerticalFlip" }

      # parameters for multi-GPU training
      multiprocessing_distributed: False
      gpu: null
      world_size: 1
      rank: 0
      dist_url: "tcp://localhost:10001"
      dist_backend: "nccl"

The ``EVAL`` section handles the evaluation of the model during training. It's an optional
section. Setting aside a validation dataset is useful for tuning training parameters though.

.. code-block:: yaml

    EVAL:
      eval_dir: "eval_mitochondria"
      # track segmentation of particular images in the
      # evaluation data, stored in MLFLOW
      eval_track_indices: [ 10, 32, 19, 7 ]

      # how often to record segmentations
      eval_track_freq: 10

      # epochs between evaluation
      epochs_per_eval: 1

      # parameters needed for eval_metrics
      # see empanada.metrics
      metrics: [ "IoU", "PQ", "F1" ]
      metric_params:
          topk: 1
          labels: [ 1 ]
          label_divisor: 1000
          iou_thr: 0.5

      # parameters used for panoptic inference
      # see empanada.inference.engines
      engine_params:
        thing_list: [ 1 ]
        label_divisor: 1000
        stuff_area: 64
        void_label: 0
        nms_threshold: 0.1
        nms_kernel: 7
        confidence_thr: 0.5

Inference
==========

The config file for model inference has only a single section and is less complicated
than the training config.

.. code-block:: yaml

    # axes to predict for 3d
    # for stack inference, just 'xy'
    axes: [ 'xy', 'xz', 'yz' ]

    # list of all segmentation labels
    labels: [ 1 ]

    # parameters for the inference engine
    # see empanada.inference.engines
    engine_params:
      median_kernel_size: 7
      thing_list: [ 1 ]
      label_divisor: 20000
      stuff_area: 64
      void_label: 0
      nms_threshold: 0.1
      nms_kernel: 7
      confidence_thr: 0.3
      input_scale: 1
      scales: [ 1 ]

    # parameters for instance matching
    # across 2d images
    matcher_params:
      merge_iou_thr: 0.25
      merge_ioa_thr: 0.25
      force_connected: True
      
    # parameters for the consensus algorithm
    consensus_params:
      pixel_vote_thr: 0.5
      cluster_iou_thr: 0.75

    # object filters, see empanada.inference.filters
    filters:
      - { name: "remove_small_objects", min_size: 500 }
      - { name: "remove_pancakes", min_span: 4 }

Inheritance
===========

Fortunately, defining a training config file from scratch is almost never necessary. To modify one or a few parameters just
add the ``BASE`` key to the top of the config file and point it to the parent config file. Any parameters that you define
in this new config file will overwrite the parameters in the ``BASE`` config, everything else gets inherited.

For example, to change the number of training epochs used by the OneCycle policy, the complete config
file would look like this:

.. code-block:: yaml

    BASE: "path_to_parent_training.yaml"
    TRAIN:
      # not necessary to change the run name
      # but it makes tracking experiments easier
      run_name: "Panoptic DeepLab PR Baseline - 100 Epochs" 
      schedule_params:
        epochs: 100

Or the change the median filter size for inference:

.. code-block:: yaml

    BASE: "path_to_parent_inference.yaml"

    engine_params:
      median_kernel_size: 3

Inheritance is a powerful way to test many different configurations with minimal effort and should
be used as often as possible. An arbitrary number of levels are possible too; for example, a config
file can inherit from another file that also uses inheritance.