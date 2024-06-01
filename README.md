# flow_models
Flow-based invertible neural networks implemented with Keras and Tensorflow

Work currently still in progress; not yet fully debugged/functional.
Currently focusing on running train.py on gpu instance.


### To run
0. For full runs on a GPU-enabled EC2 instance (as opposed to just initial
   smaller scale testing on a CPU-only instance), I recommend following
   [these instructions](https://github.com/aganse/py_tf2_gpu_dock_mlflow/blob/main/doc/aws_ec2_install.md)
   from my [py_tf2_gpu_dock_mlflow](https://github.com/aganse/py_tf2_gpu_dock_mlflow)
   repository to set that up.
   I'm also working on some scripts to kick off the training remotely in a Docker
   container via AWS ECR using AWS Batch, but that's not ready yet.  Meanwhile,
   simply installing on the GPU-enabled instance per those instructions allows
   to run the training on there.

1. Create the python environment and install dependencies:
    ```
    # within the flow_models directory:
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2. Get images to work with.  Two main options for this:

  a. Download a relevant Kaggle dataset.  E.g. I thought this
    [animal-faces](https://www.kaggle.com/datasets/andrewmvd/animal-faces) one
    was especially good, focusing on just the cats.  Other Kaggle datasets I
    may try in near future include:

    * [Facial Expressions Training Data](https://www.kaggle.com/datasets/noamsegal/affectnet-training-data?select=disgust)

    * [Facial Expression Image Data AFFECTNET YOLO Format](https://www.kaggle.com/datasets/fatihkgg/affectnet-yolo-format)

    * [Fresh and Rotten Classification](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification)

    * [Flower Classification 10 Classes](https://www.kaggle.com/datasets/utkarshsaxenadn/flower-classification-5-classes-roselilyetc)


  b. Download images into the appropriate directories:

    `python download_images_bing.py`
    This will create the following directory structure to hold the downloaded images.
    (The "cat" subdirectory is because the search keyword was "cat" - you can of
    course change that, and the unsupervised learning doesn't care what the subdirs
    are within "train" and "val" anway, it just globs them together.)
    None of these directories needs to exist already - the script can create them.

    Warning - I did find a lot of web-scraping packages don't seem to work anymore
    (search engine APIs seem to evolve quickly/regularly).  The Bing one
    technically still works but does not provide very good/reliable cat photos.
    Honestly this is what made me shift to using pre-made image datasets myself;
    really the quality/consistency is better too.

    After first getting things working with just cats, the idea is to add another
    "val" directory with not only a bunch of cats, but also a new subdir of say
    "beachball" with only a few images, and see if those show up as outliers in the
    multivariate Gaussian distribution when mapped through the model.
    ```
    data/
        train/
            cat/
        val/
            cat/
            beachball/
    ```

3. Then run train.py:
   There are a number of status/info lines spewed by Tensorflow and Tensorflow
   Probability (TFP) that I don't find helpful and that make a mess.  To squelch
   those I first set environment variable `export TF_CPP_MIN_LOG_LEVEL=2` in my
   shell that I'll run the training in.  Similarly note I've put a python line
   at the top of train.py to squelch `UserWarning`s that are spewed by TFP.
   In any case, then you can simply run:
    `python train.py`


### Main steps in the code (train.py)

#### Create data generators for training and validation
Use TF ImageDataGenerator to load and preprocess images.
The `class_mode=None` parameter indicates unsupervised
learning, meaning no labels are provided.
This approach leverages the convenience of ImageDataGenerator for handling
large datasets and built-in data augmentation while maintaining the
unsupervised nature of the RealNVP model.

#### Define the RealNVP model
Use tfp.bijectors and tfp.distributions from Tensorflow probability to create
a Flow/RealNVP-based invertible neural network.

#### Training loop: train the model using the data generator
The loop processes batches of images from the generator to train the RealNVP
model in unsupervised mode, ie without requiring labels.

#### Visualization: 

##### Plot Gaussian points generated from validation images
Generate multivariate Gaussian points and apply PCA for visualization
The transformed images are mapped to Gaussian points, and PCA is
applied to visualize the 2D projection. Outliers (e.g., non-cat images) should
appear as points that are distant from the main cluster of cat images.

##### Save new generated/simulated images
Try generating new images of cats by generating random samples from the multivariate
Gaussian and mapping those back through the network.
