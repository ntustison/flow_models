# flow_models
Flow-based invertible neural networks implemented with Keras, Tensorflow, and
Tensorflow Probability.

Work currently still in progress, but things are functional meanwhile per
instructions below.  This code is what I used to produce the materials in 
["Sim-cats! Image generation and unsupervised learning for anomaly detection as
  two sides of the same coin"](http://research.ganse.org/datasci/sim-cats)


### A. To install/prepare
1. For full runs on a GPU-enabled EC2 instance (as opposed to just initial
   smaller scale testing on a CPU-only instance), I recommend following
   [these instructions](https://github.com/aganse/py_tf2_gpu_dock_mlflow/blob/main/doc/aws_ec2_install.md)
   from my [py_tf2_gpu_dock_mlflow](https://github.com/aganse/py_tf2_gpu_dock_mlflow)
   repository to set that up.

   I'm also working on some scripts to kick off the training remotely in a
   Docker container via AWS ECR using AWS Batch (that's what's in subdir
   `awsbatch-support`), but that's not ready yet.  Meanwhile, simply installing
   on the GPU-enabled instance per those instructions linked above allows to
   run the training on there.

2. Create the python environment and install dependencies:
    ```
    # within the flow_models directory:
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3. Getting images to work with; two main options for this:

    a. Download a relevant Kaggle dataset.  E.g. I thought this
    [animal-faces](https://www.kaggle.com/datasets/andrewmvd/animal-faces) one
    was especially good, focusing on just the cats.  Other Kaggle datasets I
    may try in near future include:

    * [Facial Expressions Training Data](https://www.kaggle.com/datasets/noamsegal/affectnet-training-data?select=disgust)
    * [Facial Expression Image Data AFFECTNET YOLO Format](https://www.kaggle.com/datasets/fatihkgg/affectnet-yolo-format)
    * [Fresh and Rotten Classification](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification)
    * [Flower Classification 10 Classes](https://www.kaggle.com/datasets/utkarshsaxenadn/flower-classification-5-classes-roselilyetc)

    If using dedicated GPU-enabled instance, you could save these directly on
    that instance in a `data` subdir within the `flow_models` repo directory;
    for that case use TF's ImageDataGenerator in the train.py script.
    Or you can use image files in an S3 bucket, whether in the dedicated
    GPU-enabled instance or soon within AWS Batch configuration; for this case
    use my S3ImageDataGenerator class in the train.py script which is a drop-in
    replacement.  TF's ImageDataGenerator is set as default; once things converge
    in this repo one won't have to explicitly tweak code to enable S3.  ;-)

    b. Download web-scraped images into the appropriate directories.  This did
    not work well for me due to endless image quality issues; I recommend (a)
    instead, but including here just in case:

    `python misc/download_images_bing.py`
    This will create the following directory structure to hold the downloaded images.
    (The "cat" subdirectory is because the search keyword was "cat" - you can of
    course change that, and the unsupervised learning doesn't care what the subdirs
    are within "train" and "val" anway, it just globs them together.)
    None of these directories needs to exist already - the script can create them.

    Warning - I found a lot of web-scraping packages don't seem to work anymore
    (search engine APIs seem to evolve quickly/regularly).  The Bing one
    technically still works but does not provide very good/reliable photos.
    Honestly this is what made me shift to using pre-made image datasets myself;
    really the quality/consistency is better too.
    ```
    data/
        train/
            cat/
        val/
            cat/
            beachball/   <-- these show up as outliers in gaussian latent points
    ```

### B. To run the training
1. Enter the python environment created above if not already in:  `source .venv/bin/activate`
2. Set environment variable `export TF_CPP_MIN_LOG_LEVEL=2` to squelch a number of status/info lines spewed by Tensorflow and Tensorflow
    Probability (TFP) that I don't find too helpful and that make a mess in the console output.  (Similarly note I've put a python line
    at the top of train.py to squelch `UserWarning`s that are spewed by TFP.)
3. Set `datagen = S3ImageDataGenerator()` or `datagen = ImageDataGenerator()`
    in train.py per discussion above.  (temporary; soon unnecessary)
4. Run `python train.py`.
