"""
https://github.com/gurugaurav/bing_image_downloader
query_string : String to be searched.
limit : (optional, default is 100) Number of images to download.
output_dir : (optional, default is 'dataset') Name of output dir.
adult_filter_off : (optional, default is True) Enable of disable adult filteration.
force_replace : (optional, default is False) Delete folder if present and start a fresh download.
timeout : (optional, default is 60) timeout for connection in seconds.
Further reference: https://pypi.org/project/bing-image-downloader/

I find in practice that the process started in this script often dies by the
time it gets close to 1000 images downloaded and I must restart.  Unfortuantely
there's no parameter to choose a new starting index for the filenames - so if
you rerun as-is, it'll start overwriting the files beginning at index 0 again.
Just rename your previous directory (eg `mv cat cat0`) and then restart - the
model just pulls all the images from all the subdirs.
"""

from bing_image_downloader import downloader

# Define function to download images
def download_images(query, num_images, output_dir):
    downloader.download(
        query,
        limit=num_images,
        output_dir=output_dir,
        adult_filter_off=False,  # ie keep the adult filter on
        force_replace=False,     # to get images between 100x100 and 1000x1000
        timeout=10,
        verbose=False
    )

# Download 10k cat images for training the model.
# This will make a subdir "cat" in data/train, but the unsupervised learning
# model doesn't care about "cat" vs other subdirs in data/train; it'll take all
# images from all subdirs in data/train.  But we want to stick with a single
# photo topic for the training at this point.
download_images("cat photo pet feline", 10000, "data/train")

# Then after model trained on just cats, let's see if it can distinguish some
# new cat pictures from not-cat pictures.  But note again the model will lump
# all these images together regardless of their "cat", "beachball", and "flower"
# subdirs if we use the data generator on the data/val directory, so will need
# to distinguish them separately.
# download_images("cat", 50, "data/val")
# download_images("beachball", 3, "data/val")
# download_images("flower", 5, "data/val")
