"""AWS based file utililities for convenience in Tensorflow/Keras modeling"""
import tensorflow as tf
import numpy as np
import boto3


s3_paginator = boto3.client('s3').get_paginator('list_objects_v2')


def s3keys(s3_uri, start_after='', extensions=['png', 'jpg'], cycle=False, delimiter='/'):
    """
    Generate S3 object keys from a specified bucket/prefix with given extensions,
    optionally cycling indefinitely, e.g. to accommodate data augmentation.

    Parameters:
        s3_uri (str): S3 URI in the form "s3://<bucketname>/<prefix>"
        start_after (str): A key to start the listing after, to skip objects up to this key. Defaults to ''.
        extensions (list of str): A list of file extensions to filter objects by. Defaults to ['png', 'jpg'].
        cycle (bool): If True, the listing of keys will cycle indefinitely. Defaults to False.
        delimiter (str): The delimiter to use in listing. Defaults to '/'.

    Yields:
        str: S3 object keys that match the specified extensions.

    Raises:
        ValueError: If 'start_after' is set but does not start with the 'prefix'.

    Example1:
        # Example usage: List all '.png' and '.jpg' files in 'foo/bar/' directory of 'mybucket' bucket.
        for key in s3keys('s3://mybucket/foo/bar/', extensions=['png', 'jpg'], cycle=True):
            print('key:', key)

    Example2:
        # Start generator at file identified by start_after
        mykeys = s3keys('s3://aganse-images/train/cat', start_after='train/cat/flickr_cat_000012')
        next(mykeys)

    Note:
        Use the `cycle` parameter with caution as it can create an infinite loop.
        Ensure to handle this in your application logic to avoid endless execution.

    """
    print(f"beginning of s3uri={s3_uri[:5]}")
    if s3_uri[:5] == "s3://":
        bucket_name, _, prefix = s3_uri[5:].partition('/')
    else:
        raise ValueError("s3_uri should start with s3://...")
    print(f"bucket_name={bucket_name}, prefix={prefix}")
    prefix = prefix.lstrip(delimiter)
    if start_after and not start_after.startswith(prefix):  # Validation before modifying start_after
        raise ValueError("start_after must start with the prefix.")
    start_after = (start_after or prefix) if prefix.endswith(delimiter) else start_after
    while True:  # Loop to restart the generator for cycling
        for page in s3_paginator.paginate(Bucket=bucket_name, Prefix=prefix, StartAfter=start_after):
            for content in page.get('Contents', ()):
                if content['Key'].split('.')[-1] in extensions:
                    yield content['Key']
        if not cycle:  # Break the loop if cycling is not enabled
            break


class S3ImageDataGenerator:
    """
    Generates batches of tensor image data from an S3 bucket with real-time data
    augmentation.  Designed to be a drop-in replacement (or close) for Tensorflow's
    tensorflow.keras.preprocessing.image.ImageDataGenerator and its
    flow_from_directory() method, except pulling files from AWS S3 instead of
    local filesystem.

    Attributes (defaults are 0.0 when not specified otherwise):
        rescale (float): Rescaling factor. Defaults to None.
        horizontal_flip (bool): Randomly flip inputs horizontally. Default=False
        zoom_range (float): Range for random zoom between [1-zoom_range, 1+zoom_range].
        shear_range (float): Shear Intensity (Shear angle in counter-clockwise direction as radians)
        rotation_range (float): Degree range for random rotations.
        width_shift_range (float): Fraction of total width for random horizontal shifts.
        height_shift_range (float): Fraction of total height for random vertical shifts.

    Example:
        # Initialize generator with specific augmentation parameters
        generator = S3ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
        )
    """

    def __init__(self, rescale=None, horizontal_flip=False, zoom_range=0.0,
                 shear_range=0.0, rotation_range=0.0, width_shift_range=0.0,
                 height_shift_range=0.0):
        self.rescale = rescale
        self.horizontal_flip = horizontal_flip
        self.zoom_range = zoom_range
        self.shear_range = shear_range
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range

    def flow_from_directory(self, s3_uri, target_size=(256, 256),
                            batch_size=32, class_mode='binary', shuffle=True,
                            save_format=['png', 'jpg']):
        """
        Takes the path to an S3 bucket and generates batches of augmented/normalized data.

        Parameters:
            s3_uri (str): S3 URI in the form "s3://<bucketname>/<prefix>"
            extensions (list of str): List of acceptable image extensions.
            target_size (tuple of int): The dimensions to which all images found will be resized.
            batch_size (int): Size of the batch of images to return with each iteration.
            shuffle (bool): Whether to shuffle the order of images processed in the dataset.
            save_format (str, list, tuple): file extensions "png", "jpg", etc as individual
                                            string or as list or tuple of strings.

        Returns:
            A DirectoryIterator yielding tuples of (x, y) where x is a numpy array containing a batch
            of images with shape (batch_size, *image_size, 3) and y is a numpy array of corresponding labels.

        Example:
            train_generator = generator.flow_from_directory('mybucket', 'train_data/', batch_size=100,
                                                            shuffle=True, seed=42)
        """

        # s3 = boto3.client('s3')
        # response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        # all_keys = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith(('.jpg', '.png'))]
        all_keys = s3keys(s3_uri, start_after='', extensions=['png', 'jpg'], cycle=False)

        def generator():
            while True:
                if shuffle:
                    np.random.shuffle(all_keys)
                for i in range(0, len(all_keys), batch_size):
                    batch_keys = all_keys[i:i + batch_size]
                    batch_images = []
                    # batch_labels = []
                    for key in batch_keys:
                        img = self.download_image_from_s3(bucket, key)
                        img = self.preprocess_image(img, target_size)
                        batch_images.append(img)
                        # if class_mode != 'input' and class_mode is not None:
                        #     label = key.split('/')[-2]
                        #     if class_mode == 'binary':
                        #         label = (label == 'positive')  # Example binary condition
                        #     elif class_mode == 'sparse':
                        #         label = int(label)  # Example of converting label to integer
                        #     elif class_mode == 'categorical':
                        #         # Assume label_map is predefined dictionary mapping directory names to class indices
                        #         label = tf.keras.utils.to_categorical(label_map[label], num_classes)
                        #     batch_labels.append(label)
                    if class_mode == 'input':
                        yield np.array(batch_images), np.array(batch_images)
                    elif class_mode is None:
                        yield np.array(batch_images)
                    else:
                        raise ValueError("currently only class_modes {None, 'input'} are implemented yet.")
                        # The following line relies on batch_labels defined in
                        # commented-out lines above (which must be completed):
                        # yield np.array(batch_images), np.array(batch_labels)

        return generator()

    def download_image_from_s3(self, bucket, key):
        s3 = boto3.client('s3')
        response = s3.get_object(Bucket=bucket, Key=key)
        image_data = response['Body'].read()
        return tf.image.decode_jpeg(image_data, channels=3)

    def preprocess_image(self, image, target_size):
        image = tf.image.resize(image, target_size)
        if self.rescale:
            image = image * self.rescale
        if self.horizontal_flip and tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
        if self.rotation_range:
            angle = tf.random.uniform((), minval=-self.rotation_range, maxval=self.rotation_range)
            image = tf.image.rot90(image, k=int(angle / 90))
        if self.shear_range:
            intensity = tf.random.uniform((), minval=-self.shear_range, maxval=self.shear_range)
            image = self.apply_shear_transform(image, intensity)
            # image = self.apply_shear_transform(image, self.shear_range)
        if self.zoom_range != 0:
            image = self.apply_zoom(image, self.zoom_range, target_size)
        if self.width_shift_range:
            shift = tf.random.uniform((), -self.width_shift_range, self.width_shift_range)
            image = tf.image.translate(image, [shift * target_size[1], 0])
        if self.height_shift_range:
            shift = tf.random.uniform((), -self.height_shift_range, self.height_shift_range)
            image = tf.image.translate(image, [0, shift * target_size[0]])
        return image

    def apply_shear_transform(self, image, intensity):
        if intensity == 0:
            return image

        # Define the shear transformation matrix
        # Shear only along the x-axis for simplicity. For y-axis, modify the appropriate elements.
        shear_matrix = [1, -tf.sin(intensity), 0, 0, tf.cos(intensity), 0, 0, 0, 1]
        shear_matrix = tf.reshape(shear_matrix, [3, 3])

        # Convert image to homogeneous coordinates (add a dimension for the transformation)
        batched_image = tf.expand_dims(image, 0)  # Add batch dimension for compatibility with transform
        transformed_image = tf.raw_ops.ImageProjectiveTransformV3(
            images=batched_image,
            transforms=shear_matrix,
            output_shape=tf.shape(image),
            interpolation='BILINEAR'
        )

        return transformed_image[0]  # Remove batch dimension

    def apply_zoom(self, image, zoom_range, target_size):
        if zoom_range == 0:
            return image  # No zoom applied if the range is zero

        # Randomly choose a zoom factor within the specified range around 1
        # For example, if zoom_range is 0.2, then zoom between 0.8 and 1.2
        lower_bound = 1 - zoom_range
        upper_bound = 1 + zoom_range
        zoom_factor = tf.random.uniform((), minval=lower_bound, maxval=upper_bound)

        # Calculate new dimensions based on the zoom factor
        new_height = tf.cast(target_size[0] * zoom_factor, tf.int32)
        new_width = tf.cast(target_size[1] * zoom_factor, tf.int32)

        # Resize the image to the new dimensions
        zoomed_image = tf.image.resize(image, [new_height, new_width])

        # Crop or pad the zoomed image to match the target size
        # This ensures the output image has the same dimensions as the input image
        image = tf.image.resize_with_crop_or_pad(zoomed_image, target_size[0], target_size[1])
        return image


# Usage:
# datagen = S3ImageDataGenerator(
#     rescale=1./255,
#     horizontal_flip=True,
#     zoom_range=(0.8, 1.2),
#     shear_range=0.2,
#     rotation_range=30,
#     width_shift_range=0.1,
#     height_shift_range=0.1
# )
# train_generator = datagen.flow_from_directory(
#     "data/train",
#     target_size=(128, 128),  # images get resized to this size
#     batch_size=64,
#     class_mode=None,  # unsupervised learning so no class labels
#     shuffle=False,  # possibly helpful for training but pain for plot revamps/additions
# )
