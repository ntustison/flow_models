"""AWS based file utililities for convenience in Tensorflow/Keras modeling"""
import tensorflow as tf
import numpy as np
import boto3


class S3ImageDataGenerator:
    def __init__(self, rescale=None, horizontal_flip=False, zoom_range=(1.0, 1.0),
                 shear_range=0.0, rotation_range=0, width_shift_range=0.0,
                 height_shift_range=0.0):
        self.rescale = rescale
        self.horizontal_flip = horizontal_flip
        self.zoom_range = zoom_range
        self.shear_range = shear_range
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range

    def flow_from_directory(self, bucket, prefix, target_size=(256, 256),
                            batch_size=32, class_mode='binary', shuffle=True):
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        all_keys = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith(('.jpg', '.png'))]

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
                        raise ValueError("currently only class_mode = {None, 'input'} are implemented.")
                        # The following line relies on batch_labels defined in
                        # commented-out code above, after it's finished:
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
