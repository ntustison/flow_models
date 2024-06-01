import os
import matplotlib.pyplot as plt
import numpy as np
import pprint
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array



def imgs_to_gaussian_pts(model, image_generator, n, neigvals=100):
    """
    choose neigvals<n
    """

    # Just making sure neigvals set correctly wrt number of samples:
    if neigvals > n:
        if n <= 100:
            neigvals = n
        else:
            neigvals = 100

    def get_n_images(data_generator, n):
        images = []
        while len(images) < n:
            img_batch = next(data_generator)
            for img in img_batch:
                images.append(img)
                if len(images) == n:
                    images_np = np.array(images)
                    images_tf = tf.convert_to_tensor(images_np, dtype=tf.float32)
                    return images_tf

    images = get_n_images(image_generator, n)
    images = tf.reshape(images, (-1, np.prod(images.shape[1:])))
    gaussian_points = model.call(images)
    gaussian_points = gaussian_points.numpy()

    if n > 10:
        pca = PCA(n_components=neigvals)
        reduced_data = pca.fit_transform(gaussian_points)
        mean=np.mean(gaussian_points, axis=0)
        reduced_cov = np.cov(reduced_data, rowvar=False)
    else:
        pca = None
        mean = None
        reduced_cov = None

    return gaussian_points, mean, reduced_cov, pca


def plot_gaussian_pts(training_pts, plotfile="pca_gaussian_points.png",
    mean=None, sim_pts=None, other_pts=None, other_pts_label="anomaly images",
    num_regen=None):

    pca = PCA(n_components=2)

    if sim_pts is not None and other_pts is not None:
        # Combine and find pca of combo plus origin(mean)
        pca_pts = pca.fit_transform(np.concatenate([sim_pts, other_pts, [mean]]))
        sim_pts = pca.transform(sim_pts)
        other_pts = pca.transform(other_pts)
        train_pts = pca.transform(training_pts)
    elif sim_pts is not None and other_pts is None:
        # Find pca of sim_pts plus origin(mean)
        pca_pts = pca.fit_transform(np.concatenate([sim_pts, [mean]]))
        sim_pts = pca.transform(sim_pts)
        train_pts = pca.transform(training_pts)
    elif sim_pts is None and other_pts is not None:
        # Find pca of other_pts plus origin(mean)
        pca_pts = pca.fit_transform(np.concatenate([other_pts, [mean]]))
        other_pts = pca.transform(other_pts)
        train_pts = pca.transform(training_pts)
    elif sim_pts is None and other_pts is None:
        # Find pca of just the training pts samples
        train_pts = pca.fit_transform(training_pts)

    fig, ax = plt.subplots()
    ax.scatter(train_pts[:, 0], train_pts[:, 1], color="C0", alpha=0.5, label="real images")

    if num_regen is not None:
        ax.scatter(train_pts[:num_regen, 0], train_pts[:num_regen, 1], label="regen images", facecolors='C0', alpha=0.5, edgecolors='k')

    if sim_pts is not None:
        ax.scatter(sim_pts[:, 0], sim_pts[:, 1], color="C1", label="sim images")
        for i in range(sim_pts.shape[0]):
            ax.annotate(str(i+1), (sim_pts[i, 0], sim_pts[i, 1]), textcoords="offset points", xytext=(0,0), ha='center', va='center')

    if other_pts is not None:
        print(f"2D coordinates of the {other_pts_label} points in the scatterplot:")
        print(other_pts)
        ax.scatter(other_pts[:, 0], other_pts[:, 1], color="chartreuse", label=other_pts_label)
        for i in range(other_pts.shape[0]):
            ax.annotate(str(i+1), (other_pts[i, 0], other_pts[i, 1]), textcoords="offset points", xytext=(0,0), ha='center', va='center')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # put axis just outside plot on right side
    plt.title("2D PCA of mapped gaussian points")
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.savefig(plotfile, bbox_inches="tight")


def generate_multivariate_normal_samples(mean, reduced_cov, pca, num_samples):  #, num_components=None, regularization=None):
    """
    num_components : 100 : first N eigenvalues/eigenvectors
    regularization : 1e-6 : water-level reg
    """

    # Generate new samples in reduced space
    new_samples_reduced = np.random.multivariate_normal(
        # (make 1D mean into 2D, then rotate/reduce it, then put back to 1D)
        mean=np.squeeze(pca.transform([mean])),
        cov=reduced_cov,
        size=num_samples
    )

    # Transform new samples back to original space
    new_samples = pca.inverse_transform(new_samples_reduced)
    new_samples_tf = tf.convert_to_tensor(new_samples, dtype=tf.float32)

    return new_samples_tf


def generate_imgs_in_batches(model, num_gen_images, mean, reduced_cov, pca, filename="sim_image", batch_size=10, regen=None):
    """
    batch_size: integer - note this is batches of generated images, not training data batches!
    regen: (optional) numpy array of training_pts for regenerating images for
           first N of them, instead of generating random pts from mean & cov
    """

    num_batches = (num_gen_images + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        # Determine the number of images to generate in this batch
        current_batch_size = min(batch_size, num_gen_images - batch_idx * batch_size)

        if regen is None:
            # Generate a batch of Gaussian samples using TensorFlow
            samples_tf = generate_multivariate_normal_samples(mean, reduced_cov, pca, current_batch_size)
        else:
            # Get next batch worth of points from supplied training_points
            regen_tf = tf.convert_to_tensor(regen, dtype=tf.float32)
            samples_tf = regen_tf[(batch_idx * batch_size):(batch_idx * batch_size + current_batch_size)]


        for i in range(current_batch_size):
            # Map back through the invertible network
            generated_image = model.inverse(samples_tf[i:i+1])
            generated_image = tf.reshape(generated_image, model.image_shape)

            # Save the generated image
            img = generated_image.numpy()
            img = (img * 255).astype(np.uint8)  # Convert back to uint8 format
            img_idx = batch_idx * batch_size + i + 1
            plt.imsave(f"{filename}_{img_idx}.png", img)

        print(f"Generated and saved {batch_idx * batch_size + current_batch_size} images out of {num_gen_images}")

    return samples_tf


def print_run_params(**kwargs):
    """Generic function to dump any args list to be listed in text file"""

    if "output_dir" not in kwargs:
        raise ValueError("print_run_params: error: 'output_dir' must be one of the kwargs.")

    print("Run params:", kwargs)

    output_dir = kwargs.pop("output_dir")
    file_path = output_dir + "/run_parameters.txt"
    with open(file_path, "w") as file:
        file.write("Parameters used in this run:\n")
        file.write(pprint.pformat(kwargs))

    print("")


def print_model_summary(model):
    # Print the header
    print(f"{'Layer Name':<20}{'Output Shape':<20}{'#Parameters':<11}")
    print("-" * 51)

    # Print the layer details
    for layer in model.layers:
        layer_name = layer.name
        output_shape = str(layer.output_shape)
        num_params = layer.count_params()
        print(f"{layer_name:<20}{output_shape:<20}{num_params:<10}")


def print_model_summary_nested(model):
    for layer in model.layers:
        print(layer.name)
        if hasattr(layer, 'layers'):
            for sub_layer in layer.layers:
                print(f"  {sub_layer.name}")


def image_data_generator(filenames, target_size=(224, 224), batch_size=1):
    if isinstance(filenames, str):
        filenames = [filenames]
    datagen = ImageDataGenerator()
    def generator():
        for filename in filenames:
            img = load_img(filename, target_size=target_size)
            x = img_to_array(img)
            x = tf.expand_dims(x, axis=0)
            yield datagen.flow(x, batch_size=batch_size)
    return generator


def slerp(point1, point2, t):
    omega = tf.acos(tf.clip_by_value(tf.tensordot(point1, point2, axes=1), -1.0, 1.0))
    sin_omega = tf.sin(omega)

    t1 = tf.sin((1 - t) * omega) / sin_omega
    t2 = tf.sin(t * omega) / sin_omega

    return t1 * point1 + t2 * point2


def interpolate_between_points(gaussian_points, N, path='euclidean'):
    """
    Interpolates N points between two high-dimensional points using a specified path.

    Parameters:
    gaussian_points (numpy array): A 2xD numpy array where D is the dimension of the points.
    N (int): Number of points to interpolate.
    path (str): The path type for interpolation ('euclidean' or 'slerp').

    Returns:
    numpy array: An NxD numpy array of interpolated points.
    """
    point1 = tf.constant(gaussian_points[0], dtype=tf.float32)
    point2 = tf.constant(gaussian_points[1], dtype=tf.float32)

    t_values = tf.linspace(0.0, 1.0, N)

    if path == 'euclidean':
        interpolated_points = [(1 - t) * point1 + t * point2 for t in t_values]
    elif path == 'slerp':
        interpolated_points = [slerp(point1, point2, t) for t in t_values]
    else:
        raise ValueError("Invalid path argument. Use 'euclidean' or 'slerp'.")

    return np.array(interpolated_points)
