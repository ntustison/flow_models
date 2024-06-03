import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Parameters for the modeling
num_epochs = 10
batch_size = 50  # 32
num_gen_images = 2  # 10  number of new images to generate
#image_size = (128, 128, 3)  # (height, width, channels)
image_size = (32, 32, 3)  # (height, width, channels)
flat_image_size = np.prod(image_size)  # flattened size
num_image_files = 50  # in practice this of course should be vastly larger!
steps_per_epoch = num_image_files // batch_size

print("num_epochs", num_epochs, ", batch_size", batch_size,
      ", num_gen_images", num_gen_images, ", image_size", image_size,
      ", image_size[:2]", image_size[:2],
      ", flat_image_size", flat_image_size, flush=True)


# Step 1: Download images into data/train/images dir - this is done using
# separate script download_images_bing.py since it's a one-time operation.


print("\n\nStarting Step 2: Create data generators for training and validation")
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2
)
train_generator = datagen.flow_from_directory(
    'data/train',
    target_size=image_size[:2],  # Use image_size variable
    batch_size=batch_size,
    class_mode=None,  # Unsupervised learning, no class labels
    subset='training'
)
# val_generator = datagen.flow_from_directory(
#     'data/val',
#     target_size=image_size[:2],  # Use image_size variable
#     batch_size=batch_size,
#     class_mode=None,  # Unsupervised learning, no class labels
#     subset='validation'
# )


print("\n\nStarting Step 3: Define RealNVP model")
# code generally follows Tensorflow documentation on:
# https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/RealNVP
bijector = tfp.bijectors.RealNVP(
    num_masked=flat_image_size // 2,
    shift_and_log_scale_fn=tfp.bijectors.real_nvp_default_template(hidden_layers=[256, 256])
)
flow_bijector = tfp.bijectors.Chain([bijector, tfp.bijectors.Permute(permutation=list(range(flat_image_size))), bijector])
base_distribution = tfp.distributions.MultivariateNormalDiag(loc=[0.] * flat_image_size)
flow = tfp.distributions.TransformedDistribution(
    distribution=base_distribution,
    bijector=flow_bijector
)


print("\n\nStarting Step 4: Train the model using the data generator")
def train_flow_model(flow, train_data_generator, epochs=10):
    optimizer = tf.optimizers.Adam(learning_rate=0.0001)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} starting")
        for step in range(steps_per_epoch):
            images = next(train_data_generator)
            images = tf.reshape(images, (-1, flat_image_size))  # Use flat_image_size variable
            with tf.GradientTape() as tape:
                neg_log_likelihood = -tf.reduce_mean(flow.log_prob(images))
            gradients = tape.gradient(neg_log_likelihood, flow.trainable_variables)
            optimizer.apply_gradients(zip(gradients, flow.trainable_variables))
            print("  neg_log_likelihood:", neg_log_likelihood, flush=True)
        print(f"Epoch {epoch + 1}/{epochs} completed")
    print("Model trained.")

# tf.debugging.set_log_device_placement(True)
# print(tf.config.list_physical_devices('GPU'))
train_flow_model(flow, train_generator, epochs=num_epochs)


print("\n\nStarting Step 5: Generate Gaussian points from validation images")
# val_images, _ = next(val_generator)
val_images, _ = next(train_generator)  # for starters use same images as train
val_images = tf.reshape(val_images, (-1, flat_image_size))  # Use flat_image_size variable
gaussian_points = flow.bijector.forward(val_images)
gaussian_points = gaussian_points.numpy()

# Apply PCA to reduce to 2 dimensions
scaler = StandardScaler()
gaussian_points_scaled = scaler.fit_transform(gaussian_points)
pca = PCA(n_components=2)
pca_points = pca.fit_transform(gaussian_points_scaled)

# Save the 2D points plot to a file
plt.scatter(pca_points[:, 0], pca_points[:, 1])
plt.title("PCA of Gaussian Points")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.savefig("pca_gaussian_points.png")  # Save the plot as a PNG file


print("\n\nStarting Step 6: Save new generated/simulated images")
# first estimate the parameters of the multivariate Gaussian distribution:
mean = np.mean(gaussian_points, axis=0)
covariance = np.cov(gaussian_points, rowvar=False)
# then generate new Gaussian samples and map back through the invertible network:
new_gaussian_samples = np.random.multivariate_normal(mean, covariance, num_gen_images)
new_gaussian_samples_tf = tf.convert_to_tensor(new_gaussian_samples, dtype=tf.float32)
generated_images = flow.bijector.inverse(new_gaussian_samples_tf)
generated_images = tf.reshape(generated_images, (-1, *image_size))  # Use image_size variable

output_dir = 'generated_images'
os.makedirs(output_dir, exist_ok=True)
for i in range(num_gen_images):
    img = generated_images[i].numpy()
    img = (img * 255).astype(np.uint8)  # Convert back to uint8 format
    plt.imsave(os.path.join(output_dir, f'generated_image_{i+1}.png'), img)
