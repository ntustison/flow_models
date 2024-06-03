import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.preprocessing.image import ImageDataGenerator


print("Defining the data generators and model and training machinery...")

# Define image data generator with augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2
)

# Create data generators for training and validation
train_generator = datagen.flow_from_directory(
    'data/train',
    target_size=(128, 128),
    batch_size=32,
    class_mode=None,  # Unsupervised learning, no class labels
    subset='training'
)

val_generator = datagen.flow_from_directory(
    'data/val',
    target_size=(128, 128),
    batch_size=32,
    class_mode=None,  # Unsupervised learning, no class labels
    subset='validation'
)

# Define a bijector for RealNVP
bijector = tfp.bijectors.RealNVP(
    num_masked=128 * 128 * 3 // 2,
    shift_and_log_scale_fn=tfp.bijectors.real_nvp_default_template(hidden_layers=[256, 256])
)

# Chain multiple bijectors
flow_bijector = tfp.bijectors.Chain([bijector, tfp.bijectors.Permute(permutation=[1, 0]), bijector])

# Define the normalizing flow model
base_distribution = tfp.distributions.MultivariateNormalDiag(loc=[0.] * (128 * 128 * 3))
flow = tfp.distributions.TransformedDistribution(
    distribution=base_distribution,
    bijector=flow_bijector
)

# Define training loop
def train_flow_model(flow, train_generator, epochs=10):
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    for epoch in range(epochs):
        for images in train_generator:
            images = tf.reshape(images, (-1, 128 * 128 * 3))
            with tf.GradientTape() as tape:
                neg_log_likelihood = -tf.reduce_mean(flow.log_prob(images))
            gradients = tape.gradient(neg_log_likelihood, flow.trainable_variables)
            optimizer.apply_gradients(zip(gradients, flow.trainable_variables))
        print(f"Epoch {epoch + 1}/{epochs} completed")
    print("Model trained.")


# Train the model
print("Running the training loop...")
# train_flow_model(flow, train_generator, epochs=10)
train_flow_model(flow, train_generator, epochs=2)


# Generate Gaussian points from validation images and apply PCA for visualization
print("Generating/visualizing gaussian points from val images...")
val_images, _ = next(val_generator)
val_images = tf.reshape(val_images, (-1, 128 * 128 * 3))
gaussian_points = flow.bijector.forward(val_images)
gaussian_points = gaussian_points.numpy()

# Apply PCA to reduce to 2 dimensions
scaler = StandardScaler()
gaussian_points_scaled = scaler.fit_transform(gaussian_points)
pca = PCA(n_components=2)
pca_points = pca.fit_transform(gaussian_points_scaled)

# Plot the 2D points
plt.scatter(pca_points[:, 0], pca_points[:, 1])
plt.title("PCA of Gaussian Points")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.savefig("pca_gaussian_points.png")  # Save the plot as a PNG file
# plt.show()  # only relevant in jupyter notebook or locally-running console


print("Generating/visualizing new simulated images model...")

# Estimate the parameters of the multivariate Gaussian distribution
mean = np.mean(gaussian_points, axis=0)
covariance = np.cov(gaussian_points, rowvar=False)

# Generate new Gaussian samples
num_samples = 2  # number of new images to generate
new_gaussian_samples = np.random.multivariate_normal(mean, covariance, num_samples)

# Map these samples back through the invertible network
new_gaussian_samples_tf = tf.convert_to_tensor(new_gaussian_samples, dtype=tf.float32)
generated_images = flow.bijector.inverse(new_gaussian_samples_tf)
generated_images = tf.reshape(generated_images, (-1, 128, 128, 3))

# Save the generated images
output_dir = 'generated_images'
os.makedirs(output_dir, exist_ok=True)

for i in range(num_samples):
    img = generated_images[i].numpy()
    img = (img * 255).astype(np.uint8)  # Convert back to uint8 format
    plt.imsave(os.path.join(output_dir, f'generated_image_{i+1}.png'), img)

