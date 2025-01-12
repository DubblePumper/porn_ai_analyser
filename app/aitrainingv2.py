import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import numpy as np
import tensorflow as tf
import json
from transformers import TFViTForImageClassification, ViTImageProcessor
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.data.experimental import cardinality
from PIL import UnidentifiedImageError, Image
from tensorflow.python.platform import build_info as build
import time
from tqdm import tqdm  # Add tqdm for progress bar
import psutil  # Add psutil for memory monitoring
import signal  # Add signal for timeout handling
from concurrent.futures import ThreadPoolExecutor, TimeoutError
# Debugging Class Weights
from collections import Counter
import logging  # Add logging module

# Suppress specific TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Instellingen
MAX_EPOCHS = 20
BATCH_SIZE = 8
dataset_path = r"E:\github repos\porn_ai_analyser\app\datasets\pornstar_images"
performer_data_path = r"E:\github repos\porn_ai_analyser\app\datasets\performers_details_data.json"
output_dataset_path = r"E:\github repos\porn_ai_analyser\app\datasets\performer_images_with_metadata.npy"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# TensorFlow versie en GPU informatie
logging.info(f"tensorflow version: {tf.__version__}")
cuda_version = build.build_info.get('cuda_version', 'Not Available')
cudnn_version = build.build_info.get('cudnn_version', 'Not Available')
logging.info(f"Cuda Version: {cuda_version}")
logging.info(f"Cudnn version: {cudnn_version}")
logging.info("Num GPUs Available: %d" % len(tf.config.list_physical_devices('GPU')))


def count_performers_in_json(json_path):
    logging.info(f"Counting performers in JSON file: {json_path}")
    with open(json_path, 'r') as f:
        performer_data = json.load(f)
    count = len(performer_data)
    logging.info(f"Number of performers: {count}")
    return count


def count_subfolders(path):
    logging.info(f"Counting subfolders in path: {path}")
    subfolders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    count = len(subfolders)
    logging.info(f"Number of subfolders: {count}")
    return count


# Laad performer data uit JSON
logging.info(f"Loading performer data from JSON file: {performer_data_path}")
with open(performer_data_path, 'r') as f:
    performer_data = json.load(f)
logging.info(f"Loaded performer data. Number of performers: {len(performer_data)}")

# Maak een dictionary van performer id naar gegevens
logging.info("Creating performer info dictionary.")
performer_info = {performer['slug']: performer for performer in performer_data}
logging.info(f"Performer info dictionary created. Number of entries: {len(performer_info)}")

# Define label_to_index
logging.info("Defining label to index mapping.")
labels = [performer['slug'] for performer in performer_data]
label_to_index = {label: index for index, label in enumerate(np.unique(labels))}
logging.info(f"Label to index mapping created. Number of labels: {len(label_to_index)}")


# Check if TensorFlow can detect the GPU and required libraries are available
def check_gpu_availability():
    logging.info("Checking GPU availability.")
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            logging.info("GPU is available.")
            return True
        else:
            logging.info("GPU is not available.")
            return False
    except Exception as e:
        logging.error(f"Error checking GPU availability: {e}")
        return False


gpu_available = check_gpu_availability()
logging.info(f"GPU available: {gpu_available}")


# Functie om te controleren of een afbeelding corrupt is
def is_image_corrupt(path):
    try:
        img = Image.open(path)
        img.verify()  # Verify that it is, in fact, an image
        return False
    except (UnidentifiedImageError, IOError):
        return True


# Timeout handler
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


# Functie om afbeelding veilig te laden met timeout
def safe_load_img_with_timeout(path, target_size, timeout=5):
    def load_image():
        if is_image_corrupt(path):
            return np.zeros((target_size[0], target_size[1], 3))
        try:
            img = load_img(path, target_size=target_size)
            img_array = img_to_array(img) / 255.0  # Normalize and ensure shape
            if img_array.shape != (224, 224, 3):  # Add explicit check
                img_array = np.resize(img_array, (224, 224, 3))
            return img_array
        except UnidentifiedImageError:
            return np.zeros((target_size[0], 224, 3))
        except Exception as e:
            logging.error(f"Error loading image: {e}")
            return np.zeros((target_size[0], target_size[1], 3))

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(load_image)
            return future.result(timeout=timeout)
    except TimeoutError:
        logging.error(f"Timeout loading image: {path}")
        return np.zeros((target_size[0], target_size[1], 3))


# Update the safe_load_img function to use the new timeout mechanism
def safe_load_img(path, target_size):
    image = safe_load_img_with_timeout(path, target_size)
    if image is None or not np.all(image.shape == (224, 224, 3)):
        logging.warning("Image is invalid, returning blank image.")
        image = np.zeros((224, 224, 3), dtype=np.float32)  # Fallback to a blank image
    return image


# Create dataset with metadata
def create_dataset_with_basic_info(performer_info, output_path):
    logging.info(f"Creating dataset with basic info. Output path: {output_path}")
    data = []
    total_images = sum(len(performer['image_urls']) for performer in performer_info.values())
    logging.info(f"Total images to process: {total_images}")

    with tqdm(total=total_images, desc="Processing images") as pbar:
        for performer in performer_info.values():
            valid_image_urls = []
            for image_path in performer['image_urls']:
                if os.path.isfile(image_path):
                    valid_image_urls.append(image_path)
                    data.append({
                        'image_path': image_path,
                        'id': performer.get('id', None),
                        'name': performer.get('name', None)
                    })
                pbar.update(1)
            performer['image_urls'] = valid_image_urls  # Update with valid image URLs only

    np.save(output_path, data)
    logging.info("Dataset with basic info created and saved.")


# Create the dataset with basic info
start_time = time.time()  # Start timing
logging.info("Starting dataset creation with basic info.")
create_dataset_with_basic_info(performer_info, output_dataset_path)
end_time = time.time()  # End timing
logging.info(f"Dataset creation complete. Time taken: {end_time - start_time:.2f} seconds")

# Load the dataset
logging.info("Loading the dataset.")
dataset = np.load(output_dataset_path, allow_pickle=True)
logging.info(f"Dataset loaded. Total items: {len(dataset)}")

# Create a mapping from performer IDs to integer labels
logging.info("Creating mapping from performer IDs to integer labels.")
performer_ids = [performer['id'] for performer in performer_data]
id_to_index = {id: index for index, id in enumerate(np.unique(performer_ids))}
logging.info(f"Mapping created. Number of unique performer IDs: {len(id_to_index)}")

# Create a list of image paths and labels
logging.info("Creating list of image paths and labels.")
image_paths = [item['image_path'] for item in dataset]
labels = [id_to_index.get(item['id'], None) for item in dataset if item['id'] in id_to_index and item['id'] is not None]
logging.info(f"Image paths and labels created. Number of labels: {len(labels)}")

# Ensure proper labeling by checking the labels and image paths
def verify_labels_and_images(image_paths, labels, description):
    logging.info(f"Verifying labels and images. Description: {description}")
    valid_image_paths = []
    valid_labels = []

    with tqdm(total=len(image_paths), desc=description) as pbar:
        for image_path, label in zip(image_paths, labels):
            if label is not None and isinstance(label, int) and os.path.isfile(image_path) and not is_image_corrupt(image_path) and 0 <= label < len(label_to_index):
                valid_image_paths.append(image_path)
                valid_labels.append(int(label))  # Ensure labels are integers
            else:
                logging.warning(f"Invalid label or image: label={label}, image_path={image_path}")
            pbar.update(1)

    logging.info(f"Verification complete. Valid images: {len(valid_image_paths)}")
    logging.info(f"Valid labels: {valid_labels[:10]}... (showing first 10 labels)")
    return valid_image_paths, valid_labels

logging.info("Verifying labels and images.")
image_paths, labels = verify_labels_and_images(image_paths, labels, "Verifying labels and images")
logging.info(f"Verification complete. Valid images: {len(image_paths)}")

# Filter out any remaining None labels
logging.info("Filtering out any remaining None labels.")
image_paths, labels = zip(*[(path, label) for path, label in zip(image_paths, labels) if label is not None])
logging.info(f"Filtering complete. Number of valid labels: {len(labels)}")

# Convert image_paths and labels to numpy arrays
logging.info("Converting image paths and labels to numpy arrays.")
image_paths = np.array(image_paths)
labels = np.array(labels)
logging.info("Conversion complete.")

# Ensure no None labels remain
if any(label is None for label in labels):
    raise ValueError("There are still None labels present after filtering.")

# Add image counter
total_images = len(image_paths)
processed_images = 0
logging.info(f"Total images: {total_images}, Processed images: {processed_images}")

# Function to load image and label
def load_image_and_label(image_path, label):
    def _load_image_and_label(image_path, label):
        try:
            # Convert image_path to string
            image_path = tf.compat.as_str_any(image_path.numpy())
            # Convert label to integer
            label = int(label.numpy())

            # Load and preprocess the image
            image = safe_load_img(image_path, target_size=(224, 224))
            if image.shape != (224, 224, 3):
                raise ValueError("Image shape mismatch")

            return image, label

        except Exception as e:
            logging.error(f"Error loading image and label: {e}")
            return np.zeros((224, 224, 3), dtype=np.float32), -1

    image, label = tf.py_function(func=_load_image_and_label, inp=[image_path, label], Tout=[tf.float32, tf.int32])
    image.set_shape((224, 224, 3))  # Ensure the shape is set correctly
    label.set_shape(())  # Ensure the shape is set correctly
    return image, label

# Simplified map function for clarity
def map_fn_with_counter(image_path, label):
    global processed_images
    try:
        # Ensure label is valid before processing the image
        if label == -1:
            raise ValueError("Invalid label detected during mapping")

        # Convert image_path and label to numpy values
        image_path = image_path.numpy().decode('utf-8')
        label = int(label.numpy())

        image, label = load_image_and_label(image_path, label)

        # Image shape validation
        if not np.all(image.shape == (224, 224, 3)):
            raise ValueError("Invalid image shape")

        processed_images += 1  # Update processed count
        return image, label
    except Exception as e:
        logging.error(f"Error in map_fn_with_counter: {e}")
        return tf.zeros((224, 224, 3), dtype=tf.float32), tf.constant(-1, dtype=tf.int32)

# Apply batching early
def create_batched_dataset(image_paths, labels, batch_size):
    global dataset_size  # Declare the global variable
    dataset_size = 0  # Initialize dataset size

    # Calculate the dataset size using a for loop
    for _ in image_paths:
        dataset_size += 1

    logging.info(f"Creating batched dataset. Dataset size: {dataset_size}, Batch size: {batch_size}")

    # Create a TensorFlow Dataset with image paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    # Apply mapping with the enhanced image-label loader
    logging.info("Mapping dataset with image-label loader.")
    dataset = dataset.map(
        lambda x, y: tf.py_function(func=load_image_and_label, inp=[x, y], Tout=(tf.float32, tf.int32)),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    logging.info("Filtering invalid labels.")
    dataset = dataset.filter(lambda image, label: tf.reduce_all(label >= 0))  # Filter out invalid labels

    # Apply batch and prefetch for performance
    logging.info("Batching and prefetching dataset.")
    dataset = dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda image, label: (tf.ensure_shape(image, [None, 224, 224, 3]), tf.ensure_shape(label, [None])), num_parallel_calls=tf.data.AUTOTUNE)
    
    # Set the shape and name for the dataset elements
    dataset = dataset.map(lambda image, label: (tf.identity(image, name="image"), tf.identity(label, name="label")), num_parallel_calls=tf.data.AUTOTUNE)
    
    logging.info("Batched dataset created successfully.")
    return dataset

# Create a batched tf.data.Dataset from the image paths and labels
logging.info("Creating batched dataset.")
dataset = create_batched_dataset(image_paths, labels, BATCH_SIZE)
logging.info("Batched dataset created.")

# Verify the dataset after filtering
logging.info("Verifying the dataset after filtering.")
for image, label in dataset.take(5):
    image.set_shape([None, 224, 224, 3])
    label.set_shape([None])
logging.info("Dataset verification complete.")

# Apply parallel mapping and prefetching
logging.info("Applying parallel mapping and prefetching.")
with tqdm(total=total_images, desc="Loading images") as pbar:
    def map_fn_with_progress(image_path, label):
        image, label = tf.py_function(func=map_fn_with_counter, inp=[image_path, label], Tout=(tf.float32, tf.int32))
        pbar.update(1)
        return image, label

    dataset = dataset.map(map_fn_with_progress, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.filter(lambda image, label: tf.reduce_all(label >= 0))  # Ensure invalid labels are filtered out
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    pbar.close()  # Ensure the progress bar is closed after processing
logging.info("Parallel mapping and prefetching complete.")

# Split the dataset into training and validation sets
logging.info("Splitting the dataset into training and validation sets.")
dataset_size = len(image_paths)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)
logging.info(f"Dataset split complete. Training size: {train_size}, Validation size: {val_size}")

# Batch and prefetch the datasets
logging.info("Batching and prefetching the datasets.")
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
logging.info("Batching and prefetching complete.")

logging.info("Loading the base model and feature extractor.")
base_model = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224", from_pt=True)
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
logging.info("Base model and feature extractor loaded.")

# Bevries alle lagen behalve de top lagen
logging.info("Freezing all layers except the top layers.")
for layer in base_model.vit.encoder.layer[:-1]:  # Bevries alle lagen behalve de laatste encoderlaag
    layer.trainable = False
logging.info("Layers frozen.")

# Data Augmentation
logging.info("Setting up data augmentation.")
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ]
)
logging.info("Data augmentation setup complete.")

# Bouw het aangepaste model
logging.info("Building the custom model.")
model = models.Sequential([
    data_augmentation,
    layers.Input(shape=(224, 224, 3)),  # Ensure the input shape is correct
    layers.Lambda(lambda x: tf.transpose(x, perm=[0, 3, 1, 2])),  # Transpose to (None, 3, 224, 224)
    base_model,
    layers.Lambda(lambda x: x.logits),  # Extract logits from TFSequenceClassifierOutput
    layers.Dense(512, activation='relu'),  # Directly use Dense layer
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(id_to_index), activation='softmax')  # Number of performers
])
logging.info("Custom model built.")

# Adjust the learning rate and compile the model
logging.info("Compiling the model.")
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
logging.info("Model compilation complete.")



# Count label occurrences
label_counts = Counter(labels)

# Log the first 10 label counts for debugging
logging.info(f"Label counts (first 10): {list(label_counts.items())[:10]}")

# Ensure all labels are integers
label_counts = {int(label): count for label, count in label_counts.items()}

# Compute class weights
total_samples = sum(label_counts.values())
class_weights = {label: total_samples / count for label, count in label_counts.items() if count > 0}

# Ensure class weights do not contain None values
class_weights = {label: weight for label, weight in class_weights.items() if label is not None}

# Normalize class weights to ensure they are within a reasonable range
max_weight = max(class_weights.values())
class_weights = {label: weight / max_weight for label, weight in class_weights.items()}

# Summarize class weights
min_weight = min(class_weights.values())
max_weight = max(class_weights.values())
avg_weight = sum(class_weights.values()) / len(class_weights)

logging.info(f"Class weights summary: min={min_weight}, max={max_weight}, avg={avg_weight}")
logging.info(f"Class weights: {list(class_weights.items())[:10]}... (showing first 10 weights)")

# Add detailed logging to debug class weights
for label, weight in class_weights.items():
    if not isinstance(label, int) or not isinstance(weight, (int, float)):
        logging.error(f"Invalid class weight: label={label}, weight={weight}")
        raise ValueError("Class weights are not correctly formatted. Ensure all labels are integers and weights are numeric.")

# Train the model with class weights
try:
    logging.info(f"Training for {MAX_EPOCHS} epochs.")
    if MAX_EPOCHS is None or MAX_EPOCHS <= 0:
        raise ValueError("Invalid number of epochs. Check MAX_EPOCHS value.")
    
    logging.info("Starting model training.")
    if gpu_available:
        logging.info("Using GPU for training.")
        with tf.device('/GPU:0'):
            epoch_pbar = tqdm(total=MAX_EPOCHS, desc="Training epochs")
            history = model.fit(
                train_dataset,
                epochs=MAX_EPOCHS,
                validation_data=val_dataset,
                class_weight=class_weights,
                callbacks=[
                    tf.keras.callbacks.LambdaCallback(
                        on_epoch_end=lambda epoch, logs: epoch_pbar.update(1),
                        on_train_batch_end=lambda batch, logs: None  # Disable batch logging
                    )
                ]
            )
            epoch_pbar.close()
    else:
        logging.info("Using CPU for training.")
        with tf.device('/CPU:0'):
            epoch_pbar = tqdm(total=MAX_EPOCHS, desc="Training epochs")
            history = model.fit(
                train_dataset,
                epochs=MAX_EPOCHS,
                validation_data=val_dataset,
                class_weight=class_weights,
                callbacks=[
                    tf.keras.callbacks.LambdaCallback(
                        on_epoch_end=lambda epoch, logs: epoch_pbar.update(1),
                        on_train_batch_end=lambda batch, logs: None  # Disable batch logging
                    )
                ]
            )
            epoch_pbar.close()
    logging.info("Model training complete.")
except Exception as e:
    logging.error(f"Error during model training: {e}")
    logging.error(f"MAX_EPOCHS: {MAX_EPOCHS}, gpu_available: {gpu_available}")
    logging.error(f"train_dataset: {train_dataset}, val_dataset: {val_dataset}")
    history = None

# Sla het model op
logging.info("Saving the model.")
model.build(input_shape=(None, 224, 224, 3))  # Define the input shape before saving
model.save("performer_recognition_model", save_format="tf")  # Sla het getrainde model op als een bestand
logging.info("Model saved successfully.")

# Evaluate the model after training
if val_size > 0:
    try:
        logging.info("Evaluating the model.")
        evaluation = model.evaluate(val_dataset)
        logging.info(f"Validation loss: {evaluation[0]:.4f}, Validation accuracy: {evaluation[1]:.4f}")
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
else:
    logging.warning("Validation dataset is empty. Skipping evaluation.")
