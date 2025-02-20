import os
import sys
import gc
# Append the repository root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config  # Import global configuration
# tensorboard --logdir=./logs --bind_all
# streamlit run dashboard/dashboard.py
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Standaard Python modules
import os
import json
import time
import math
import signal
import shutil
import logging
import argparse
from functools import lru_cache
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import matplotlib.pyplot as plt
import io

# Externe libraries
import numpy as np
import cv2
import psutil
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import precision_score, recall_score, f1_score

# TensorFlow & Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, mixed_precision
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import (
    TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.data.experimental import cardinality
from tensorflow.python.platform import build_info as build
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

# TensorFlow Addons
import tensorflow_addons as tfa

# Hugging Face Transformers
from transformers import TFViTForImageClassification, ViTImageProcessor

# Global Constants
ENABLE_IMAGE_CACHE = False  # Set to False to disable image caching

# Move cache_key definition to the top, after imports
@lru_cache(maxsize=100000)
def cache_key(path):
    """Generate a unique cache key for an image path based on path and modification time."""
    try:
        mtime = os.path.getmtime(path)
        return f"{path}_{mtime}"
    except OSError:
        return path  # Fallback to just the path if can't get mtime

# Suppress specific TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Use global variables from config.py instead of command-line args:
BATCH_SIZE = config.BATCH_SIZE
MAX_EPOCHS = config.MAX_EPOCHS
dataset_path = config.DATASET_PATH
performer_data_path = config.PERFORMER_DATA_PATH
output_dataset_path = config.OUTPUT_DATASET_PATH
model_save_path = config.MODEL_SAVE_PATH
checkpoint_dir = config.CHECKPOINT_DIR
UNFREEZE_COUNT = config.UNFREEZE_COUNT
WHEN_INCLUDE_DATA_AUGMENTATION = config.DATA_AUGMENTATION_EPOCH_THRESHOLD
os.makedirs(checkpoint_dir, exist_ok=True)
best_model_path = os.path.join(checkpoint_dir, 'best_model')
latest_model_path = os.path.join(checkpoint_dir, 'latest_model')

# Add the custom learning rate schedule subclass to support multiplication
class MyExponentialDecay(tf.keras.optimizers.schedules.ExponentialDecay):
    def __mul__(self, other):
        # Return the initial learning rate multiplied by other.
        # This simple implementation is sufficient for the logging/scaling use-case.
        return self.initial_learning_rate * other

# Replace the create_learning_rate_schedule function with a custom schedule

class WarmUpExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, decay_steps, decay_rate):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return tf.cond(
            step < self.warmup_steps,
            lambda: self.initial_lr * (step / self.warmup_steps),
            lambda: self.initial_lr * (self.decay_rate ** ((step - self.warmup_steps) / self.decay_steps))
        )

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate
        }
    
    # Add multiplication operator support for scaling the schedule
    def __mul__(self, other):
        return WarmUpExponentialDecay(self.initial_lr * other, self.warmup_steps, self.decay_steps, self.decay_rate)

def create_learning_rate_schedule(warmup_steps=config.WARMUP_STEPS):
    initial_lr = config.INITIAL_LR
    decay_steps = config.DECAY_STEPS
    decay_rate = config.DECAY_RATE
    return WarmUpExponentialDecay(initial_lr, warmup_steps, decay_steps, decay_rate)

# 4) Use AdamW instead of Adam, remove clipnorm
def get_optimizer():
    """Create and return a new optimizer instance using TensorFlow Addons' AdamW."""
    return tfa.optimizers.AdamW(
        learning_rate=create_learning_rate_schedule(),
        weight_decay=1e-5
    )

def compile_model(model):
    """Compile model with consistent settings"""
    model.compile(
        optimizer=get_optimizer(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        ),
        metrics=['accuracy']
    )
    return model

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


# Add a global counter for logging intervals
image_counter = 0

# Add a flag to track if memory usage has been logged
memory_logged = False

# Add more detailed logging for memory usage at specific intervals
def log_memory_usage(message=""):
    global memory_logged
    if not memory_logged:
        mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        logging.info(f"{message} Current memory usage: {mem_usage:.1f} MB")
        memory_logged = True


# Add memory logging before and after loading images
cached_images_bar = tqdm(desc="Images cached", unit="image")

def safe_load_img_with_timeout(path, target_size, timeout=15):  # Increased default timeout from 5 to 15 seconds
    max_retries = 3  # New: Retry up to 3 times before giving up
    if not hasattr(safe_load_img_with_timeout, '_cache'):
        safe_load_img_with_timeout._cache = {}
    # Try to get from cache first
    if path in safe_load_img_with_timeout._cache:
        return safe_load_img_with_timeout._cache[path]
    
    for attempt in range(1, max_retries + 1):
        try:
            # ...existing code to load image...
            # For example purposes, call the original load function:
            image = load_img(path, target_size=target_size)  # Placeholder for actual image loading
            image = img_to_array(image)
            if image is not None and np.all(image.shape == (target_size[0], target_size[1], 3)):
                safe_load_img_with_timeout._cache[path] = image
                return image
        except (TimeoutError, Exception) as e:
            logging.warning(f"Attempt {attempt} failed for image: {path} with error: {e}")
            # Optionally add a short delay between retries here
    # Return blank image if all attempts fail
    blank = np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)
    safe_load_img_with_timeout._cache[path] = blank  # Cache the blank image to avoid retrying
    return blank


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
id_to_index = {performer_id: i for i, performer_id in enumerate(np.unique(performer_ids))}
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
            if label is not None and isinstance(label, int) and os.path.isfile(image_path) and not is_image_corrupt(
                    image_path) and 0 <= label < len(label_to_index):
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
            image_path = image_path.numpy().decode('utf-8') if isinstance(image_path.numpy(), bytes) else str(image_path.numpy())
            label = label.numpy()
            if isinstance(label, np.ndarray) and label.size == 1:
                label = label.item()

            # Load and preprocess the image with retry mechanism
            for attempt in range(2):  # Try twice
                image = safe_load_img(image_path, target_size=(224, 224))
                if not np.all(image == 0):  # If not a blank image
                    break
                time.sleep(0.1)  # Short delay before retry

            return image, label

        except Exception as e:
            logging.error(f"Error loading image and label: {e}")
            return np.zeros((224, 224, 3), dtype=np.float32), -1

    # Use tf.py_function
    image, label = tf.py_function(func=_load_image_and_label, inp=[image_path, label], Tout=(tf.float32, tf.int32))
    image.set_shape((224, 224, 3))
    label.set_shape([])  # Ensure label is a scalar
    return image, label


# Simplified map function for clarity
def map_fn_with_counter(image_paths, labels):
    global processed_images
    try:
        images = []
        new_labels = []
        for image_path, label in zip(image_paths, labels):
            image, label = load_image_and_label(image_path, label)

            # Validate shapes
            if not np.all(image.shape == (224, 224, 3)):
                raise ValueError("Invalid image shape")
            if isinstance(label, tf.Tensor):
                label = int(label.numpy())  # Convert EagerTensor to int
            if not isinstance(label, int):
                logging.error(f"Invalid label type: {type(label)} - {label}")
                raise ValueError("Invalid label type")

            images.append(image)
            new_labels.append(label)

        # Convert to TensorFlow tensors for further processing
        return tf.convert_to_tensor(images, dtype=tf.float32), tf.convert_to_tensor(new_labels, dtype=tf.int32)
    except Exception as e:
        logging.error(f"Error in map_fn_with_counter: {e}")
        # Return default tensors for error cases
        return tf.zeros((BATCH_SIZE, 224, 224, 3), dtype=tf.float32), tf.constant([-1] * BATCH_SIZE, dtype=tf.int32)


# Apply batching early
def create_batched_dataset(image_paths, labels, batch_size):
    logging.info(f"Creating batched dataset. Dataset size: {len(image_paths)}, Batch size: {batch_size}")
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.shuffle(buffer_size=config.SHUFFLE_BUFFER_SIZE)
    dataset = dataset.map(
        lambda x, y: tf.py_function(func=load_image_and_label, inp=[x, y], Tout=(tf.float32, tf.int32)),
        num_parallel_calls=config.NUM_PARALLEL_CALLS
    )
    dataset = dataset.map(
        lambda x, y: (tf.ensure_shape(x, [224, 224, 3]), tf.ensure_shape(y, []))
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Create a batched tf.data.Dataset from the image paths and labels
logging.info("Creating batched dataset.")
dataset = create_batched_dataset(image_paths, labels, BATCH_SIZE)
logging.info("Batched dataset created.")

# Verify the dataset after filtering
logging.info("Verifying the dataset after filtering.")
for image, label in dataset.take(5):
    logging.info(f"Image shape: {image.shape}, Label shape: {label.shape}, Label: {label.numpy()}")
    assert image.shape == (BATCH_SIZE, 224, 224, 3), "Incorrect image batch shape"
    assert label.shape == (BATCH_SIZE,), "Incorrect label batch shape"
logging.info("Dataset verification complete.")

# Split the dataset into training and validation sets
logging.info("Splitting the dataset into training and validation sets.")
dataset_size = len(image_paths)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)
logging.info(f"Dataset split complete. Training size: {train_size}, Validation size: {val_size}")

# Batch and prefetch the datasets
# Removed the second batch calls:
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
logging.info("Batching and prefetching complete.")

logging.info("Loading the base model and feature extractor.")
base_model = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224", from_pt=True)
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
logging.info("Base model and feature extractor loaded.")

logging.info(f"=====================================================================")
logging.info(f"Number of layers in the base model encoder: {len(base_model.vit.encoder.layer)}")

# Freeze fewer layers (unfreeze the last four)
logging.info("Freezing all layers except the top layers.")
# Log initial state
total_layers = len(base_model.vit.encoder.layer)
logging.info(f"Total encoder layers: {total_layers}")
logging.info(f"Unfreezing last {UNFREEZE_COUNT} layers")

# Freeze all encoder layers first
frozen_count = 0
for layer in base_model.vit.encoder.layer:
    layer.trainable = False
    frozen_count += 1
logging.info(f"Initially froze {frozen_count} layers")

# Unfreeze only the last UNFREEZE_COUNT layers
start_idx = total_layers - UNFREEZE_COUNT
unfrozen_count = 0
for i in range(start_idx, total_layers):
    base_model.vit.encoder.layer[i].trainable = True
    unfrozen_count += 1
    logging.info(f"Unfroze layer {i}")

logging.info(f"Final layer status: {frozen_count - unfrozen_count} frozen, {unfrozen_count} unfrozen")
logging.info(f"Layer freezing complete")
logging.info(f"=====================================================================")

# Define initial data augmentation (empty)
initial_data_augmentation = keras.Sequential([
    layers.Rescaling(1./255)  # At least one layer is required
])

# Define full data augmentation
full_data_augmentation = keras.Sequential([
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomBrightness(factor=0.2),
    layers.RandomContrast(factor=0.2),
])

# Set initial data augmentation
data_augmentation = initial_data_augmentation

# Modify create_model_architecture to use the current data_augmentation
# 5) Add a global average pooling layer before Dense
def create_model_architecture():
    """Create a consistent model architecture with proper layer sizes"""
    num_classes = len(id_to_index)
    base_model.gradient_checkpointing_enable()  # 6) Gradient checkpointing
    return FixInputSpecSequential([
        data_augmentation,
        layers.Input(shape=(224, 224, 3)),
        layers.Lambda(lambda x: tf.transpose(x, perm=[0, 3, 1, 2])),
        HFViTLogitsLayer(base_model),
        layers.GlobalAveragePooling2D(),  # reduce memory usage
        layers.BatchNormalization(),
        # First dense block
        layers.Dense(512, activation='relu'),  # Reduced size
        layers.BatchNormalization(), # Batch Normalization after Dense
        layers.Dropout(0.3),
        # Second dense block
        layers.Dense(256, activation='relu'),  # Reduced size
        layers.BatchNormalization(), # Batch Normalization after Dense
        layers.Dropout(0.3),
        # Output layer
        layers.Dense(num_classes)  # Final layer matches number of classes
    ])

class HFViTLogitsLayer(tf.keras.layers.Layer):
    def __init__(self, base_model, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model

    def call(self, inputs):
        outputs = self.base_model(inputs).logits
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({'base_model': None})
        return config

class FixInputSpecSequential(tf.keras.Sequential):
    """Custom Sequential model that handles input_spec properly."""
    
    def __init__(self, *args, **kwargs):
        # Remove name before passing to parent
        name = kwargs.pop('name', None)
        super().__init__(*args, **kwargs)
        if name:
            self._name = name
            
        # Set dtype policy after initialization
        self._dtype_policy = tf.keras.mixed_precision.global_policy()
        self._dtype = self._dtype_policy.compute_dtype

    def get_config(self):
        config = super().get_config()
        if "input_spec" in config:
            del config["input_spec"]
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if "input_spec" in config:
            del config["input_spec"]
        return super().from_config(config, custom_objects)

# Define custom_objects before using load_saved_model
custom_objects = {
    'HFViTLogitsLayer': HFViTLogitsLayer,
    'FixInputSpecSequential': FixInputSpecSequential,
    'LayerNormalization': tf.keras.layers.LayerNormalization,
    'Conv2D': tf.keras.layers.Conv2D,
    'Dense': tf.keras.layers.Dense,
    'Dropout': tf.keras.layers.Dropout,
    'BatchNormalization': tf.keras.layers.BatchNormalization,
    'Lambda': tf.keras.layers.Lambda,
    'RandomFlip': tf.keras.layers.RandomFlip,
    'RandomRotation': tf.keras.layers.RandomRotation,
    'RandomZoom': tf.keras.layers.RandomZoom,
    'RandomBrightness': tf.keras.layers.RandomBrightness,
    'RandomContrast': tf.keras.layers.RandomContrast
}

def inspect_saved_model(model_path):
    """Inspect saved model architecture to determine layer sizes"""
    try:
        reader = tf.train.load_checkpoint(os.path.join(model_path, 'variables', 'variables'))
        shape_map = reader.get_variable_to_shape_map()
        # Try to get dense layer sizes
        for var_name, shape in shape_map.items():
            if 'dense' in var_name.lower() and 'kernel' in var_name.lower():
                logging.info(f"Found dense layer: {var_name} with shape {shape}")
        return shape_map
    except Exception as e:
        logging.warning(f"Failed to inspect saved model: {e}")
        return None

def create_model_architecture():
    """Create a consistent model architecture with proper layer sizes"""
    num_classes = len(id_to_index)
    
    # Try to inspect existing model first
    latest_checkpoint = find_latest_checkpoint()
    if latest_checkpoint:
        shape_map = inspect_saved_model(latest_checkpoint)
        if shape_map:
            # Use the same architecture as saved model
            return FixInputSpecSequential([
                data_augmentation,
                layers.Input(shape=(224, 224, 3)),
                layers.Lambda(lambda x: tf.transpose(x, perm=[0, 3, 1, 2])),
                HFViTLogitsLayer(base_model),
                layers.BatchNormalization(),
                layers.Dense(512, activation='relu'),  # Match saved model size
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(num_classes)
            ])
    
    # Default architecture if no saved model found
    return FixInputSpecSequential([
        # ...same architecture as before...
    ])

def load_saved_model(model_path, custom_objects):
    """Load model with proper error handling and dtype policy initialization."""
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        return model
    except Exception as e:
        logging.warning(f"Standard loading failed, trying alternative method: {str(e)}")
        try:
            temp_model = create_model_architecture()
            temp_model.build((None, 224, 224, 3))
            
            # Try loading weights
            try:
                weights_path = os.path.join(model_path, 'variables', 'variables')
                temp_model.load_weights(weights_path)
            except:
                temp_model.load_weights(model_path)
            
            return temp_model
        except Exception as e2:
            logging.error(f"Both loading methods failed: {str(e2)}")
            return None

def find_latest_checkpoint():
    
    """Find the latest checkpoint in the checkpoint directory or logs directory."""
    directories = [checkpoint_dir, './logs']
    checkpoints = []
    
    for directory in directories:
        logging.info(f"Checking directory: {directory}")
        if not os.path.exists(directory):
            continue
        
        for dirname in os.listdir(directory):
            checkpoint_path = os.path.join(directory, dirname)
            if dirname.startswith('model_epoch_') and os.path.isdir(checkpoint_path):
                try:
                    epoch_num = int(dirname.split('_')[-1])
                    if os.path.exists(os.path.join(checkpoint_path, 'saved_model.pb')):
                        checkpoints.append((epoch_num, checkpoint_path))
                        logging.info(f"Found checkpoint: {checkpoint_path}")
                except ValueError:
                    continue
    
    if not checkpoints:
        return None
    
    # Sort by epoch number and get the latest
    latest_checkpoint = max(checkpoints, key=lambda x: x[0])[1]
    logging.info(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint

logging.info(f"=====================================================================")
# Replace the model loading section
logging.info("Loading saved model and stripping input_spec.")
model = None

latest_checkpoint = find_latest_checkpoint()  # Scan once
if latest_checkpoint:
    try:
        logging.info(f"Trying to load model from checkpoint: {latest_checkpoint}")
        model = load_saved_model(latest_checkpoint, custom_objects)
        logging.info(f"Loaded model from checkpoint: {latest_checkpoint}")
    except Exception as e:
        logging.warning("Standard loading failed, trying alternative method: " + str(e))
        try:
            # Alternative loading attempt using the same latest_checkpoint
            model = load_saved_model(latest_checkpoint, custom_objects)  # define alternative_load_model accordingly
            logging.info(f"Loaded alternative checkpoint model from: {latest_checkpoint}")
        except Exception as e_alt:
            logging.error("Alternative loading also failed: " + str(e_alt))

# If no checkpoint model was loaded, try the best model
if model is None and os.path.exists(best_model_path):
    try:
        model = load_saved_model(best_model_path, custom_objects)
        if model is not None:
            logging.info("Loaded best performing model.")
            model = compile_model(model)
            logging.info("Loaded model compiled successfully.")
    except Exception as e:
        logging.warning(f"Failed to load best model: {e}")
        model = None

# Only create a new model if no existing model could be loaded
if model is None:
    logging.info("No existing model found. Creating new model.")
    model = create_model_architecture()
    dummy_input = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
    model(dummy_input)
    model = compile_model(model)
    logging.info("New model created and compiled successfully.")
else:
    logging.info("Using loaded model, skipping build step.")
    # Just recompile the loaded model
    model = compile_model(model)

logging.info(f"=====================================================================")

# Save the initial model if it's new
if not os.path.exists(model_save_path):
    model.save(model_save_path, save_format='tf')
    logging.info(f"Initial model saved at {model_save_path}")

# Adjust the learning rate and compile the model
logging.info("Compiling the model.")

model = compile_model(model)
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

# Clamp the maximum class weight to 10.0
class_weights = {label: min(weight, 10.0) for label, weight in class_weights.items()}

# Apply log-based scaling to class weights
class_weights = {label: 1 + math.log1p(weight) for label, weight in class_weights.items()}

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
        raise ValueError(
            "Class weights are not correctly formatted. Ensure all labels are integers and weights are numeric.")

# Ensure MAX_EPOCHS is properly defined
assert MAX_EPOCHS is not None and MAX_EPOCHS > 0, "MAX_EPOCHS must be a positive integer."

# Ensure all labels are integers and not None
logging.info("Ensuring all labels are integers and not None.")
labels = [label for label in labels if label is not None]
labels = np.array(labels, dtype=int)
logging.info(f"Labels after filtering: {labels[:10]}... (showing first 10 labels)")

# Ensure class weights do not contain None values and are properly formatted
logging.info("Ensuring class weights do not contain None values and are properly formatted.")
class_weights = {label: weight for label, weight in class_weights.items() if
                 label is not None and isinstance(label, int) and isinstance(weight, (int, float))}
logging.info(f"Class weights after filtering: {list(class_weights.items())[:10]}... (showing first 10 weights)")

# Normalize class weights to ensure they are within a reasonable range
if class_weights:
    max_weight = max(class_weights.values())
    class_weights = {label: weight / max_weight for label, weight in class_weights.items()}
    logging.info(f"Class weights after normalization: {list(class_weights.items())[:10]}... (showing first 10 weights)")

# Summarize class weights
if class_weights:
    min_weight = min(class_weights.values())
    max_weight = max(class_weights.values())
    avg_weight = sum(class_weights.values()) / len(class_weights)

    logging.info(f"Class weights summary: min={min_weight}, max={max_weight}, avg={avg_weight}")
    logging.info(f"Class weights: {list(class_weights.items())[:10]}... (showing first 10 weights)")

# Add detailed logging to debug class weights
for label, weight in class_weights.items():
    if not isinstance(label, int) or not isinstance(weight, (int, float)):
        logging.error(f"Invalid class weight: label={label}, weight={weight}")
        raise ValueError(
            "Class weights are not correctly formatted. Ensure all labels are integers and weights are numeric.")

# Ensure no None labels remain
logging.info("Ensuring no None labels remain.")
if any(label is None for label in labels):
    raise ValueError("There are still None labels present after filtering.")

# Add detailed logging for dataset preparation
logging.info(f"Total images: {len(image_paths)}, Labels: {len(labels)}")
for i, label in enumerate(labels):
    if label is None:
        logging.error(f"Label at index {i} is None. Associated image: {image_paths[i]}")

# Add detailed logging for dataset elements
logging.info("Logging dataset elements for debugging.")
for image, label in dataset.take(5):
    logging.info(f"Image shape: {image.shape}, Label: {label}")

# Validate dataset contents
logging.info("Validating dataset contents.")
for x, y in train_dataset.take(1):
    assert x is not None, "Image data contains None."
    assert y is not None, "Label data contains None."
    logging.info(f"Sample image: {x.shape}, Label: {y}")


# Add check_class_weights function
def check_class_weights(class_weights):
    logging.info(f"=====================================================================")
    if not class_weights:
        logging.warning("No class weights found.")
        return
    min_w = min(class_weights.values())
    max_w = max(class_weights.values())
    ratio = max_w / min_w if min_w > 0 else float('inf')
    logging.info(f"Class weights range from {min_w:.3f} to {max_w:.3f}, ratio={ratio:.3f}")
    if ratio > 10:
        logging.warning("Class weight imbalance is very high, some classes may be undertrained.")

# Check class weights before training
logging.info("Checking class weights before training.")
check_class_weights(class_weights)
logging.info(f"=====================================================================")

# Check dataset consistency for shape and type
logging.info("Checking dataset consistency for shape and type.")
for images, labels in train_dataset.take(1):
    assert images is not None and labels is not None, "Dataset contains None."
    logging.info(f"Train batch shapes: {images.shape}, {labels.shape}")
    
for images, labels in val_dataset.take(1):
    logging.info(f"Doing some weird things....")
    assert images is not None and labels is not None, "Dataset contains None."
    logging.info(f"Validation batch shapes: {images.shape}, {labels.shape}")

# Calculate steps_per_epoch and validation_steps
steps_per_epoch = len(train_dataset) if hasattr(train_dataset, "__len__") else None
validation_steps = len(val_dataset) if hasattr(val_dataset, "__len__") else None

if steps_per_epoch is None or validation_steps is None:
    logging.error(f"Invalid dataset splitting: steps_per_epoch={steps_per_epoch}, validation_steps={validation_steps}")
else:
    logging.info(f"steps_per_epoch: {steps_per_epoch}, validation_steps: {validation_steps}")

class TQDMProgressBar(tf.keras.callbacks.Callback):
    def __init__(self, steps_per_epoch):
        self.steps_per_epoch = steps_per_epoch
        self.epoch_bar = None

    def on_epoch_begin(self, epoch, logs=None):
        if self.epoch_bar:
            self.epoch_bar.close()
        self.epoch_bar = tqdm(total=self.steps_per_epoch, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", position=0)

    def on_batch_end(self, batch, logs=None):
        self.epoch_bar.update(1)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_bar.close()

class DebugCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        log_memory_usage(f"Epoch {epoch} start")
        gc.collect()

    def on_epoch_end(self, epoch, logs=None):
        log_memory_usage(f"Epoch {epoch} end")
        gc.collect()

# Add LoggingModelCheckpoint class definition
class LoggingModelCheckpoint(ModelCheckpoint):
    """Custom ModelCheckpoint class that adds logging"""
    def __init__(self, filepath, **kwargs):
        super().__init__(filepath, **kwargs)
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        logging.info(f"Model checkpoint saved at epoch {epoch + 1}")


reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# Add early stopping callback definition before the callbacks list
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

class CustomTensorBoardCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric, value in logs.items():
            tf.summary.scalar(metric, value, step=epoch)
        tf.summary.flush()

logging.info("Adding callbacks.")
progress_bar_callback = TQDMProgressBar(steps_per_epoch=steps_per_epoch)
debug_callback = DebugCallback()
tensorboard_callback = TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='batch',
    profile_batch=0,
    write_steps_per_second=True
)
custom_tensorboard_callback = CustomTensorBoardCallback()
checkpoint_callback = LoggingModelCheckpoint(
    filepath=os.path.join('./logs', 'model_epoch_{epoch:02d}'),
    save_best_only=True,  # Save only the best model
    monitor='val_loss',
    mode='min',
    save_format='tf',
    verbose=1  # Add verbose output
)
best_checkpoint_callback = ModelCheckpoint(
    filepath=best_model_path,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    save_format='tf'
)

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset):
        super().__init__()
        self.val_dataset = val_dataset

    def on_epoch_end(self, epoch, logs=None):
        y_true_all = []
        y_pred_all = []
        for images, labels in self.val_dataset:
            logits = self.model.predict(images)
            # Apply softmax to the logits here for predictions
            probs = tf.nn.softmax(logits)
            y_pred_all.extend(np.argmax(probs, axis=1))
            y_true_all.extend(labels.numpy())
        precision_val = precision_score(y_true_all, y_pred_all, average='macro')
        recall_val = recall_score(y_true_all, y_pred_all, average='macro')
        f1_val = f1_score(y_true_all, y_pred_all, average='macro')
        logging.info(f"Epoch {epoch+1} - Precision: {precision_val:.4f}, Recall: {recall_val:.4f}, F1: {f1_val:.4f}")

# Add gradient norm logging callback
class GradientLoggingCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if batch % 100 == 0:  # Log every 100 batches
            grads = [layer.weights[0] for layer in self.model.trainable_layers if layer.weights]
            grad_norms = [tf.norm(grad).numpy() for grad in grads]
            logging.info(f"Batch {batch} - Gradient norms: {grad_norms}")

# Add a callback to switch to full data augmentation
class StagedAugmentationCallback(tf.keras.callbacks.Callback):
    def __init__(self, accuracy_threshold=WHEN_INCLUDE_DATA_AUGMENTATION * 0.01):
        super().__init__()
        self.accuracy_threshold = accuracy_threshold
        self.augmentation_enabled = False

    def on_epoch_end(self, epoch, logs=None):
        global data_augmentation
        current_accuracy = logs.get('accuracy', 0.0)
        
        if not self.augmentation_enabled and current_accuracy <= self.accuracy_threshold:
            logging.info(f"Accuracy ({current_accuracy:.4f}) below threshold ({self.accuracy_threshold:.4f}). Enabling data augmentation.")
            data_augmentation = full_data_augmentation
            self.augmentation_enabled = True
            
            # Create new model with same architecture
            new_model = create_model_architecture()
            
            # Get the current weights before recompiling
            old_weights = self.model.get_weights()
            
            # Compile new model with fresh optimizer
            new_model.compile(
                optimizer=get_optimizer(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True,
                    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
                ),
                metrics=['accuracy']
            )
            
            # Set the weights after compiling
            new_model.set_weights(old_weights)
            self.model = new_model
            
            logging.info("Model rebuilt with data augmentation enabled.")

# Update callbacks list to use modified StagedAugmentationCallback
staged_augmentation_callback = StagedAugmentationCallback(accuracy_threshold=0.02)  # 2% threshold

callbacks = [
    progress_bar_callback,
    debug_callback,
    tensorboard_callback,
    custom_tensorboard_callback,
    checkpoint_callback,
    best_checkpoint_callback,
    early_stopping_callback,
    reduce_lr_callback,  # Insert the ReduceLROnPlateau callback
    MetricsCallback(val_dataset),
    GradientLoggingCallback(),
    staged_augmentation_callback
]

logging.info("Callbacks added.")

def get_last_epoch_number():
    """Find the highest epoch number from existing checkpoints"""
    max_epoch = -1
    directories = [checkpoint_dir, './logs']
    
    for directory in directories:
        if not os.path.exists(directory):
            continue
            
        for dirname in os.listdir(directory):
            if dirname.startswith('model_epoch_'):
                try:
                    epoch_num = int(dirname.split('_')[-1])
                    max_epoch = max(max_epoch, epoch_num)
                except ValueError:
                    continue
    
    return max_epoch

# Update the checkpoint callback configuration
last_epoch = get_last_epoch_number()
initial_epoch = last_epoch + 1 if last_epoch >= 0 else 0
if initial_epoch > 0:
    initial_epoch -= 1

checkpoint_callback = LoggingModelCheckpoint(
    filepath=os.path.join('./logs', 'model_epoch_{epoch:02d}'),
    save_best_only=True,  # Save only the best model
    monitor='val_loss',
    mode='min',
    save_format='tf',
    verbose=1
)

# Add memory logging before and after model training
try:
    logging.info(f"Training for {MAX_EPOCHS} epochs.")
    logging.info("Starting model training.")
    progress_bar_callback = TQDMProgressBar(steps_per_epoch=steps_per_epoch)

    # Add DebugCallback to callbacks list
    debug_callback = DebugCallback()

    callbacks = [progress_bar_callback, debug_callback, tensorboard_callback, checkpoint_callback, early_stopping_callback, reduce_lr_callback]

    log_memory_usage("Before training")  # Log memory usage before training

    # Clear the TensorFlow session to free up memory
    tf.keras.backend.clear_session()

    if gpu_available:
        logging.info("Using GPU for training.")
        with tf.device('/GPU:0'):
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=MAX_EPOCHS,
                initial_epoch=initial_epoch,  # Add this line
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                class_weight=class_weights,
                callbacks=callbacks
            )
    else:
        logging.info("Using CPU for training.")
        with tf.device('/CPU:0'):
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=MAX_EPOCHS,
                initial_epoch=initial_epoch,  # Add this line
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                class_weight=class_weights,
                callbacks=callbacks
            )
    
    log_memory_usage("After training")  # Log memory usage after training
    logging.info("Model training complete.")
    logging.info(f"Training history: {history.history}")

except Exception as e:
    logging.error(f"Error during model training: {e}")
    logging.error(f"MAX_EPOCHS: {MAX_EPOCHS}, gpu_available: {gpu_available}")
    logging.error(f"train_dataset: {train_dataset}, val_dataset: {val_dataset}")
    history = None

# Save the label_to_index mapping
logging.info("Saving the label_to_index mapping.")
with open('label_to_index.json', 'w') as f:
    json.dump(label_to_index, f)
logging.info("label_to_index mapping saved successfully.")

# Ensure the input shape is known during evaluation
if val_size > 0 and validation_steps > 0:
    try:
        logging.info("Evaluating the model with a progress bar.")
        # Remove the old evaluation code:
        # for image, label in val_dataset.take(1):
        #     image.set_shape([None, 224, 224, 3])
        #     label.set_shape([None])
        # evaluation = model.evaluate(val_dataset)
        # logging.info(f"Validation loss: {evaluation[0]:.4f}, Validation accuracy: {evaluation[1]:.4f}")

        # Use a manual evaluation loop with TQDM instead:
        total_loss = 0.0
        total_acc = 0.0
        steps = 0
        with tqdm(total=len(val_dataset), desc="Evaluating", unit="batch") as pbar:
            for images, labels in val_dataset:
                result = model.test_on_batch(images, labels, reset_metrics=False)
                total_loss += result[0]
                total_acc += result[1]
                steps += 1
                pbar.update(1)

        avg_loss = total_loss / steps
        avg_acc = total_acc / steps
        logging.info(f"Validation loss: {avg_loss:.4f}, Validation accuracy: {avg_acc:.4f}")

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
else:
    logging.warning("Validation dataset is empty. Skipping evaluation.")

# Add cache for processed images
@lru_cache(maxsize=100000)
def cache_key(path):
    """Generate a unique cache key for an image path based on path and modification time."""
    try:
        mtime = os.path.getmtime(path)
        return f"{path}_{mtime}"
    except OSError:
        return path  # Fallback to just the path if can't get mtime

if history is not None and 'accuracy' in history.history and 'val_accuracy' in history.history:
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    diff = final_train_acc - final_val_acc
    logging.info(f"Final training accuracy: {final_train_acc:.4f}")
    logging.info(f"Final validation accuracy: {final_val_acc:.4f}")
    logging.info(f"Accuracy difference (train - val): {diff:.4f}")
    if diff > 0.1:
        logging.info("Potential overfitting detected. Consider regularization or early stopping.")

def create_model_architecture():
    """Create a consistent model architecture with proper layer sizes"""
    num_classes = len(id_to_index)
    
    return FixInputSpecSequential([
        data_augmentation,
        layers.Input(shape=(224, 224, 3)),
        layers.Lambda(lambda x: tf.transpose(x, perm=[0, 3, 1, 2])),
        HFViTLogitsLayer(base_model),
        layers.BatchNormalization(),
        # First dense block
        layers.Dense(768, activation='relu'),  # Match ViT hidden size
        layers.BatchNormalization(), # Batch Normalization after Dense
        layers.Dropout(0.3),
        # Second dense block
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(), # Batch Normalization after Dense
        layers.Dropout(0.3),
        # Output layer
        layers.Dense(num_classes)  # Final layer matches number of classes
    ])

def load_saved_model(model_path, custom_objects):
    """Load model with proper error handling and dtype policy initialization."""
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        return model
    except Exception as e:
        logging.warning(f"Standard loading failed, trying alternative method: {str(e)}")
        try:
            temp_model = create_model_architecture()
            temp_model.build((None, 224, 224, 3))
            
            # Try loading weights
            try:
                weights_path = os.path.join(model_path, 'variables', 'variables')
                temp_model.load_weights(weights_path)
            except:
                temp_model.load_weights(model_path)
            
            return temp_model
        except Exception as e2:
            logging.error(f"Both loading methods failed: {str(e2)}")
            return None


# Voeg onderstaande functies en callback toe

def compute_gradcam(model, img_tensor, class_index, conv_layer_name):
    # Bepaal de output van de gekozen convolutielaag
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(tf.multiply(weights[:, tf.newaxis, tf.newaxis, :], conv_outputs), axis=-1)
    cam = tf.maximum(cam, 0)
    heatmap = cam[0].numpy()
    # Normaliseer de heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap

class LiveVisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, sample_images, sample_labels, conv_layer_name, interval=1):
        super().__init__()
        self.log_dir = log_dir
        self.sample_images = sample_images  # Numpy array met enkele sample inputbeelden
        self.sample_labels = sample_labels  # Ground truth labels corresponderend aan sample_images
        self.conv_layer_name = conv_layer_name
        self.interval = interval
        self.file_writer = tf.summary.create_file_writer(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        # Neem een batch sample en haal de voorspelling op
        img_batch = tf.convert_to_tensor(self.sample_images, dtype=tf.float32)
        preds = self.model(img_batch, training=False)
        pred_probs = tf.nn.softmax(preds, axis=-1).numpy()
        pred_labels = pred_probs.argmax(axis=-1)
        
        # Creëer een visuele compositie per sample
        viz_images = []
        for i, img in enumerate(self.sample_images):
            # Bereken Grad-CAM heatmap voor de voorspelde klasse
            heatmap = compute_gradcam(self.model, tf.expand_dims(img, axis=0), pred_labels[i], self.conv_layer_name)
            # Overlay de heatmap op het origineel
            plt.figure(figsize=(4,4))
            plt.imshow(img.astype("uint8"))
            plt.imshow(heatmap, cmap='jet', alpha=0.5)
            plt.title(f"Pred: {pred_labels[i]} ({pred_probs[i][pred_labels[i]]*100:.1f}%)\nGT: {self.sample_labels[i]}")
            buf = io.BytesIO()
            plt.axis('off')
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            viz_img = tf.image.decode_png(buf.getvalue(), channels=4)
            viz_img = tf.expand_dims(viz_img, 0)
            viz_images.append(viz_img)
        if viz_images:
            concat_img = tf.concat(viz_images, axis=0)
            with self.file_writer.as_default():
                tf.summary.image("Live Visualisatie", concat_img, step=epoch, max_outputs=len(viz_images))

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("Error setting memory growth:", e)

# Enable mixed precision training for RTX 3060ti
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Enable XLA for improved performance
tf.config.optimizer.set_jit(True)

# 1) Switch to image_dataset_from_directory
def create_image_datasets(train_dir, val_split=0.2):
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset='training',
        seed=42,
        image_size=(224, 224),
        batch_size=config.BATCH_SIZE,
        label_mode='int'
    )
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset='validation',
        seed=42,
        image_size=(224, 224),
        batch_size=config.BATCH_SIZE,
        label_mode='int'
    )
    # Instead of caching everything in RAM, store on disk
    train_dataset = train_dataset.cache('train_dataset_cache.tf-data')
    val_dataset = val_dataset.cache('val_dataset_cache.tf-data')
    # 3) Prefetch
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    return train_dataset, val_dataset

# Replace manual dataset creation with image_dataset_from_directory
train_dataset, val_dataset = create_image_datasets(dataset_path, val_split=0.2)

