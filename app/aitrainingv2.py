import os
# tensorboard --logdir=./logs --bind_all
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
import argparse  # Add argparse for configuration
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler
import shutil
from tensorflow.keras import mixed_precision
from functools import lru_cache
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score
import math
import tensorflow_addons as tfa

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

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train an AI model to recognize a person inside an image.')
parser.add_argument('--batch_size', type=int, default=16, help='Further reduce batch size to help avoid OOM')  # Reduced batch size
parser.add_argument('--max_epochs', type=int, default=50, help='Increase epoch count to 50')
parser.add_argument('--dataset_path', type=str, default=r"E:\github repos\porn_ai_analyser\app\datasets\pornstar_images", help='Path to the dataset')
parser.add_argument('--performer_data_path', type=str, default=r"E:\github repos\porn_ai_analyser\app\datasets\performers_details_data.json", help='Path to the performer data JSON file')
parser.add_argument('--output_dataset_path', type=str, default=r"E:\github repos\porn_ai_analyser\app\datasets\performer_images_with_metadata.npy", help='Path to save the output dataset with metadata')
parser.add_argument('--model_save_path', type=str, default="performer_recognition_model", help='Path to save the trained model')
parser.add_argument('--checkpoint_dir', type=str, default='model_checkpoints', help='Directory to save model checkpoints')
args = parser.parse_args()

# Use parsed arguments
BATCH_SIZE = args.batch_size
MAX_EPOCHS = args.max_epochs
dataset_path = args.dataset_path
performer_data_path = args.performer_data_path
output_dataset_path = args.output_dataset_path
model_save_path = args.model_save_path  # Add this line
checkpoint_dir = args.checkpoint_dir
LEARNING_RATE = 0.0005  # Further reduced
UNFREEZE_COUNT = 10
os.makedirs(checkpoint_dir, exist_ok=True)
best_model_path = os.path.join(checkpoint_dir, 'best_model')
latest_model_path = os.path.join(checkpoint_dir, 'latest_model')

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

def safe_load_img_with_timeout(path, target_size, timeout=2):  # Reduced timeout to 2 seconds
    def load_image():
        try:
            # Try using cv2 first (faster)
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (target_size[1], target_size[0]))
                img_array = img.astype(np.float32) / 255.0
                return img_array
            
            # Fallback to PIL if cv2 fails
            img = Image.open(path)
            img = img.resize((target_size[1], target_size[0]))
            img_array = img_to_array(img) / 255.0
            return img_array
            
        except Exception as e:
            logging.warning(f"Error loading image {path}: {str(e)}")
            return np.zeros((target_size[0], target_size[1], 3))

    # Only use cache if enabled
    if ENABLE_IMAGE_CACHE:
        cache_str = cache_key(path)
        if hasattr(safe_load_img_with_timeout, 'cache') and cache_str in safe_load_img_with_timeout.cache:
            return safe_load_img_with_timeout.cache[cache_str]

    try:
        # Reduce max_workers
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(load_image)
            result = future.result(timeout=timeout)
            
            # Only cache if enabled
            if ENABLE_IMAGE_CACHE:
                if not hasattr(safe_load_img_with_timeout, 'cache'):
                    safe_load_img_with_timeout.cache = {}
                if cache_str not in safe_load_img_with_timeout.cache:
                    safe_load_img_with_timeout.cache[cache_str] = result
                    cached_images_bar.update(1)
            
            return result
    except TimeoutError:
        logging.warning(f"Timeout loading image: {path}")
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

    # Create a TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    # Add shuffling before mapping
    dataset = dataset.shuffle(buffer_size=5000)
    
    # Add caching after shuffling
    dataset = dataset.cache()

    # Map over dataset
    dataset = dataset.map(
        lambda x, y: tf.py_function(func=load_image_and_label, inp=[x, y], Tout=(tf.float32, tf.int32)),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Assert shapes of tensors
    dataset = dataset.map(
        lambda x, y: (
            tf.ensure_shape(x, [224, 224, 3]),  # Ensure image shape
            tf.ensure_shape(y, [])              # Ensure label shape
        )
    )

    # Batch dataset, ensuring no double batching
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
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

# Add RandomBrightness and RandomContrast to the data augmentation
logging.info("Setting up data augmentation.")
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(factor=0.2),
        layers.RandomContrast(factor=0.2),
    ]
)
logging.info("Data augmentation setup complete.")

class HFViTLogitsLayer(tf.keras.layers.Layer):
    def __init__(self, base_model):
        super(HFViTLogitsLayer, self).__init__(name='HFViTLogitsLayer')
        self.base_model = base_model  # No need for self.input_spec here

    def call(self, inputs):
        return self.base_model({"pixel_values": inputs}).logits

    def get_config(self):
        config = super().get_config()
        return config  # Ensure no `input_spec` is saved in config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

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

# Replace the model loading code
def load_saved_model(model_path, custom_objects):
    """Load model with proper error handling and dtype policy initialization."""
    try:
        # First try loading normally
        model = tf.keras.models.load_model(model_path, 
                                       custom_objects=custom_objects, 
                                       compile=False)
        return model
    except Exception as e:
        logging.warning(f"Standard loading failed, trying alternative method: {str(e)}")
        try:
            # Create model architecture - remove softmax activation
            temp_model = FixInputSpecSequential([
                data_augmentation,
                layers.Input(shape=(224, 224, 3)),
                layers.Lambda(lambda x: tf.transpose(x, perm=[0, 3, 1, 2])),
                HFViTLogitsLayer(base_model),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(len(id_to_index))  # Remove softmax activation
            ])
            
            # Try to load weights
            try:
                weights_path = os.path.join(model_path, 'variables', 'variables')
                temp_model.load_weights(weights_path)
            except:
                temp_model.load_weights(model_path)
            
            return temp_model
        except Exception as e2:
            logging.error(f"Both loading methods failed: {str(e2)}")
            return None

# ...existing code...
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

# Try loading the latest checkpoint first
latest_checkpoint = find_latest_checkpoint()
if (latest_checkpoint):
    try:
        model = load_saved_model(latest_checkpoint, custom_objects)
        if model is not None:
            logging.info(f"Loaded model from checkpoint: {latest_checkpoint}")
            # Compile the loaded model
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                        loss=tfa.losses.SigmoidFocalCrossEntropy(gamma=2.0),
                        metrics=['accuracy'])
            logging.info("Loaded checkpoint model compiled successfully.")
    except Exception as e:
        logging.warning(f"Failed to load checkpoint model: {e}")
        model = None

# If no checkpoint model was loaded, try the best model
if model is None and os.path.exists(best_model_path):
    try:
        model = load_saved_model(best_model_path, custom_objects)
        if model is not None:
            logging.info("Loaded best performing model.")
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                        loss=tfa.losses.SigmoidFocalCrossEntropy(gamma=2.0),
                        metrics=['accuracy'])
            logging.info("Loaded model compiled successfully.")
    except Exception as e:
        logging.warning(f"Failed to load best model: {e}")
        model = None

# Only create a new model if no existing model could be loaded
if model is None:
    logging.info("No existing model found. Creating new model.")
    model = FixInputSpecSequential([
        data_augmentation,
        layers.Input(shape=(224, 224, 3)),
        layers.Lambda(lambda x: tf.transpose(x, perm=[0, 3, 1, 2])),
        HFViTLogitsLayer(base_model),
        layers.BatchNormalization(),  # Add BatchNorm
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),  # Add BatchNorm
        layers.Dropout(0.1),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),  # Add BatchNorm
        layers.Dropout(0.1),
        layers.Dense(len(id_to_index))  # Remove softmax activation here
    ])
    
    # Compile with sparse categorical crossentropy and label smoothing
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=1.0
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,  # Keep this as True since we're using raw logits
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        ),
        metrics=['accuracy']
    )
    logging.info("New model created and compiled successfully.")

# Update the model compilation after loading
else:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=1.0,
            epsilon=1e-7
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,  # Keep this as True
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        ),
        metrics=['accuracy']
    )

logging.info(f"=====================================================================")

# Build the model with an input shape (only needed for new models)
if not model.built:
    model.build((None, 224, 224, 3))
    logging.info("Model built with input shape.")

# Save the initial model if it's new
if not os.path.exists(model_save_path):
    model.save(model_save_path, save_format='tf')
    logging.info(f"Initial model saved at {model_save_path}")

# Adjust the learning rate and compile the model
logging.info("Compiling the model.")
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0,  # Add gradient clipping
        epsilon=1e-7   # Increase epsilon for better numerical stability
    ),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,  # Change to True since we're using raw logits
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    ),
    metrics=['accuracy']
)
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
        self.epoch_start_time = time.time()
        log_memory_usage(f"Epoch {epoch+1} start")  # Log memory usage at the start of each epoch

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.epoch_start_time
        log_memory_usage(f"Epoch {epoch+1} end")  # Log memory usage at the end of each epoch
        mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        logging.info(f"Epoch {epoch+1} took {elapsed:.2f}s, memory usage ~ {mem_usage:.1f} MB")

# Add LoggingModelCheckpoint class definition
class LoggingModelCheckpoint(ModelCheckpoint):
    """Custom ModelCheckpoint class that adds logging"""
    def __init__(self, filepath, **kwargs):
        super().__init__(filepath, **kwargs)
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        logging.info(f"Model checkpoint saved at epoch {epoch + 1}")

# Add learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch > 10:
        lr = lr * 0.1
    return lr

# Add early stopping callback definition before the callbacks list
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

# Add learning rate scheduler callback
lr_scheduler_callback = LearningRateScheduler(lr_scheduler)

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
    save_best_only=False,
    save_freq='epoch',
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

callbacks = [
    progress_bar_callback,
    debug_callback,
    tensorboard_callback,
    custom_tensorboard_callback,
    checkpoint_callback,
    best_checkpoint_callback,
    early_stopping_callback,
    lr_scheduler_callback,
    MetricsCallback(val_dataset),
    GradientLoggingCallback()
]
logging.info("Callbacks added.")




# Add memory logging before and after model training
try:
    logging.info(f"Training for {MAX_EPOCHS} epochs.")
    logging.info("Starting model training.")
    progress_bar_callback = TQDMProgressBar(steps_per_epoch=steps_per_epoch)

    # Add DebugCallback to callbacks list
    debug_callback = DebugCallback()

    callbacks = [progress_bar_callback, debug_callback, tensorboard_callback, checkpoint_callback, early_stopping_callback, lr_scheduler_callback]

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
if val_size > 0:
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



