import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
import json
import logging
from transformers import TFViTForImageClassification, ViTFeatureExtractor
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import UnidentifiedImageError, Image
from tensorflow.python.platform import build_info as build
import time
from tqdm import tqdm  # Add tqdm for progress bar
import psutil  # Add psutil for memory monitoring
import signal  # Add signal for timeout handling
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Instellingen
MAX_EPOCHS = 20
BATCH_SIZE = 8
dataset_path = r"E:\github repos\porn_ai_analyser\app\datasets\pornstar_images"
performer_data_path = r"E:\github repos\porn_ai_analyser\app\datasets\performers_details_data.json"
output_dataset_path = r"E:\github repos\porn_ai_analyser\app\datasets\performer_images_with_metadata.npy"

# Zet de logging-configuratie op
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# TensorFlow versie en GPU informatie
logging.info(f"tensorflow version: {tf.__version__}")
cuda_version = build.build_info.get('cuda_version', 'Not Available')
cudnn_version = build.build_info.get('cudnn_version', 'Not Available')
logging.info(f"Cuda Version: {cuda_version}")
logging.info(f"Cudnn version: {cudnn_version}")
logging.info("Num GPUs Available: %d", len(tf.config.list_physical_devices('GPU')))

def count_performers_in_json(json_path):
    logging.info(f"Counting performers in JSON file: {json_path}")
    with open(json_path, 'r') as f:
        performer_data = json.load(f)
    count = len(performer_data)
    logging.info(f"Total performers found in JSON file: {count}")
    return count

logging.info("Total performers found in JSON file: %d", count_performers_in_json(performer_data_path))

def count_subfolders(path):
    logging.info(f"Counting subfolders in path: {path}")
    subfolders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    count = len(subfolders)
    logging.info(f"Total subfolders found: {count}")
    return count

logging.info("Total performers found in subfolders: %d", count_subfolders(dataset_path))

# Laad performer data uit JSON
logging.info(f"Loading performer data from JSON file: {performer_data_path}")
with open(performer_data_path, 'r') as f:
    performer_data = json.load(f)
logging.info(f"Loaded performer data for {len(performer_data)} performers.")

# Maak een dictionary van performer id naar gegevens
logging.info("Creating performer info dictionary...")
performer_info = {performer['slug']: performer for performer in performer_data}
logging.info(f"Created performer info dictionary with {len(performer_info)} entries.")

# Define label_to_index
logging.info("Creating label to index mapping...")
labels = [performer['slug'] for performer in performer_data]
label_to_index = {label: index for index, label in enumerate(np.unique(labels))}
logging.info(f"Label to index mapping created with {len(label_to_index)} labels.")

# Check if TensorFlow can detect the GPU and required libraries are available
def check_gpu_availability():
    logging.info("Checking GPU availability...")
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            logging.info(f"GPUs found: {[device.name for device in physical_devices]}")
            return True
        else:
            logging.warning("No GPU found. Using CPU instead.")
            return False
    except Exception as e:
        logging.error(f"Error checking GPU availability: {e}")
        return False

gpu_available = check_gpu_availability()

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
            logging.error(f"Corrupt image file: {path}")
            return np.zeros((target_size[0], target_size[1], 3))
        try:
            img = load_img(path, target_size=target_size)
            return img_to_array(img) / 255.0  # Normalize to 0-1
        except UnidentifiedImageError:
            logging.error(f"UnidentifiedImageError: cannot identify image file {path}")
            return np.zeros((target_size[0], target_size[1], 3))
        except Exception as e:
            logging.error(f"Error loading image file {path}: {e}")
            return np.zeros((target_size[0], target_size[1], 3))

    with ThreadPoolExecutor() as executor:
        future = executor.submit(load_image)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            logging.error(f"Timeout while loading image file: {path}")
            return np.zeros((target_size[0], target_size[1], 3))

# Update the safe_load_img function to use the new timeout mechanism
def safe_load_img(path, target_size):
    return safe_load_img_with_timeout(path, target_size)

# Create dataset with metadata
def create_dataset_with_metadata_from_json(performer_info, output_path):
    logging.info("Creating dataset with metadata from JSON...")
    data = []
    total_files_found = 0
    total_images = sum(len(performer['image_urls']) for performer in performer_info.values())
    
    with tqdm(total=total_images, desc="Processing images") as pbar:
        for performer in performer_info.values():
            valid_image_urls = []
            for image_path in performer['image_urls']:
                if os.path.isfile(image_path):
                    valid_image_urls.append(image_path)
                    data.append({
                        'image_path': image_path,
                        'details': {
                            'id': performer.get('id', None),
                            'slug': performer.get('slug', None),
                            'name': performer.get('name', None),
                            'bio': performer.get('bio', None),
                            'rating': performer.get('rating', None),
                            'is_parent': performer.get('is_parent', None),
                            'gender': performer.get('gender', None),
                            'birthday': performer.get('birthday', None),
                            'deathday': performer.get('deathday', None),
                            'birthplace': performer.get('birthplace', None),
                            'ethnicity': performer.get('ethnicity', None),
                            'nationality': performer.get('nationality', None),
                            'hair_color': performer.get('hair_color', None),
                            'eye_color': performer.get('eye_color', None),
                            'height': performer.get('height', None),
                            'weight': performer.get('weight', None),
                            'measurements': performer.get('measurements', None),
                            'waist_size': performer.get('waist_size', None),
                            'hip_size': performer.get('hip_size', None),
                            'cup_size': performer.get('cup_size', None),
                            'tattoos': performer.get('tattoos', None),
                            'piercings': performer.get('piercings', None),
                            'fake_boobs': performer.get('fake_boobs', None),
                            'same_sex_only': performer.get('same_sex_only', None),
                            'career_start_year': performer.get('career_start_year', None),
                            'career_end_year': performer.get('career_end_year', None),
                            'image_urls': performer.get('image_urls', None),
                            'image_amount': performer.get('image_amount', None),
                            'page': performer.get('page', None),
                            'performer_number': performer.get('performer_number', None),
                            'image_folder': performer.get('image_folder', None)  # Added field
                        }
                    })
                else:
                    logging.warning(f"Image file not found: {image_path}")
                
                pbar.update(1)
            performer['image_urls'] = valid_image_urls  # Update with valid image URLs only
    
    logging.info("Finished processing all performers and images.")
    np.save(output_path, data)
    logging.info(f"Dataset with metadata saved to {output_path}")
    logging.info(f"Total images processed: {total_files_found}")

# Monitor memory usage
def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logging.info(f"Memory usage: RSS={mem_info.rss / (1024 ** 2):.2f} MB, VMS={mem_info.vms / (1024 ** 2):.2f} MB")

# Create the dataset with metadata
logging.info("Creating dataset with metadata...")
start_time = time.time()  # Start timing
create_dataset_with_metadata_from_json(performer_info, output_dataset_path)
end_time = time.time()  # End timing
logging.info(f"Dataset creation complete. Time taken: {end_time - start_time:.2f} seconds")

# Load the dataset
logging.info("Loading dataset...")
dataset = np.load(output_dataset_path, allow_pickle=True)
logging.info("Dataset loaded.")

# Prepare the data for training using tf.data.Dataset
def load_image_and_label(image_path, label):
    try:
        image = safe_load_img(image_path.numpy().decode('utf-8'), target_size=(224, 224))
    except Exception as e:
        logging.error(f"Error loading image and label: {e}")
        image = np.zeros((224, 224, 3))
        label = -1
    return image, label

# Create a list of image paths and labels
image_paths = [item['image_path'] for item in dataset]
labels = [label_to_index[item['details']['slug']] for item in dataset]

# Ensure proper labeling by checking the labels and image paths
def verify_labels_and_images(image_paths, labels):
    for image_path, label in zip(image_paths, labels):
        if not os.path.isfile(image_path):
            logging.error(f"Image file not found: {image_path}")
        if label not in label_to_index.values():
            logging.error(f"Invalid label: {label}")

verify_labels_and_images(image_paths, labels)

# Apply batching early
def create_batched_dataset(image_paths, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.batch(batch_size)
    return dataset

# Create a batched tf.data.Dataset from the image paths and labels
logging.info("Creating batched tf.data.Dataset from image paths and labels...")
dataset = create_batched_dataset(image_paths, labels, BATCH_SIZE)

# Add image counter
total_images = len(image_paths)
processed_images = 0

def map_fn_with_counter(image_path, label):
    global processed_images
    try:
        image, label = tf.py_function(
            func=load_image_and_label, inp=[image_path, label], Tout=(tf.float32, tf.int32)
        )
        image = tf.ensure_shape(image, (224, 224, 3))
        label = tf.ensure_shape(label, ())
        processed_images += 1
        pbar.update(1)  # Update the progress bar
        return image, label
    except Exception as e:
        logging.error(f"Error in map_fn_with_counter: {e}")
        return tf.zeros((224, 224, 3), dtype=tf.float32), tf.constant(-1, dtype=tf.int32)

# Apply parallel mapping and prefetching
with tqdm(total=total_images, desc="Loading images") as pbar:
    dataset = dataset.map(map_fn_with_counter, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    pbar.close()  # Ensure the progress bar is closed after processing

logging.info("tf.data.Dataset created.")

# Split the dataset into training and validation sets
logging.info("Splitting dataset into training and validation sets...")
dataset_size = dataset.cardinality().numpy()
logging.info(f"Dataset size: {dataset_size}")
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)
logging.info(f"Dataset split: {train_size} training samples, {val_size} validation samples.")

# Batch and prefetch the datasets
logging.info("Batching and prefetching datasets...")
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
logging.info("Datasets batched and prefetched.")

# Laad Vision Transformer (ViT) zonder de laatste lagen
logging.info("Laad Vision Transformer (ViT) model zonder top lagen...")
base_model = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
base_model.trainable = False  # Bevries de convolutionele lagen
logging.info("Vision Transformer (ViT) model geladen en de convolutionele lagen bevroren.")

# Data Augmentation
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ]
)

# Bouw het aangepaste model
logging.info("Bouwen van het verbeterde model met Vision Transformer...")
model = models.Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(label_to_index), activation='softmax')  # Aantal performers
])
logging.info("Verbeterde model gebouwd met Vision Transformer.")

# Adjust the learning rate and compile the model
logging.info("Model gecompileerd met optimizer 'adam' en aangepaste learning rate.")
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Class weights to handle class imbalance
class_weights = {i: 1.0 for i in range(len(label_to_index))}
# Optionally, adjust weights based on class distribution

# Train het model met class weights
logging.info("Start met trainen van het model met class weights...")
try:
    if gpu_available:
        logging.info("Using GPU for training.")
        with tf.device('/GPU:0'):
            with tqdm(total=total_images, desc="Processing images") as pbar:
                history = model.fit(
                    train_dataset,
                    epochs=MAX_EPOCHS,
                    validation_data=val_dataset,
                    class_weight=class_weights,
                    callbacks=[
                        tf.keras.callbacks.LambdaCallback(
                            on_epoch_end=lambda epoch, logs: logging.info(
                                f"Epoch {epoch+1}/{MAX_EPOCHS} - "
                                f"Train loss: {logs['loss']:.4f}, "
                                f"Train accuracy: {logs['accuracy']:.4f}, "
                                f"Validation loss: {logs['val_loss']:.4f}, "
                                f"Validation accuracy: {logs['val_accuracy']:.4f}"
                            )
                        ),
                        tf.keras.callbacks.LambdaCallback(
                            on_epoch_end=lambda epoch, logs: log_memory_usage()
                        ),
                        tf.keras.callbacks.LambdaCallback(
                            on_batch_end=lambda batch, logs: pbar.update(BATCH_SIZE)
                        )
                    ]
                )
                pbar.close()  # Ensure the progress bar is closed after processing
                logging.info("Training completed successfully.")
    else:
        logging.info("Using CPU for training.")
        with tf.device('/CPU:0'):
            with tqdm(total=total_images, desc="Processing images") as pbar:
                history = model.fit(
                    train_dataset,
                    epochs=MAX_EPOCHS,
                    validation_data=val_dataset,
                    class_weight=class_weights,
                    callbacks=[
                        tf.keras.callbacks.LambdaCallback(
                            on_epoch_end=lambda epoch, logs: logging.info(
                                f"Epoch {epoch+1}/{MAX_EPOCHS} - "
                                f"Train loss: {logs['loss']:.4f}, "
                                f"Train accuracy: {logs['accuracy']:.4f}, "
                                f"Validation loss: {logs['val_loss']:.4f}, "
                                f"Validation accuracy: {logs['val_accuracy']:.4f}"
                            )
                        ),
                        tf.keras.callbacks.LambdaCallback(
                            on_epoch_end=lambda epoch, logs: log_memory_usage()
                        ),
                        tf.keras.callbacks.LambdaCallback(
                            on_batch_end=lambda batch, logs: pbar.update(BATCH_SIZE)
                        )
                    ]
                )
                pbar.close()  # Ensure the progress bar is closed after processing
                logging.info("Training completed successfully.")
except Exception as e:
    logging.error(f"Error during model training: {e}")
    history = None

# Logging per epoch
if history is not None:
    for epoch in range(MAX_EPOCHS):
        train_loss, train_acc = history.history['loss'][epoch], history.history['accuracy'][epoch]
        val_loss, val_acc = history.history['val_loss'][epoch], history.history['val_accuracy'][epoch]
        logging.info(f"Epoch {epoch+1} - Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")
        logging.info(f"Epoch {epoch+1} - Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")
else:
    logging.warning("Training history is None. No epochs to log.")

# Sla het model op
logging.info("Sla het model op...")
model.save('performer_recognition_model.keras')  # Sla het getrainde model op als een bestand
logging.info("Model opgeslagen als performer_recognition_model.keras")

# Evaluate the model after training
logging.info("Evaluating the model...")
evaluation = model.evaluate(val_dataset)
logging.info(f"Validation loss: {evaluation[0]:.4f}, Validation accuracy: {evaluation[1]:.4f}")

# Functie om voorspellingen te doen
def predict_performer(image_path):
    logging.info(f"Predicting performer for image: {image_path}")
    img = safe_load_img(image_path, target_size=(224, 224))  # Laad en verwerk de afbeelding
    inputs = feature_extractor(images=img, return_tensors="tf")
    prediction = model.predict(inputs['pixel_values'])  # Voorspel de performer
    
    # Verkrijg de index van de voorspelling
    predicted_class_index = np.argmax(prediction)
    
    # Verkrijg de performer details uit de JSON met de voorspelde index
    predicted_performer = list(label_to_index.keys())[predicted_class_index]
    performer_details = performer_info.get(predicted_performer)

    if performer_details:
        logging.info(f"Predicted performer: {performer_details['name']}")
        logging.info(f"Details: {performer_details['birthday']}, {performer_details['ethnicity']}, {performer_details['hair_color']}")
    else:
        logging.warning("Performer details not found.")