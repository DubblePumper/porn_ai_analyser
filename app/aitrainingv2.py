import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
import json
import logging
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import UnidentifiedImageError, Image
from tensorflow.python.platform import build_info as build

# Instellingen
MAX_EPOCHS = 20
BATCH_SIZE = 8
dataset_path = r"E:\github repos\porn_ai_analyser\app\datasets\pornstar_images"
performer_data_path = r"E:\github repos\porn_ai_analyser\app\datasets\performers_details_data.json"
output_dataset_path = r"E:\github repos\porn_ai_analyser\app\datasets\performer_images_with_metadata.npy"

# Zet de logging-configuratie op
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# TensorFlow versie en GPU informatie
print(f"tensorflow version: {tf.__version__}")
cuda_version = build.build_info.get('cuda_version', 'Not Available')
cudnn_version = build.build_info.get('cudnn_version', 'Not Available')
print(f"Cuda Version: {cuda_version}")
print(f"Cudnn version: {cudnn_version}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def count_performers_in_json(json_path):
    with open(json_path, 'r') as f:
        performer_data = json.load(f)
    return len(performer_data)

print("total performers found in jsonfile " + str(count_performers_in_json(performer_data_path)))

def count_subfolders(path):
    subfolders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    return len(subfolders)

print("total performers found subfolders " + str(count_subfolders(dataset_path)))
# Laad performer data uit JSON
with open(performer_data_path, 'r') as f:
    performer_data = json.load(f)

# Log the number of performers in the JSON file
logging.info(f"Total performers in JSON: {len(performer_data)}")

# Maak een dictionary van performer id naar gegevens
performer_info = {performer['slug']: performer for performer in performer_data}

# Controleer of het laden van de JSON correct was
logging.info(f"Gegevens van {len(performer_info)} performers geladen.")

# Check if TensorFlow can detect the GPU and required libraries are available
def check_gpu_availability():
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            logging.info(f"GPU found: {physical_devices[0].name}")
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

# Functie om afbeelding veilig te laden
def safe_load_img(path, target_size):
    if is_image_corrupt(path):
        logging.error(f"Corrupt image file: {path}")
        # Return a blank image if loading failed
        return np.zeros((target_size[0], target_size[1], 3))
    try:
        img = load_img(path, target_size=target_size)
        return img_to_array(img) / 255.0  # Normalize to 0-1
    except UnidentifiedImageError:
        logging.error(f"UnidentifiedImageError: cannot identify image file {path}")
        # Return a blank image if loading failed
        return np.zeros((target_size[0], target_size[1], 3))

# Create dataset with metadata
def create_dataset_with_metadata_from_json(performer_info, output_path):
    data = []
    performer_found = 0
    total_files_found = 0
    total_performers = len(performer_info)
    
    for performer in performer_info.values():
        performer_found += 1
        performer_images_found = 0
        total_images = len(performer['image_urls'])
        logging.info(f"Processing performer {performer_found}/{total_performers}: {performer['name']} with {total_images} images")
        
        for image_path in performer['image_urls']:
            performer_images_found += 1
            total_files_found += 1
            logging.info(f"Processing image {performer_images_found}/{total_images} for performer {performer_found}/{total_performers}")
            
            if os.path.isfile(image_path):
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
    
    logging.info("Finished processing all performers and images.")
    np.save(output_path, data)
    logging.info(f"Dataset with metadata saved to {output_path}")
    logging.info(f"Total performers processed: {performer_found}")
    logging.info(f"Total images processed: {total_files_found}")

# Create the dataset with metadata
create_dataset_with_metadata_from_json(performer_info, output_dataset_path)

# Load the dataset
dataset = np.load(output_dataset_path, allow_pickle=True)

# Prepare the data for training using tf.data.Dataset
def load_image_and_label(item):
    image = safe_load_img(item['image_path'], target_size=(224, 224))
    label = label_to_index[item['details']['slug']]
    return image, label

def data_generator(dataset):
    for item in dataset:
        yield load_image_and_label(item)

# Create a tf.data.Dataset from the generator
dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(dataset),
    output_signature=(
        tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)

# Split the dataset into training and validation sets
dataset_size = len(list(dataset))
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# Batch and prefetch the datasets
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Laad VGG16 zonder de laatste lagen
logging.info("Laad VGG16 model zonder top lagen...")
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Bevries de convolutionele lagen
logging.info("VGG16 model geladen en de convolutionele lagen bevroren.")

# Bouw het aangepaste model
logging.info("Bouwen van het aangepaste model...")
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(label_to_index), activation='softmax')  # Aantal performers
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
logging.info("Model gecompileerd met optimizer 'adam' en loss 'sparse_categorical_crossentropy'.")

# Train het model
logging.info("Start met trainen van het model...")
try:
    with tf.device('/GPU:0' if gpu_available else '/CPU:0'):
        history = model.fit(
            train_dataset,
            epochs=MAX_EPOCHS,
            validation_data=val_dataset
        )
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

# Sla het model op
logging.info("Sla het model op...")
model.save('performer_recognition_model.keras')  # Sla het getrainde model op als een bestand
logging.info("Model opgeslagen als performer_recognition_model.keras")

# Functie om voorspellingen te doen
def predict_performer(image_path):
    img = safe_load_img(image_path, target_size=(224, 224))  # Laad en verwerk de afbeelding
    prediction = model.predict(np.expand_dims(img, axis=0))  # Voorspel de performer
    
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