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
MAX_EPOCHS = 10
BATCH_SIZE = 8
dataset_path = r"E:\github repos\porn_ai_analyser\app\datasets\pornstar_images"
performer_data_path = r"E:\github repos\porn_ai_analyser\app\datasets\performers_data.json"

# Zet de logging-configuratie op
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# TensorFlow versie en GPU informatie
print(f"tensorflow version: {tf.__version__}")
cuda_version = build.build_info.get('cuda_version', 'Not Available')
cudnn_version = build.build_info.get('cudnn_version', 'Not Available')
print(f"Cuda Version: {cuda_version}")
print(f"Cudnn version: {cudnn_version}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Laad performer data uit JSON
with open(performer_data_path, 'r') as f:
    performer_data = json.load(f)

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
    layers.Dense(len(performer_info), activation='softmax')  # Aantal performers
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
logging.info("Model gecompileerd met optimizer 'adam' en loss 'sparse_categorical_crossentropy'.")

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

# Functie om performer-details uit de filename te halen
def get_performer_details_from_filename(filename):
    # Haal de performer_slug uit de mapnaam, zet alles om naar kleine letters en vervang underscores door streepjes
    performer_slug = filename.split(os.sep)[-2].lower().replace('_', '-')
    logging.debug(f"Extracted performer slug: {performer_slug}")
    
    # Zoeken naar performer in de JSON-gegevens met dezelfde slug (ook omgezet)
    performer = None
    for p in performer_info.values():  # Correctly iterate over the values of the dictionary
        # Zorg ervoor dat de slug uit de JSON hetzelfde wordt aangepast (lowercase en underscores vervangen door streepjes)
        json_slug = p['slug'].lower().replace('_', '-')
        logging.debug(f"Comparing with JSON slug: {json_slug}")
        
        if performer_slug == json_slug:
            performer = p
            break
    
    if performer:
        return performer['name'], performer['birthday'], performer['ethnicity'], performer['hair_color'], performer['image_urls']
    else:
        logging.warning(f"Geen gegevens gevonden voor performer met slug: {performer_slug}")
        return None

# Custom function for preprocessing and augmentation
def custom_preprocessing_function(img):
    if img is None:
        return np.zeros((224, 224, 3))  # Return a blank image if loading failed
    return img

# Gebruik ImageDataGenerator voor data augmentatie en batch verwerking
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # Gebruik 20% van de data voor validatie
    horizontal_flip=True,
    rotation_range=30,
    zoom_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
    preprocessing_function=custom_preprocessing_function
)

# Pas de ImageDataGenerator aan om extra metadata toe te voegen
def custom_data_generator(generator):
    for data_batch, label_batch in generator:
        if data_batch is None or label_batch is None:
            logging.error("Received None data batch or label batch.")
            continue
        for i in range(len(generator.filenames)):
            logging.debug(f"Processing file: {generator.filenames[i]}")
        metadata = [get_performer_details_from_filename(generator.filenames[i]) for i in range(len(generator.filenames))]
        yield data_batch, label_batch, metadata

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training',
    shuffle=True
)

train_generator_with_metadata = custom_data_generator(train_generator)

# Zelfde voor de validatie generator
validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation',
    shuffle=True
)

validation_generator_with_metadata = custom_data_generator(validation_generator)

# Train het model
logging.info("Start met trainen van het model...")
try:
    with tf.device('/GPU:0' if gpu_available else '/CPU:0'):
        history = model.fit(
            train_generator_with_metadata,
            epochs=MAX_EPOCHS,
            validation_data=validation_generator_with_metadata
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
    predicted_performer = list(performer_info.keys())[predicted_class_index]
    performer_details = performer_info.get(predicted_performer)

    if performer_details:
        logging.info(f"Predicted performer: {performer_details['name']}")
        logging.info(f"Details: {performer_details['birthday']}, {performer_details['ethnicity']}, {performer_details['hair_color']}")
    else:
        logging.warning("Performer details not found.")