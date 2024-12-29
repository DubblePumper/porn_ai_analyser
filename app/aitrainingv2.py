import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
print("TensorFlow version:", tf.__version__)  # Print TensorFlow version
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import UnidentifiedImageError, Image
from tensorflow.python.platform import build_info as build

MAX_EPOCHS = 10
BATCH_SIZE = 32

# Zet de logging-configuratie op
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print(f"tensorflow version: {tf.__version__}")
cuda_version = build.build_info.get('cuda_version', 'Not Available')
cudnn_version = build.build_info.get('cudnn_version', 'Not Available')
print(f"Cuda Version: {cuda_version}")
print(f"Cudnn version: {cudnn_version}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Check if TensorFlow can detect the GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("TensorFlow detected a GPU.")
    for device in physical_devices:
        print(f"Device: {device}")
else:
    print("TensorFlow did not detect a GPU.")

# Dataset paden
dataset_path = r"E:\github repos\porn_ai_analyser\app\datasets\pornstar_images"
performer_data_path = r"E:\github repos\porn_ai_analyser\app\datasets\performers_data.json"

# Laad performer namen uit de mappen
performer_names = os.listdir(dataset_path)  # Lijst van performers (mapnamen)
logging.info(f"Er zijn {len(performer_names)} performers gevonden.")

# Verify GPU availability
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    logging.info(f"GPU found: {physical_devices[0].name}")
else:
    logging.warning("No GPU found. Using CPU instead.")

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
    layers.Dense(len(performer_names), activation='softmax')  # Aantal performers
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
logging.info("Model gecompileerd met optimizer 'adam' en loss 'sparse_categorical_crossentropy'.")

# Custom function to handle image loading errors
def safe_load_img(path, target_size):
    try:
        img = load_img(path, target_size=target_size)
        return img_to_array(img) / 255.0  # Normalize to 0-1
    except UnidentifiedImageError:
        logging.error(f"UnidentifiedImageError: cannot identify image file {path}")
        # Return a blank image if loading failed
        return np.zeros((target_size[0], target_size[1], 3))

# Custom preprocessing function for ImageDataGenerator
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

logging.info("Creating train data generator...")
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training',
    shuffle=True  # Enable shuffling
)

logging.info("Creating validation data generator...")
validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation',
    shuffle=True  # Enable shuffling
)

# Custom PyDataset class
class PyDataset(tf.data.Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Additional initialization if needed

# Train het model met logging per epoch
logging.info("Start met trainen van het model...")
try:
    with tf.device('/GPU:0' if len(physical_devices) > 0 else '/CPU:0'):
        history = model.fit(
            train_generator,
            epochs=MAX_EPOCHS,
            validation_data=validation_generator
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