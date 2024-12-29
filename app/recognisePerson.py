# Laad het opgeslagen model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np
import json

loaded_model = tf.keras.models.load_model('performer_recognition_model.h5')
print("Model opnieuw geladen")

# Functie om een afbeelding te voorspellen
def predict_performer(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Batch dim toevoegen
    predictions = loaded_model.predict(img_array)
    predicted_label = np.argmax(predictions)
    return performer_names[predicted_label]

# Functie om performer gegevens op te halen uit het JSON-bestand
def get_performer_info(name):
    with open(r'E:\github repos\porn_ai_analyser\app\datasets\performers_data.json') as f:
        performer_data = json.load(f)
        
    for performer in performer_data:
        if performer['name'].lower() == name.lower():
            return performer
    return None

# Test de voorspelling en voeg performer informatie toe
test_image = r"E:\path\to\test_image.jpg"  # Geef hier het pad naar je testafbeelding
result = predict_performer(test_image)
performer_info = get_performer_info(result)

if performer_info:
    print(f"De performer is: {performer_info['name']}")
    print(f"Geslacht: {performer_info['gender']}")
    print(f"Verjaardag: {performer_info['birthday']}")
    print(f"Nationaliteit: {performer_info['birthplace']}")
else:
    print("Geen informatie gevonden voor de performer.")