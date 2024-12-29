import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import matplotlib.pyplot as plt
import numpy as np
import random
import warnings
import json
from torch.amp import GradScaler, autocast
import logging

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Zet de logging-configuratie op
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuratie
LEARNING_RATE = 0.0001
BATCH_SIZE = 10
NUM_EPOCHS = 100
IMG_SIZE = 160  # VGGFace2 gebruikt 160x160 afbeeldingen
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENABLE_RANDOMIZATION = True  # Boolean to enable or disable randomization
NUM_PREDICTIONS = 5000  # Global variable for the number of predictions
ENABLE_REALTIME_VISUALIZATION = True  # Global variable to enable or disable real-time visualization
ENABLE_JSON_LOGGING = False  # Global variable to enable or disable JSON logging
GRADUATION_RANDOMIZATION_AMOUNT = 3  # Gradually increase the randomization amount

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
CSV_PATH = os.path.join(SCRIPT_DIR, 'datasets', 'dataset.csv')
IMG_DIR = os.path.join(SCRIPT_DIR, 'datasets', 'recognize_person')
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, 'ai_things', 'saved_ai', 'person_recognition_model_vggface2.pth')
LOG_FILE_PATH = os.path.join(SCRIPT_DIR, 'prediction_log.json')
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Transformaties (meer randomisatie toegevoegd)
def get_transform(enable_randomization, phase):
    logging.info(f"Getting transform for phase {phase} with randomization {'enabled' if enable_randomization else 'disabled'}")
    if enable_randomization:
        if phase == 1:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif phase == 2:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                transforms.RandomAffine(30, shear=10),
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization for VGGFace2
            ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalisatie voor VGGFace2
        ])

# Dataset
class PersonDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        logging.info(f"Loading dataset from {csv_file}")
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Filter alleen rijen met geldige afbeeldingen
        self.data = self.data[self.data['Name'].apply(self._has_image)]
        unique_names = self.data['Name'].unique()
        self.labels = {name: idx for idx, name in enumerate(unique_names)}
        self.num_classes = len(self.labels)

        logging.info(f"Dataset loaded with {len(self.data)} valid images and {self.num_classes} classes")

    def _has_image(self, name):
        img_name = f"{name.replace(' ', '_')}.jpg"
        img_exists = os.path.exists(os.path.join(self.img_dir, img_name))
        return img_exists

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        name = row['Name']
        label = self.labels[name]  # Ensures that labels are reassigned

        img_name = f"{name.replace(' ', '_')}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, img_name

# Model
class CustomInceptionResnetV1(InceptionResnetV1):
    def __init__(self, *args, **kwargs):
        super(CustomInceptionResnetV1, self).__init__(*args, **kwargs)
        self.feature_dim = 512
        self.feature_layer = nn.Linear(self.last_linear.in_features, self.feature_dim)

    def forward(self, x, return_embeddings=False):
        # Extract features before the final layer
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        features = self.feature_layer(x)  # Reduce feature dimension to 512

        if return_embeddings:
            return features
        else:
            return self.classifier(features)

def build_model(num_classes):
    logging.info(f"Building model with {num_classes} classes")
    model = CustomInceptionResnetV1(pretrained='vggface2', classify=False).to(DEVICE)
    model.num_classes = num_classes  # Set the number of classes
    model.classifier = nn.Linear(model.feature_dim, num_classes).to(DEVICE)
    model.last_bn = nn.BatchNorm1d(num_classes).to(DEVICE)

    # Set batchnorm to evaluation mode
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.eval()
            m.track_running_stats = False

    return model

def show_images_realtime(images, labels, predictions, img_names, epoch, fig, axes, label_names):
    logging.info(f"Visualizing images for epoch {epoch}")
    # De-normalize the images
    mean = torch.tensor([0.485, 0.456, 0.406]).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(images.device)
    images = images * std[None, :, None, None] + mean[None, :, None, None]

    # Converteer tensor naar numpy array voor weergave
    images = images.cpu().numpy()
    grid = np.transpose(images, (0, 2, 3, 1))  # HWC (Height, Width, Channels)

    num_images = min(len(images), len(axes))  # Ensure we don't exceed the number of axes

    for i in range(num_images):
        ax = axes[i]
        ax.clear()
        ax.imshow(grid[i])

        true_label = labels[i].item()
        true_name = label_names[true_label]
        predicted_labels = predictions[:, i].cpu().numpy()
        predicted_names = [label_names[pred] for pred in predicted_labels]

        # Determine if the prediction is correct or incorrect
        is_correct = "Correct" if true_label in predicted_labels else "Incorrect"

        # Place each prediction side by side
        predicted_labels_str = " ".join(predicted_names)

        ax.set_title(f"True: {true_name}\nPred: {predicted_labels_str}\n{is_correct}", fontsize=8)
        ax.axis('off')

    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    fig.canvas.draw_idle()  # Efficiently update the figure
    plt.pause(0.001)  # Zorg ervoor dat de update zichtbaar is

def calculate_bonus(predictions, true_label):
    logging.info(f"Calculating bonus for true label {true_label}")
    # Calculate the bonus based on the distance of the predictions to the correct label
    distances = torch.abs(predictions - true_label)
    min_distance = torch.min(distances).item()
    if (min_distance <= 200):
        max_distance = torch.max(distances).item()
        bonuses = 1 - (distances.float() / max_distance)
        return bonuses.max().item()  # Return the maximum bonus for the closest prediction
    return 0.0  # No bonus if no prediction is within the max distance

def load_prediction_log():
    logging.info("Loading prediction log")
    if not ENABLE_JSON_LOGGING:
        return {}
    if os.path.exists(LOG_FILE_PATH):
        try:
            with open(LOG_FILE_PATH, 'r') as log_file:
                return json.load(log_file)
        except json.JSONDecodeError:
            logging.error("Error decoding JSON log file. Starting with an empty log.")
            return {}
    return {}

def save_prediction_log(log_data):
    logging.info("Saving prediction log")
    if not ENABLE_JSON_LOGGING:
        return
    with open(LOG_FILE_PATH, 'w') as log_file:
        json.dump(log_data, log_file, indent=4)

def log_predictions(log_data, img_name, true_label, predictions):
    logging.info(f"Logging predictions for image {img_name}")
    if not ENABLE_JSON_LOGGING:
        return
    is_correct = true_label in predictions
    predictions_tensor = torch.tensor(predictions)
    
    # Vind de voorspelling die het dichtst bij het juiste label ligt
    closest_prediction = predictions_tensor[torch.argmin(torch.abs(predictions_tensor - true_label))].item()

    if img_name in log_data:
        # Controleer of de huidige voorspelling dichterbij komt dan de vorige
        previous_best = log_data[img_name]['predictions'][0]  # Aangenomen dat de eerste voorspelling de beste is
        if abs(closest_prediction - true_label) < abs(previous_best - true_label):
            log_data[img_name]['predictions'][0] = closest_prediction  # Update met nieuwe beste voorspelling
        log_data[img_name]['is_correct'] = is_correct or log_data[img_name]['is_correct']
    else:
        log_data[img_name] = {
            'true_label': true_label,
            'predictions': [closest_prediction],
            'is_correct': is_correct
        }

def get_previous_predictions(img_name):
    logging.info(f"Getting previous predictions for image {img_name}")
    if not ENABLE_JSON_LOGGING:
        return []
    log_data = load_prediction_log()
    if img_name in log_data:
        return log_data[img_name]['predictions']
    return []

# Function to save the model state
def save_model_state(model, path):
    logging.info(f"Saving model state to {path}")
    torch.save(model.state_dict(), path)

# Function to load the model state
def load_model_state(model, path):
    logging.info(f"Loading model state from {path}")
    if os.path.exists(path):
        map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
        
        # Filter out mismatched keys
        model_state_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
        
        model_state_dict.update(filtered_state_dict)
        model.load_state_dict(model_state_dict)
        model.train()

# Trainen van het model
def train_model():
    logging.info("Starting model training")
    # Laad dataset
    phase = 1  # Start with phase 1 of randomization
    transform = get_transform(ENABLE_RANDOMIZATION, phase)
    dataset = PersonDataset(CSV_PATH, IMG_DIR, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)  # Increase num_workers

    model = build_model(dataset.num_classes)
    criterion = nn.CrossEntropyLoss()  # Gebruik CrossEntropyLoss voor eenvoud
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()  # Initialize GradScaler for mixed precision training

    # Maak de figuur voor visualisatie
    if ENABLE_REALTIME_VISUALIZATION:
        plt.ion()  # Enable interactive mode
        fig, axes = plt.subplots(4, 4, figsize=(16, 8))
        axes = axes.flatten()
        plt.show(block=False)  # Display the figure without blocking

    # Laad de log data
    log_data = load_prediction_log()

    # Begin training
    for epoch in range(NUM_EPOCHS):
        logging.info(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")

        # Load model state if it exists
        load_model_state(model, MODEL_SAVE_PATH)

        running_loss = 0.0
        correct = 0
        total = 0
        total_bonus = 0.0
        for i, (images, labels, img_names) in enumerate(dataloader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            # Mixed precision training
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # Verkrijg NUM_PREDICTIONS verschillende voorspellingen voor elke afbeelding in de batch
            predictions = []
            for _ in range(NUM_PREDICTIONS):
                with autocast(device_type='cuda'):
                    output = model(images)
                _, predicted = torch.max(output, 1)
                predictions.append(predicted)

            # Stapel de voorspellingen
            predictions = torch.stack(predictions)

            # Calculate accuracy
            most_common_predictions, _ = torch.mode(predictions, dim=0)
            correct += (most_common_predictions == labels).sum().item()
            total += labels.size(0)
            accuracy = correct / total

            # Calculate bonus per image
            correct_guesses = (most_common_predictions == labels).sum().item()
            incorrect_guesses = labels.size(0) - correct_guesses
            for j in range(len(labels)):
                total_bonus += calculate_bonus(predictions[:, j], labels[j])
                log_predictions(log_data, img_names[j], labels[j].item(), predictions[:, j].cpu().numpy())

            # Log batch progress
            if (i % BATCH_SIZE == 0):  # Log progress every batch
                logging.info(f"Epoch [{epoch + 1}] | Batch [{i}/{len(dataloader)}], Loss: {loss.item()}, Accuracy: {accuracy:.10f}, Bonus: {total_bonus / total:.4f}")
                logging.info(f"{correct_guesses} correct / {incorrect_guesses} incorrect - total guesses: {correct_guesses + incorrect_guesses}")

            # Visualiseer de afbeeldingen en de voorspellingen
            if ENABLE_REALTIME_VISUALIZATION and (i % BATCH_SIZE == 0):  # Update the plot every batch
                show_images_realtime(images, labels, predictions, img_names, epoch, fig, axes, dataset.labels)

        logging.info(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {running_loss / len(dataloader)}, Accuracy: {accuracy:.10f}, Bonus: {total_bonus / total:.4f}")

        # Bewaar de log data na elke epoch
        save_prediction_log(log_data)
        logging.info(f"Saved prediction log to {LOG_FILE_PATH}")

        # Bewaar het model na elke epoch
        save_model_state(model, MODEL_SAVE_PATH)
        logging.info(f"Saved model state to {MODEL_SAVE_PATH}")

        # Gradually increase the randomization phase
        if ENABLE_RANDOMIZATION and (epoch + 1) % (NUM_EPOCHS // GRADUATION_RANDOMIZATION_AMOUNT) == 0:
            phase += 1
            transform = get_transform(ENABLE_RANDOMIZATION, phase)
            dataset = PersonDataset(CSV_PATH, IMG_DIR, transform)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)  # Increase num_workers

if __name__ == '__main__':
    train_model()