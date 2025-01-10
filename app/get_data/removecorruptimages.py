import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Pad naar de hoofdmap met subfolders
main_folder = r"E:\github repos\porn_ai_analyser\app\datasets\pornstar_images"

# Logs voor geldige en corrupte bestanden
valid_files = []
corrupt_files = []

# Functie om te controleren of een bestand een geldige afbeelding is
def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()  # Verifieer de afbeelding
        return filepath, True
    except Exception as e:
        print(f"Corrupt bestand: {filepath} - Fout: {e}")
        return filepath, False

# Functie om bestanden te controleren en te sorteren
def process_files(file_paths):
    with ThreadPoolExecutor() as executor:
        results = executor.map(is_valid_image, file_paths)
    for file_path, is_valid in results:
        if is_valid:
            valid_files.append(file_path)
        else:
            corrupt_files.append(file_path)

# Verzamel alle bestanden in de hoofdmap en subfolders
all_files = []
for root, dirs, files in os.walk(main_folder):
    for file in files:
        all_files.append(os.path.join(root, file))

# Controleer en sorteer bestanden
process_files(all_files)

# Log resultaten
print(f"Aantal geldige bestanden: {len(valid_files)}")
print(f"Aantal corrupte bestanden: {len(corrupt_files)}")

# Optioneel: verwijder corrupte bestanden
remove_corrupt = input("Wil je corrupte bestanden verwijderen? (ja/nee): ").strip().lower()
if len(corrupt_files) == 0:
    print("Er zijn geen corrupte bestanden om te verwijderen.")
else:
        for corrupt_file in corrupt_files:
            try:
                os.remove(corrupt_file)
                print(f"Verwijderd: {corrupt_file}")
            except Exception as e:
                print(f"Kon bestand niet verwijderen: {corrupt_file} - Fout: {e}")

print("Controle voltooid.")