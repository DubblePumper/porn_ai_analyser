import os
from PIL import Image

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
        return True
    except Exception as e:
        print(f"Corrupt bestand: {filepath} - Fout: {e}")
        return False

# Loop door alle subfolders en bestanden
for root, dirs, files in os.walk(main_folder):
    for file in files:
        file_path = os.path.join(root, file)
        if is_valid_image(file_path):
            valid_files.append(file_path)
        else:
            corrupt_files.append(file_path)

# Log resultaten
print(f"Aantal geldige bestanden: {len(valid_files)}")
print(f"Aantal corrupte bestanden: {len(corrupt_files)}")

# Optioneel: verwijder corrupte bestanden
remove_corrupt = input("Wil je corrupte bestanden verwijderen? (ja/nee): ").strip().lower()
if len(corrupt_files) == 0:
    print("Er zijn geen corrupte bestanden om te verwijderen.")
else:
    if remove_corrupt == "ja":
        for corrupt_file in corrupt_files:
            try:
                os.remove(corrupt_file)
                print(f"Verwijderd: {corrupt_file}")
            except Exception as e:
                print(f"Kon bestand niet verwijderen: {corrupt_file} - Fout: {e}")

print("Controle voltooid.")