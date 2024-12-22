import json
import os

# Path to the JSON file
JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets', 'performers_data.json')

def load_performers(json_path):
    if os.path.exists(json_path):
        with open(json_path, 'r') as file:
            return json.load(file)
    return []

def save_performers(json_path, performers):
    with open(json_path, 'w') as file:
        json.dump(performers, file, indent=4)

def merge_performers(existing, new):
    for key, value in new.items():
        if key in existing:
            if isinstance(existing[key], list) and isinstance(value, list):
                existing[key] = list(set(existing[key] + value))
            elif isinstance(existing[key], dict) and isinstance(value, dict):
                existing[key] = {**existing[key], **value}
            else:
                existing[key] = value
        else:
            existing[key] = value
    return existing

def flatten_performers(performers):
    flat_list = []
    for performer in performers:
        if isinstance(performer, list):
            flat_list.extend(performer)
        else:
            flat_list.append(performer)
    return flat_list

def remove_duplicates(performers):
    seen = {}
    duplicates = []
    unique_performers = []
    for performer in performers:
        identifier = performer.get('id') or performer.get('name').strip().lower()
        if identifier in seen:
            seen[identifier] = merge_performers(seen[identifier], performer)
            duplicates.append(identifier)
        else:
            seen[identifier] = performer
    print(f"Found {len(duplicates)} duplicates: {duplicates}")
    return list(seen.values())

def count_images(performers):
    total_images = 0
    for performer in performers:
        if 'image_urls' in performer and isinstance(performer['image_urls'], list):
            total_images += len(performer['image_urls'])
    return total_images

def main():
    
    # alura-jenson
    performers = load_performers(JSON_PATH)
    performers = flatten_performers(performers)
    unique_performers = remove_duplicates(performers)
    save_performers(JSON_PATH, unique_performers)
    total_images = count_images(unique_performers)
    print(f"Removed duplicates. Total performers: {len(unique_performers)}")
    print(f"Total images: {total_images}")

if __name__ == '__main__':
    main()
