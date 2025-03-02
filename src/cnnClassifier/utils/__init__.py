import shutil
import os

def extract_and_save_model():
    source_path = 'artifacts/training/model.h5'
    destination_path = 'model/model.h5'

    # Ensure destination directory exists
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    # Copy the model file to the new location
    shutil.copy(source_path, destination_path)

    print(f'Model successfully extracted to {destination_path}')

def main():
    extract_and_save_model()

if __name__ == "__main__":
    main()