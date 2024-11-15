from PIL import Image
import os

def removeCorruptedImages(dirPath):
    # Scan through each folder and subfolder to check all images
    corrupted_images = []
    for root, _, files in os.walk(dirPath):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Attempt to open the image file
                with Image.open(file_path) as img:
                    img.verify()  # Verify that the file is a valid image
            except (IOError, SyntaxError) as e:
                # If an error occurs, its corrupted
                print(f"Corrupted or invalid image found: {file_path}")
                #delete the file
                os.remove(file_path)
                corrupted_images.append(file_path)

    print(f"Total corrupted images found: {len(corrupted_images)}")