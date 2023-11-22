import os
import shutil

def renumber_images(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    next_image_number = 1

    for subdir in sorted(os.listdir(source_folder)):
        subdir_path = os.path.join(source_folder, subdir)

        if os.path.isdir(subdir_path):
            for file in sorted(os.listdir(subdir_path)):
                if file.endswith('.jpg'):
                    print(file)
                    new_file_name = f"image_{next_image_number}.jpeg"
                    source_file = os.path.join(subdir_path, file)
                    destination_file = os.path.join(destination_folder, new_file_name)

                    shutil.copy(source_file, destination_file)
                    next_image_number += 1

# Usage
source = r'C:\Users\giudi\Desktop\BackUP\Programming\Python\Pixel-art\images' # Replace with your source folder path
destination = r'C:\Users\giudi\Desktop\BackUP\Programming\Python\Pixel-art\images\pixel-art'  # Replace with your destination folder path
renumber_images(source, destination)
 