import os
import shutil

def gather_images(source_folders, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    count = 0
    for folder in source_folders:
        for filename in sorted(os.listdir(folder)):
            if filename.endswith(".jpg"):
                new_filename = f"image_{count}.jpg"
                source_path = os.path.join(folder, filename)
                dest_path = os.path.join(dest_folder, new_filename)
                shutil.move(source_path, dest_path)
                count += 1

source_folders = [r'C:\Users\giudi\Desktop\BackUP\Programming\Python\Pixel-art\imgs\real\peoples']  # Add your folder paths here
dest_folder = r'C:\Users\giudi\Desktop\BackUP\Programming\Python\Pixel-art\imgs\real\people'
gather_images(source_folders, dest_folder)