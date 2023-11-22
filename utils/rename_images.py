import os

def rename_images(folder_path):
    # List all the files and filter out non-jpeg and hidden files
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg') and not f.startswith('.')])

    # Rename files to a temporary naming scheme to avoid conflicts
    temp_name = "temp_image_{}"
    for i, file in enumerate(files):
        os.rename(os.path.join(folder_path, file), os.path.join(folder_path, temp_name.format(i)))

    # List files again to get the temporary-named files
    temp_files = sorted([f for f in os.listdir(folder_path) if f.startswith('temp_image_')])

    # Rename files to their final naming scheme
    for i, file in enumerate(temp_files):
        new_file_name = f"image_{i}.jpg"
        os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_file_name))

    print(f"Renaming complete. {len(temp_files)} images have been renamed with no gaps in numbering.")

# Usage
folder_path = r'C:\Users\giudi\Desktop\BackUP\Programming\Python\Pixel-art\imgs\pixel-art\people'  # Replace with the actual path to your images
rename_images(folder_path)