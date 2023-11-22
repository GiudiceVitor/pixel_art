'''
This script downloads images from the pixilart gallery.
Author: Vitor Giudice on 2023-11-01
'''

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import StaleElementReferenceException  # Add this line
import requests
import os
import time
import re
from PIL import Image
import io

def webscrape(base_url):
    driver = webdriver.Firefox()
    driver.get(base_url)
    
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    processed_urls = set()
    extracted_urls = []
    
    while True:
        # Scroll and wait for the page to load
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        
        # Fetch all image elements after scrolling
        image_tags = driver.find_elements(By.XPATH, '//img')
        for index in range(len(image_tags)):
            try:
                # Fetch the image element by index
                img = image_tags[index]
                url = img.get_attribute('src')
                if url is None or url in processed_urls:
                    continue
                
                match = re.search(r'https://art\.pixilart\.com/.*', url)
                if match:
                    extracted_url = match.group()
                    extracted_urls.append(extracted_url)
                    processed_urls.add(url)
            except StaleElementReferenceException:
                # If stale element encountered, continue to the next element
                continue

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    
    driver.quit()
    print(f"Found {len(extracted_urls)} images.")
    return extracted_urls



def download_images(image_urls, dest_dir):
    # Create destination directory
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Download images
    for i, url in enumerate(image_urls):
        response = requests.get(url)
        
        # Check if image size is at least 100x100
        image = Image.open(io.BytesIO(response.content))
        if image.size[0] < 100 or image.size[1] < 100:
            continue

        file_path = os.path.join(dest_dir, f"image_{i}.jpg")
        with open(file_path, "wb") as file:
            file.write(response.content)

# Base URL
type = 'buildings'
base_url = f"https://www.pixilart.com/gallery/topics/{type}"
image_urls = webscrape(base_url)
download_images(image_urls, fr'C:\Users\giudi\Desktop\BackUP\Programming\Python\Pixel-art\almost-images\{type}')