from selenium import webdriver
from selenium.webdriver.common.by import By
from PIL import Image
from io import BytesIO
import time
import hashlib
import requests
import os

def image_hash(img_data):
    # Create a hash of the image data
    return hashlib.md5(img_data).hexdigest()

def download_images(base_url, download_path):
    # Initialize Selenium WebDriver
    driver = webdriver.Firefox()
    driver.get(base_url)
    time.sleep(5)  # Allow the page to load

    downloaded_hashes = set()  # Set to store image hashes
    image_number = 1  # Initialize image counter

    try:
        while True:  # You should replace this with some stopping condition
            # Scroll down
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

            # Extract URLs of images
            img_elements = driver.find_elements(By.XPATH, '//img')
            for img in img_elements:
                img_url = img.get_attribute('src')
                if img_url:
                    try:
                        response = requests.get(img_url, stream=True)
                        if response.status_code == 200:
                            img_data = response.content
                            img_hash = image_hash(img_data)

                            if img_hash not in downloaded_hashes:  # Check hash value for uniqueness
                                downloaded_hashes.add(img_hash)
                                img = Image.open(BytesIO(img_data))
                                if img.size != (30, 30):  # Skip if image is 30x30
                                    image_path = os.path.join(download_path, f'image_{image_number}.jpg')
                                    # Check if a file with the same name exists to avoid overwriting
                                    if not os.path.isfile(image_path):
                                        with open(image_path, 'wb') as handler:
                                            handler.write(img_data)
                                        image_number += 1
                    except requests.RequestException as e:
                        print(f"Request failed: {e}")
                    except IOError as e:
                        print(f"Image save failed: {e}")

            # Some stopping criteria should be here
            # For example, you can decide to stop after a certain condition is met

    finally:
        driver.quit()  # Ensure the driver quits

# Adjust the base URL and the download path as needed
download_images("https://br.pinterest.com/search/pins/?q=animals&rs=typed", r'C:\Users\giudi\Desktop\BackUP\Programming\Python\Pixel-art\imgs\real\animals')