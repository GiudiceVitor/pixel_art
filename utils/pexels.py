import requests
import os

API_KEY = '<YOUR_API_KEY>'
SEARCH_QUERY = 'portrait'
PER_PAGE = 15
PAGE = 1
IMAGE_COUNTER = 0

headers = {
    'Authorization': API_KEY
}

save_directory = fr'C:\Users\giudi\Desktop\BackUP\Programming\Python\Pixel-art\imgs\real\{SEARCH_QUERY}'

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

try:
    while True:
        response = requests.get(f'https://api.pexels.com/v1/search?query={SEARCH_QUERY}&per_page={PER_PAGE}&page={PAGE}', headers=headers)
        data = response.json()

        # If no more images are found, exit the loop
        if not data['photos']:
            break

        for photo in data['photos']:
            img_url = photo['src']['original']
            img_response = requests.get(img_url, stream=True)

            if img_response.status_code == 200:
                filename = os.path.join(save_directory, f'image_{IMAGE_COUNTER}.jpg')
                with open(filename, 'wb') as f:
                    for chunk in img_response:
                        f.write(chunk)
                IMAGE_COUNTER += 1  # Increment the counter after saving each image

        PAGE += 1  # Go to the next page
except Exception as e:
    print(f"An error occurred: {e}")

print(f"Download completed. {IMAGE_COUNTER-1} images saved.")