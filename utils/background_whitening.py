import cv2
import numpy as np

def whiten_background(image_path, output_path, edge_thresh1, edge_thresh2):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges in the image
    edges = cv2.Canny(gray, edge_thresh1, edge_thresh2)

    # Dilate the edges to create a mask for the main object
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=1)

    # Find contours from the dilated edges
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for contours (filled areas)
    contour_mask = np.zeros_like(gray)
    cv2.drawContours(contour_mask, contours, -1, 255, thickness=cv2.FILLED)

    # Invert the contour mask to get the background
    background_mask = cv2.bitwise_not(contour_mask)

    # Create a white background and combine it with the original image
    white_background = np.full(image.shape, 255, dtype=np.uint8)
    background = cv2.bitwise_and(white_background, white_background, mask=background_mask)
    foreground = cv2.bitwise_and(image, image, mask=contour_mask)
    result = cv2.add(background, foreground)

    # Save the result
    cv2.imwrite(output_path, result)


for img in range(31, 32+1):
    whiten_background(fr"C:\Users\giudi\Desktop\BackUP\Programming\Python\Pixel-art\data\simple_imgs\pixel_art\{img}.png", fr"C:\Users\giudi\Desktop\BackUP\Programming\Python\Pixel-art\data\simple_imgs\pixel_art\{img}.png", 100, 200)