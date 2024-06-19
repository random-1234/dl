import cv2
import numpy as np
from google.colab.patches import cv2_imshow

def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    equalized = cv2.equalizeHist(img)
    _, thresholded = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(img, 100, 200)
    flipped = cv2.flip(img, 1)
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    cv2_imshow(img)
    cv2_imshow(equalized)
    cv2_imshow(thresholded)
    cv2_imshow(edges)
    cv2_imshow(flipped)
    cv2_imshow(morphed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

process_image('/content/abcd.png')
