import cv2
import numpy as np
from PIL import Image
import tesserocr
import argparse

# Simple number plate detection using contours and OCR

def detect_plate(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 100, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    plates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if 2 < aspect_ratio < 6 and 1000 < w * h < 15000:
            plate_img = image[y:y+h, x:x+w]
            plates.append(plate_img)
    results = []
    for plate in plates:
        pil_img = Image.fromarray(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
        text = tesserocr.image_to_text(pil_img)
        results.append(text.strip())
    return results

def main():
    parser = argparse.ArgumentParser(description='Vehicle Number Plate Detection')
    parser.add_argument('--image', required=True, help='Path to input image')
    args = parser.parse_args()
    results = detect_plate(args.image)
    print('Detected Plates:')
    for plate_text in results:
        print(plate_text)

if __name__ == '__main__':
    main()
