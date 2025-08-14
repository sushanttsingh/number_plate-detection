import cv2
import numpy as np
from PIL import Image
import pytesseract
import os
import glob

# Simple number plate detection using contours and OCR

def detect_plate(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
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
        text = pytesseract.image_to_string(pil_img)
        results.append(text.strip())
    return results

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images_dir = os.path.join(script_dir, 'images')
    
    # Check if images directory exists
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        print("Please create an 'images' folder and place your images there.")
        return
    
    # Get all image files in the images directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, extension)))
        image_files.extend(glob.glob(os.path.join(images_dir, extension.upper())))
    
    # Filter out .gitkeep and other non-image files
    image_files = [f for f in image_files if not f.endswith('.gitkeep')]
    
    if not image_files:
        print(f"No image files found in {images_dir}")
        print("Supported formats: JPG, JPEG, PNG, BMP, TIFF")
        print("Please place your images in the 'images' folder and run again.")
        return
    
    print(f"Found {len(image_files)} image(s) in the images folder:")
    print("-" * 50)
    
    # Process each image
    for image_path in image_files:
        filename = os.path.basename(image_path)
        print(f"\nProcessing: {filename}")
        
        try:
            results = detect_plate(image_path)
            print(f"Detected plates in {filename}:")
            if results:
                for i, plate_text in enumerate(results, 1):
                    if plate_text.strip():  # Only show non-empty results
                        print(f"  Plate {i}: {plate_text}")
                if not any(result.strip() for result in results):
                    print("  No readable text detected")
            else:
                print("  No plates detected")
        except Exception as e:
            print(f"  Error processing {filename}: {str(e)}")
        
        print("-" * 30)

if __name__ == '__main__':
    main()
