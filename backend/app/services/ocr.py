import pytesseract
import cv2

class OCREngine:
    def scan_image(self, img_cv2):
        # Convert to grayscale for better Tesseract results
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        
        # Get data from Tesseract (coordinates and text)
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        
        text_blocks = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 40:  # Confidence threshold
                text_blocks.append({
                    "text": data['text'][i],
                    "x": data['left'][i],
                    "y": data['top'][i],
                    "w": data['width'][i],
                    "h": data['height'][i]
                })
        return text_blocks

ocr_engine = OCREngine()