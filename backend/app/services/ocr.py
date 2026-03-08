import easyocr
import numpy as np

class OCRService:
    def __init__(self):
        # This downloads the model weights on the first run (approx 150MB)
        # 'en' for English; you can add 'fr', 'es', etc.
        self.reader = easyocr.Reader(['en'], gpu=False) 

    def scan_image(self, image_np: np.ndarray):
        """
        Scans the image and returns a list of text blocks with coordinates.
        """
        # detail=1 gives us bounding box, text, and confidence score
        results = self.reader.readtext(image_np)
        
        extracted_data = []
        for (bbox, text, prob) in results:
            # bbox is a list of 4 [x, y] coordinates: 
            # [top-left, top-right, bottom-right, bottom-left]
            (tl, tr, br, bl) = bbox
            
            extracted_data.append({
                "text": text,
                "confidence": float(prob),
                "coords": {
                    "x": int(tl[0]),
                    "y": int(tl[1]),
                    "w": int(tr[0] - tl[0]),
                    "h": int(bl[1] - tl[1])
                }
            })
            
        return extracted_data

# Initialize a singleton instance
ocr_engine = OCRService()