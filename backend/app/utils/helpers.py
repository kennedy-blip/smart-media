import cv2
import numpy as np
import base64
import io
from PIL import Image

def bytes_to_cv2(image_bytes: bytes):
    """Converts raw upload bytes to an OpenCV BGR image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def cv2_to_base64(image_cv2: np.ndarray):
    """Converts an OpenCV image to a Base64 string for the frontend."""
    _, buffer = cv2.imencode('.jpg', image_cv2)
    img_as_text = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_as_text}"

def apply_slight_blur(image_cv2: np.ndarray, strength=1):
    """
    To make text look 'unaltered', we add a tiny Gaussian blur.
    This helps the new text blend into the camera's natural sensor noise.
    """
    if strength == 0:
        return image_cv2
    return cv2.GaussianBlur(image_cv2, (3, 3), strength)