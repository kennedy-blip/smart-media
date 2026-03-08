import os
import sys
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class EditorService:
    def __init__(self):
        # 1. Detect if running as PyInstaller EXE
        if hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        else:
            # 2. Get the directory of the current file (editor.py)
            # Then move up one level to 'app', then look for 'fonts'
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_path = os.path.dirname(current_dir) # This lands in 'app'

        self.font_dir = os.path.join(base_path, "fonts")
        print(f"DEBUG: Font directory set to: {self.font_dir}")

    def overlay_text(self, background_cv2, text_metadata, font_choice="Roboto-Regular.ttf", use_auto_color=True, manual_color="#000000"):
        img_pil = Image.fromarray(cv2.cvtColor(background_cv2, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        for block in text_metadata:
            coords = block.get('coords', {})
            x, y, w, h = int(coords.get('x', 0)), int(coords.get('y', 0)), int(coords.get('w', 0)), int(coords.get('h', 0))
            display_text = block.get('text', "")
            
            if not display_text.strip():
                continue

            # Style Logic
            current_font = font_choice
            if h > 30:
                if "Roboto" in font_choice: current_font = "Roboto-BoldItalic.ttf"
                elif "Arimo" in font_choice: current_font = "Arimo-Bold.ttf"

            font_path = os.path.join(self.font_dir, current_font)
            font_size = max(8, int(h * 1.05)) 
            
            try:
                # If path exists, use it; otherwise, fallback to default
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                else:
                    print(f"WARNING: Font not found at {font_path}. Using default.")
                    font = ImageFont.load_default()
            except Exception as e:
                print(f"ERROR: Font loading failed: {e}")
                font = ImageFont.load_default()

            # --- DYNAMIC WIDTH & KERNING ---
            text_width = draw.textbbox((0, 0), display_text, font=font)[2]
            while text_width > w and font_size > 8:
                font_size -= 1
                font = ImageFont.truetype(font_path, font_size)
                text_width = draw.textbbox((0, 0), display_text, font=font)[2]

            spacing = 0
            if len(display_text) > 1 and text_width < w:
                spacing = (w - text_width) / (len(display_text) - 1)

            # --- COLOR SAMPLING ---
            if use_auto_color:
                sample_area = background_cv2[max(0, y-2):min(background_cv2.shape[0], y+h+2), 
                                             max(0, x-2):min(background_cv2.shape[1], x+w+2)]
                if sample_area.size > 0:
                    b, g, r = np.percentile(sample_area, 10, axis=(0, 1))
                    avg_color = (r + g + b) / 3
                    if avg_color > 215: text_color = (40, 40, 40)
                    elif (max(r, g, b) - min(r, g, b)) > 50: text_color = (0, 0, 0)
                    else: text_color = (int(r), int(g), int(b))
                else:
                    text_color = (0, 0, 0)
            else:
                hex_val = manual_color.lstrip('#')
                text_color = tuple(int(hex_val[i:i+2], 16) for i in (0, 2, 4))

            # --- RENDERING ---
            current_x = x
            for char in display_text:
                draw.text((current_x, y), char, font=font, fill=text_color)
                current_x += draw.textbbox((0, 0), char, font=font)[2] + spacing

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def apply_scan_degradation(self, img_cv2):
        if img_cv2 is None: return img_cv2
        h, w = img_cv2.shape[:2]
        blurred = cv2.GaussianBlur(img_cv2, (1, 1), 0)
        noise = np.random.normal(0, 1.2, (h, w, 3)).astype(np.int16)
        noise_img = np.clip(blurred.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noise_img

    def overlay_stamp(self, background_cv2, stamp_cv2, x, y, width, height, opacity=0.85):
        bg_pil = Image.fromarray(cv2.cvtColor(background_cv2, cv2.COLOR_BGR2RGB)).convert("RGBA")
        if stamp_cv2.shape[2] == 3:
            stamp_cv2 = cv2.cvtColor(stamp_cv2, cv2.COLOR_BGR2BGRA)
        stamp_pil = Image.fromarray(stamp_cv2).convert("RGBA")
        stamp_pil = stamp_pil.resize((int(width), int(height)), Image.Resampling.LANCZOS)
        alpha = stamp_pil.split()[3].point(lambda p: p * opacity)
        stamp_pil.putalpha(alpha)
        bg_pil.paste(stamp_pil, (int(x), int(y)), stamp_pil)
        return cv2.cvtColor(np.array(bg_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

editor_engine = EditorService()