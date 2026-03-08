import io
import json
import cv2
import numpy as np
import os
import sys
import multiprocessing
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Internal service imports (Ensure these folders exist in your 'app' directory)
from app.services.ocr import ocr_engine
from app.services.inpaint import inpaint_engine
from app.services.editor import editor_engine
from app.utils.helpers import bytes_to_cv2

app = FastAPI(title="SmartCanvas Pro Engine")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SMART PATHING FOR STANDALONE EXE ---
if hasattr(sys, '_MEIPASS'):
    # Path when running as a compiled .exe (PyInstaller temporary folder)
    base_dir = sys._MEIPASS
else:
    # Path when running locally (Go up one level from 'app' to 'backend' root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define key paths
ui_path = os.path.join(base_dir, "ui")
fonts_path = os.path.join(base_dir, "app", "fonts")

# --- UI SERVING ---
@app.get("/")
async def serve_ui():
    """Serves the single-file UI from the ui folder."""
    index_file = os.path.join(ui_path, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    
    # Fallback if UI folder is missing
    return {
        "status": "online",
        "engine": "SmartCanvas Pro Neural Engine",
        "error": "UI folder/index.html not found",
        "searched_at": ui_path
    }

# --- API ENDPOINTS ---

@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    """Handles OCR text detection."""
    try:
        contents = await file.read()
        img_cv2 = bytes_to_cv2(contents)
        text_blocks = ocr_engine.scan_image(img_cv2)
        return {"text_blocks": text_blocks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scan Error: {str(e)}")

@app.post("/process-final")
async def process_final(
    file: UploadFile = File(...),
    stamp_file: UploadFile = File(None),
    modified_blocks: str = Form(...),
    font_family: str = Form("Roboto-Regular.ttf"),
    use_auto_color: str = Form("true"),
    manual_color: str = Form("#000000"),
    stamp_coords: str = Form(None),
    apply_scan_filter: str = Form("true")
):
    """Handles Inpainting, Text Overlay, and Final Image Generation."""
    try:
        blocks = json.loads(modified_blocks)
        is_auto_color = use_auto_color.lower() == "true"
        is_scan_filter = apply_scan_filter.lower() == "true"
        
        contents = await file.read()
        img_cv2 = bytes_to_cv2(contents)

        # 1. Remove old text (Inpainting)
        clean_bg, _ = inpaint_engine.heal_text_areas(img_cv2, blocks)

        # 2. Add new text
        baked_img = editor_engine.overlay_text(
            clean_bg, 
            blocks, 
            font_choice=font_family,
            use_auto_color=is_auto_color,
            manual_color=manual_color
        )

        # 3. Handle Optional Stamp/Signature
        if stamp_file and stamp_coords:
            s_coords = json.loads(stamp_coords)
            stamp_contents = await stamp_file.read()
            nparr = np.frombuffer(stamp_contents, np.uint8)
            stamp_cv2 = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            
            if stamp_cv2 is not None:
                baked_img = editor_engine.overlay_stamp(
                    baked_img, stamp_cv2,
                    x=s_coords['x'], y=s_coords['y'],
                    width=s_coords['w'], height=s_coords['h']
                )

        # 4. Apply 'Scanned Document' effect if requested
        final_output = editor_engine.apply_scan_degradation(baked_img) if is_scan_filter else baked_img

        # Encode to JPEG and stream back to browser
        _, encoded_img = cv2.imencode('.jpg', final_output)
        return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")

    except Exception as e:
        print(f"PIPELINE ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- SERVER BOOT ---
if __name__ == "__main__":
    import uvicorn
    import os
    # Render provides the port via an environment variable
    # If it's not found (like on your local PC), it defaults to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # Use 0.0.0.0 so it's accessible to the outside world
    uvicorn.run(app, host="0.0.0.0", port=port)