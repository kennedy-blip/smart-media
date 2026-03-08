import io
import json
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.services.ocr import ocr_engine
from app.services.inpaint import inpaint_engine
from app.services.editor import editor_engine
from app.utils.helpers import bytes_to_cv2

app = FastAPI(title="SmartCanvas AI Forgery Prevention")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    contents = await file.read()
    img_cv2 = bytes_to_cv2(contents)
    text_blocks = ocr_engine.scan_image(img_cv2)
    return {"text_blocks": text_blocks}

@app.post("/process-final")
async def process_final(
    file: UploadFile = File(...),
    stamp_file: UploadFile = File(None),
    modified_blocks: str = Form(...),
    font_family: str = Form("Roboto-Regular.ttf"),
    use_auto_color: str = Form("true"),
    manual_color: str = Form("#000000"),
    stamp_coords: str = Form(None),
    apply_scan_filter: str = Form("true") # Allows optional blur from frontend
):
    try:
        blocks = json.loads(modified_blocks)
        contents = await file.read()
        img_cv2 = bytes_to_cv2(contents)

        # 1. Clear original text (HEAL Colored and Plain backgrounds)
        clean_bg, _ = inpaint_engine.heal_text_areas(img_cv2, blocks)

        # 2. Add new text (VISIBLE Ink Logic for Colored Bars)
        baked_img = editor_engine.overlay_text(
            clean_bg, 
            blocks, 
            font_choice=font_family,
            use_auto_color=(use_auto_color.lower() == "true"),
            manual_color=manual_color
        )

        # 3. Handle Stamp/Signature if present
        if stamp_file and stamp_coords:
            coords = json.loads(stamp_coords)
            stamp_bytes = await stamp_file.read()
            nparr = np.frombuffer(stamp_bytes, np.uint8)
            stamp_cv2 = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if stamp_cv2 is not None:
                baked_img = editor_engine.overlay_stamp(
                    baked_img,
                    stamp_cv2,
                    x=coords['x'],
                    y=coords['y'],
                    width=coords['w'],
                    height=coords['h']
    
                )

        # 4. FINAL TOUCH: SCANNER DEGRADATION (Blur + Noise)
        if apply_scan_filter.lower() == "true":
            # Apply fuzzy scanned texture for authentic lookalike
            final_img = editor_engine.apply_scan_degradation(baked_img)
        else:
            # Leave pristine/digital if filter is disabled
            final_img = baked_img

        # 5. Conversion and Response
        _, encoded_img = cv2.imencode('.jpg', final_img)
        return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))