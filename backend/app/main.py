import io, json, os, sys, cv2, numpy as np, multiprocessing
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# UTILITY FUNCTIONS

def bytes_to_cv2(contents):
    """Converts uploaded bytes to an OpenCV BGR image."""
    nparr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def get_sampling_mask(shape, x, y, w, h, thickness=10):
    """Creates a 'donut' mask around the text area to sample color."""
    mask = np.zeros(shape, dtype=np.uint8)
    
    # Inner rectangle (text area - do not sample here)
    inner_coords = (max(0, x), max(0, y), min(shape[1], x + w), min(shape[0], y + h))
    cv2.rectangle(mask, (inner_coords[0], inner_coords[1]), (inner_coords[2], inner_coords[3]), 255, -1)
    
    # Outer rectangle (dilated text area)
    dilated_coords = (max(0, x - thickness), max(0, y - thickness), min(shape[1], x + w + thickness), min(shape[0], y + h + thickness))
    outer_rect = np.zeros(shape, dtype=np.uint8)
    cv2.rectangle(outer_rect, (dilated_coords[0], dilated_coords[1]), (dilated_coords[2], dilated_coords[3]), 255, -1)
    
    # Combine (Outer - Inner = Donut)
    final_mask = cv2.subtract(outer_rect, mask)
    return final_mask

def sample_average_color(img, x, y, w, h):
    """Samples the image color *around* the text bounding box."""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Create a dynamic 'donut' mask around the area
    donut_mask = get_sampling_mask(img_gray.shape, x, y, w, h, thickness=5)
    
    # 2. Extract the relevant pixels
    sample_pixels = img[donut_mask == 255]
    
    # 3. Handle edge case (no pixels)
    if sample_pixels.size == 0:
        return (50, 50, 50) # Default dark gray fallback
    
    # 4. Calculate the Median (more robust than Average for this use case)
    median_color = np.median(sample_pixels, axis=0)
    return (int(median_color[0]), int(median_color[1]), int(median_color[2]))

# FASTAPI APP SETUP

app = FastAPI(title="SmartCanvas Pro | Lookalike Engine")

# CORS middleware for open accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# SMART PATHING FOR RENDER & LOCAL WINDOWS EXE
if hasattr(sys, '_MEIPASS'):
    # Running as a compiled exe
    base_dir = sys._MEIPASS
else:
    # Running locally or on Render
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ui_path = os.path.join(base_dir, "ui")

# --- UI SERVING ---

@app.get("/")
async def serve_ui():
    """Serves the main dashboard (index.html)."""
    index_file = os.path.join(ui_path, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"status": "online", "message": "UI not found", "searching_at": ui_path}

# --- THE LOOKALIKE PIPELINE ---

@app.post("/process-final")
async def process_final(
    file: UploadFile = File(...),
    modified_blocks: str = Form(...) # The JSON from the editable form
):
    try:
        # Load the blocks and the original image
        blocks = json.loads(modified_blocks)
        contents = await file.read()
        img = bytes_to_cv2(contents)
        
        # PIPELINE PART 1: INPAINTING (Removal of old text)
        
        # Create the high-precision inpainting mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for b in blocks:
            x, y, w, h = int(b['x']), int(b['y']), int(b['w']), int(b['h'])
            # Expand slightly to clean edges
            cv2.rectangle(mask, (x-2, y-2), (x+w+2, y+h+2), 255, -1)
            
        # Perform inpainting to generate clean background (Telea Algorithm)
        # Perfect for Render Free Tier (low RAM usage)
        inpaint_res = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        
        # PIPELINE PART 2: LOOKALIKE TEXT SYNTHESIS (Add new text)
        
        synth_res = inpaint_res.copy()
        
        for b in blocks:
            x, y, w, h = int(b['x']), int(b['y']), int(b['w']), int(b['h'])
            new_text = b['text']
            
            # --- The Lookalike Heuristics ---
            
            # 1. Lookalike Position: Match the original baseline
            text_x = x
            text_y = y + h - 5 # Approximate baseline offset
            
            # 2. Lookalike Font: Simple san-serif font (standard in docs)
            font_choice = cv2.FONT_HERSHEY_SIMPLEX
            
            # 3. Lookalike Size: Scale font mathematically based on original height
            font_scale = h / 30.0 
            thickness = max(1, int(font_scale * 1.5)) # Adjust thickness relative to scale
            
            # 4. Lookalike Color: Dynamic Sampling (The Fix!)
            # Sample color from the original image BEFORE inpainting wiped it.
            sampled_color = sample_average_color(img, x, y, w, h)
            
            # 5. Lookalike Smoothing (Anti-Aliasing)
            #LINE_AA ensures text edges are not pixelated
            font_aa = cv2.LINE_AA 

            # Render the new text
            cv2.putText(synth_res, new_text, (text_x, text_y), 
                        font_choice, font_scale, sampled_color, thickness, font_aa)

        # Encode and stream the final image (JPEG)
        _, encoded_img = cv2.imencode('.jpg', synth_res)
        return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")

    except Exception as e:
        print(f"Synthesis Pipeline Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- SERVER BOOT ---

if __name__ == "__main__":
    import uvicorn
    # Multiprocessing support required for Windows exe
    multiprocessing.freeze_support()
    
    # Render passes port via an environment variable
    port = int(os.environ.get("PORT", 8000))
    # Local: Runs on 127.0.0.1:8000
    # Render: Runs on 0.0.0.0:PORT
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")