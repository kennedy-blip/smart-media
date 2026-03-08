import io, json, os, sys, cv2, numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Utility to convert upload to OpenCV format
def bytes_to_cv2(contents):
    nparr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Pathing for Render
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ui_path = os.path.join(base_dir, "ui")

@app.get("/")
async def serve_ui():
    return FileResponse(os.path.join(ui_path, "index.html"))

@app.post("/process-final")
async def process_final(
    file: UploadFile = File(...),
    modified_blocks: str = Form(...)
):
    try:
        blocks = json.loads(modified_blocks)
        contents = await file.read()
        img = bytes_to_cv2(contents)

        # LIGHTWEIGHT INPAINTING (Using OpenCV Telea Algorithm)
        # This removes the text based on the coordinates sent from the browser
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for b in blocks:
            # Expand coordinates slightly to ensure clean removal
            x, y, w, h = int(b['x']), int(b['y']), int(b['w']), int(b['h'])
            cv2.rectangle(mask, (x-2, y-2), (x+w+2, y+h+2), 255, -1)
        
        # Inpaint the regions
        cleaned_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

        # Optional: Add a subtle 'scanned' filter to make it look authentic
        # (This is great for your Data Science demo)
        cleaned_img = cv2.detailEnhance(cleaned_img, sigma_s=10, sigma_r=0.15)

        _, encoded_img = cv2.imencode('.jpg', cleaned_img)
        return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)