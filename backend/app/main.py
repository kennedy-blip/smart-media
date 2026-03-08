import io, json, os, cv2, numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fpdf import FPDF

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def get_ink_color(img, x, y, w, h):
    """Samples the original text area to find the median ink color."""
    roi = img[max(0, y):y+h, max(0, x):x+w]
    if roi.size == 0: return (40, 40, 40)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink_pixels = roi[mask == 0]
    if ink_pixels.size == 0: return (40, 40, 40)
    median = np.median(ink_pixels, axis=0)
    return (int(median[0]), int(median[1]), int(median[2]))

def run_synthesis(img, blocks):
    """Core logic to sharpen, inpaint, and overlay lookalike text."""
    # 1. SHARPEN to fix document blur
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)

    # 2. INPAINT (Clean old text)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for b in blocks:
        cv2.rectangle(mask, (int(b['x'])-1, int(b['y'])-1), (int(b['x']+b['w'])+1, int(b['y']+b['h'])+1), 255, -1)
    clean_bg = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    # 3. SYNTHESIS (Add New Text)
    for b in blocks:
        x, y, w, h = int(b['x']), int(b['y']), int(b['w']), int(b['h'])
        color = get_ink_color(img, x, y, w, h)
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = (h * 0.8) / 22.0
        (tw, th), bl = cv2.getTextSize(b['text'], font, font_scale, 1)
        tx = x + (w - tw) // 2
        ty = y + (h + th) // 2 - 2
        cv2.putText(clean_bg, b['text'], (tx, ty), font, font_scale, color, 1, cv2.LINE_AA)
    
    return cv2.GaussianBlur(clean_bg, (1, 1), 0)

@app.post("/process-final")
async def process_final(file: UploadFile = File(...), modified_blocks: str = Form(...)):
    try:
        blocks = json.loads(modified_blocks)
        img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
        final_img = run_synthesis(img, blocks)
        _, encoded = cv2.imencode('.jpg', final_img)
        return StreamingResponse(io.BytesIO(encoded.tobytes()), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export-pdf")
async def export_pdf(file: UploadFile = File(...), modified_blocks: str = Form(...)):
    try:
        blocks = json.loads(modified_blocks)
        img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
        final_img = run_synthesis(img, blocks)
        _, encoded = cv2.imencode('.jpg', final_img)
        
        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.add_page()
        pdf.image(io.BytesIO(encoded.tobytes()), x=0, y=0, w=210)
        
        pdf_stream = io.BytesIO(pdf.output())
        return StreamingResponse(pdf_stream, media_type="application/pdf", 
                                 headers={"Content-Disposition": "attachment; filename=document.pdf"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def ui():
    # Serves the index.html from a /ui folder
    return FileResponse(os.path.join("ui", "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))