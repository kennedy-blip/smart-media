import io, json, os, cv2, numpy as np, random
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fpdf import FPDF

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def apply_scan_filter(img):
    """Adds subtle noise and lighting gradients to mimic a physical scanner."""
    # 1. Subtle Gaussian Blur (Ink Bleed)
    img = cv2.GaussianBlur(img, (1, 1), 0)
    # 2. Add Stochastic Noise (Paper Grain)
    noise = np.zeros(img.shape, np.uint8)
    cv2.randn(noise, 128, 20)
    img = cv2.addWeighted(img, 0.98, noise.astype(img.dtype), 0.02, 0)
    # 3. Brightness Gradient
    rows, cols = img.shape[:2]
    brightness_map = np.linspace(1.0, 0.97, rows).reshape(rows, 1)
    img = (img * brightness_map[:, :, np.newaxis]).astype(np.uint8)
    return img

def get_ink_color_safe(img, x, y, w, h):
    """OTSU binarization to sample ink while ignoring paper background."""
    roi = img[max(0, y):y+h, max(0, x):x+w]
    if roi.size == 0: return (45, 45, 45)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink_pixels = roi[mask == 0]
    if ink_pixels.size == 0: return (45, 45, 45)
    median = np.median(ink_pixels, axis=0)
    return (int(median[0]), int(median[1]), int(median[2]))

def run_synthesis(img, blocks, use_scan_filter=True):
    # Base Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)

    # Inpaint Removal
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for b in blocks:
        cv2.rectangle(mask, (int(b['x']), int(b['y'])), (int(b['x']+b['w']), int(b['y']+b['h'])), 255, -1)
    clean_bg = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    # Kerning & Spacing Synthesis
    for b in blocks:
        x, y, w, h = int(b['x']), int(b['y']), int(b['w']), int(b['h'])
        color = get_ink_color_safe(img, x, y, w, h)
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = (h * 0.78) / 22.0
        txt = b['text']
        char_count = len(txt) if len(txt) > 0 else 1
        step_w = w / char_count
        for i, char in enumerate(txt):
            char_x = int(x + (i * step_w) + (step_w * 0.1))
            char_y = y + int(h * 0.8)
            cv2.putText(clean_bg, char, (char_x, char_y), font, font_scale, color, 1, cv2.LINE_AA)

    if use_scan_filter:
        clean_bg = apply_scan_filter(clean_bg)
    return clean_bg

@app.post("/process-final")
async def process_final(file: UploadFile = File(...), modified_blocks: str = Form(...), scan_filter: bool = Form(True)):
    blocks = json.loads(modified_blocks)
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    final = run_synthesis(img, blocks, use_scan_filter=scan_filter)
    # Metadata Wipe via re-encoding
    _, encoded = cv2.imencode('.jpg', final, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return StreamingResponse(io.BytesIO(encoded.tobytes()), media_type="image/jpeg")

@app.post("/export-pdf")
async def export_pdf(file: UploadFile = File(...), modified_blocks: str = Form(...), scan_filter: bool = Form(True)):
    blocks = json.loads(modified_blocks)
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    final = run_synthesis(img, blocks, use_scan_filter=scan_filter)
    _, encoded = cv2.imencode('.jpg', final)
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    pdf.image(io.BytesIO(encoded.tobytes()), x=0, y=0, w=210)
    return StreamingResponse(io.BytesIO(pdf.output()), media_type="application/pdf")

@app.get("/")
async def ui(): return FileResponse(os.path.join("ui", "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))