from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
import numpy as np
import cupy as cp

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # or ["*"] for all origins (not recommended for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def read_root():
    return {"message": "CUDA Image Filter Backend is running."}


@app.post("/upload")
async def upload_image(
    image: UploadFile = File(...),
    filter_type: str = Form("grayscale")
):
    # Read image into memory
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    np_img = np.array(img)

    # Move to GPU and convert to grayscale
    cp_img = cp.asarray(np_img)
    gray_cp = (0.299 * cp_img[:,:,0] + 0.587 * cp_img[:,:,1] + 0.114 * cp_img[:,:,2]).astype(cp.uint8)

    if filter_type == "bw":
        # Apply threshold for black and white
        bw_cp = cp.where(gray_cp > 128, 255, 0).astype(cp.uint8)
        result_np = cp.asnumpy(bw_cp)
        result_img = Image.fromarray(result_np, mode='L')
    else:
        # Grayscale
        result_np = cp.asnumpy(gray_cp)
        result_img = Image.fromarray(result_np, mode='L')

    buf = io.BytesIO()
    result_img.save(buf, format='PNG')
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png") 