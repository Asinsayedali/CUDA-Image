from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "CUDA Image Filter Backend is running."}

@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    # For now, just confirm receipt and return filename
    return JSONResponse(content={"filename": image.filename, "content_type": image.content_type, "message": "Image received."}) 