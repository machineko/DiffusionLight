from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Optional
import threading
from full import endpoint, process_image, exposure2hdr
from PIL import Image
from io import BytesIO
import uvicorn
import ezexr
import base64
import skimage
import numpy as np
import os
import tempfile

app = FastAPI()

# Create a global lock
upload_lock = threading.Lock()

@app.post("/predict/")
async def upload_image(image: UploadFile = File(...), image_name: Optional[str] = None):
    if upload_lock.acquire(blocking=False):
        try:
            contents = await image.read()
            filename = image_name if image_name else image.filename
            image_data = BytesIO(contents)
            img = Image.open(image_data).resize((1024,1024), Image.BICUBIC)
            img_data = {
                "img": img,
                "name": filename,
            }
            img, square = endpoint(img_data)
            env_map_default = process_image(square)
            hdr = exposure2hdr(env_map_default)

            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            buffered = BytesIO()
            square.save(buffered, format="PNG")
            square_base64 = base64.b64encode(buffered.getvalue()).decode()

            buffered = BytesIO()
            skimage.io.imsave(buffered, skimage.img_as_ubyte(env_map_default), format="png")

            # env_map_default.save(buffered, format="PNG")
            env_map_default_base64 = base64.b64encode(buffered.getvalue()).decode()

            hdr_buffered = BytesIO()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".exr") as tmp:
                ezexr.imwrite(tmp.name, hdr)
                tmp.seek(0)
                hdr_buffered.write(tmp.read())
            os.unlink(tmp.name)  # delete the temp file
            hdr_base64 = base64.b64encode(hdr_buffered.getvalue()).decode()

            return {
                "img": img_base64,
                "square": square_base64,
                "env_map_default": env_map_default_base64,
                "hdr": hdr_base64
            }
        finally:
            upload_lock.release()
    else:
        raise HTTPException(status_code=429, detail="Another prediction is in progress. Please try again later.")
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)