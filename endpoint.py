from fastapi import Body, FastAPI, Form, UploadFile, File, HTTPException, WebSocket, Request

from typing import Optional
import threading

from fastapi.responses import JSONResponse
from full import endpoint, process_image, exposure2hdr, cropped_ball
from PIL import Image
from io import BytesIO
import uvicorn
import ezexr
import base64
import skimage
import numpy as np
import os
import tempfile
import torch
import os
from fastapi.exceptions import RequestValidationError
import json
app = FastAPI()
PORT = os.environ.get('HDR_PORT', 8001)
upload_lock = threading.Lock()
from pydantic import BaseModel, ValidationError

class DiffusionData(BaseModel):
    denoising_step: int
    num_iters: int
    ball_per_iteration: int
    image_name: str
    algorithm: str

@app.post("/predict/")
async def upload_image(
    data: str = Form(...),
    image: UploadFile = File(...),
):
    if upload_lock.acquire(blocking=False):
        data = DiffusionData(**json.loads(data))
        print(data)
        try:
            contents = await image.read()
            filename = data.image_name if data.image_name else image.filename
            image_data = BytesIO(contents)
            img = Image.open(image_data).resize((1024, 1024), Image.BICUBIC).convert("RGB")
            img_data = {
                "img": img,
                "name": filename,
            }
            imgs, squares = endpoint(
                img_data,
                denoising_step = data.denoising_step,
                num_iter = data.num_iters,
                ball_per_iteration = data.ball_per_iteration,
                algorithm = data.algorithm
            )
            env_map_defaults = [process_image(sq) for sq in squares]
            hdr = exposure2hdr(env_map_defaults)
            balls = [cropped_ball(sq) for sq in squares]
            hdr_ball = exposure2hdr([(b.astype(np.float32)[..., :3] / 255.0) for b in balls])
            hdr_buffered = BytesIO()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".exr") as tmp:
                ezexr.imwrite(tmp.name, hdr)
                tmp.seek(0)
                hdr_buffered.write(tmp.read())
            os.unlink(tmp.name)
            hdr_base64 = base64.b64encode(hdr_buffered.getvalue()).decode()

            hdr_ball_buffer = BytesIO()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".exr") as tmp:
                ezexr.imwrite(tmp.name, hdr_ball)
                tmp.seek(0)
                hdr_ball_buffer.write(tmp.read())
            os.unlink(tmp.name)
            hdr_ball_base64 = base64.b64encode(hdr_ball_buffer.getvalue()).decode()
            return_dict = {
                f"hdr_ball": hdr_ball_base64,
                f"hdr": hdr_base64,
            }
            for ev_value, img, square, env_map_default, ball in zip(
                ["00", "25", "50"],
                imgs,
                squares,
                env_map_defaults,
                balls,
            ):
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()

                buffered = BytesIO()
                square.save(buffered, format="PNG")
                square_base64 = base64.b64encode(buffered.getvalue()).decode()

                buffered = BytesIO()
                skimage.io.imsave(buffered, skimage.img_as_ubyte(ball), format="PNG")
                ball_base64 = base64.b64encode(buffered.getvalue()).decode()

                buffered = BytesIO()
                skimage.io.imsave(
                    buffered, skimage.img_as_ubyte(env_map_default), format="png"
                )
                env_map_default_base64 = base64.b64encode(buffered.getvalue()).decode()

               
                return_dict.update(
                    {
                        f"img_{ev_value}": img_base64,
                        f"square_{ev_value}": square_base64,
                        f"env_map_default_{ev_value}": env_map_default_base64,
                        f"ball_{ev_value}": ball_base64,
                    }
                )
            return return_dict
        finally:
            upload_lock.release()
    else:
        raise HTTPException(
            status_code=429,
            detail="Another prediction is in progress. Please try again later.",
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(PORT),  timeout_keep_alive=6000)
