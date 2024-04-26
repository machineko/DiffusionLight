import torch
import random
from relighting.inpainter import BallInpainter
import time
from relighting.mask_utils import MaskGenerator
from relighting.ball_processor import (
    get_ideal_normal_ball,
    crop_ball
)
from relighting.dataset import GeneralLoader
from relighting.utils import name2hash
import relighting.dist_utils as dist_util
from relighting.argument import (
    SD_MODELS, 
    CONTROLNET_MODELS,
    VAE_MODELS
)
BALL_SIZE = 256
BALL_DILATE = 20
PROMPT = "a perfect mirrored reflective chrome ball sphere"
PROMPT_DARK = "a perfect black dark mirrored reflective chrome ball sphere"
NEGATIVE_PROMPT = "matte, diffuse, flat, dull"
IMG_H, IMG_W = 1024, 1024
CONTROL_SCALE = 0.5
GUIDANCE_SCALE = 5.0
SEED = random.choice((0, 37, 71, 125, 140, 196, 307, 434, 485, 575, 9021, 9166, 9560, 9814))
EV = "0,-2.5,-5"
DEVICE = "cuda:3"

model, controlnet = SD_MODELS["sdxl"], CONTROLNET_MODELS["sdxl"]
pipe = BallInpainter.from_sdxl(
    model=model, 
    controlnet=controlnet, 
    device=DEVICE,
    torch_dtype = torch.float16,
    offload = False
)

pipe.pipeline.load_lora_weights("models/ThisIsTheFinal-lora-hdr-continuous-largeT@900/0_-5/checkpoint-2500")
pipe.pipeline.fuse_lora(lora_scale=0.75) # fuse lora weight w' = w + \alpha \Delta w
enabled_lora = True

print("compiling unet model")
start_time = time.time()
pipe.pipeline.unet = torch.compile(pipe.pipeline.unet, mode="reduce-overhead", fullgraph=True)
print("Model compilation time: ", time.time() - start_time)

def get_ball_location(image_data, args):
    if 'boundary' in image_data:
        # support predefined boundary if need
        x = image_data["boundary"]["x"]
        y = image_data["boundary"]["y"]
        r = image_data["boundary"]["size"]
        
        # support ball dilation
        half_dilate = args.ball_dilate // 2

        # check if not left out-of-bound
        if x - half_dilate < 0: x += half_dilate
        if y - half_dilate < 0: y += half_dilate

        # check if not right out-of-bound
        if x + r + half_dilate > args.img_width: x -= half_dilate
        if y + r + half_dilate > args.img_height: y -= half_dilate   
            
    else:
        # we use top-left corner notation
        x, y, r = ((args.img_width // 2) - (args.ball_size // 2), (args.img_height // 2) - (args.ball_size // 2), args.ball_size)
    return x, y, r

def interpolate_embedding(pipe, args):
    print("interpolate embedding...")

    # get list of all EVs
    ev_list = [float(x) for x in args.ev.split(",")]
    interpolants = [ev / args.max_negative_ev for ev in ev_list]

    print("EV : ", ev_list)
    print("EV : ", interpolants)

    # calculate prompt embeddings
    prompt_normal = args.prompt
    prompt_dark = args.prompt_dark
    prompt_embeds_normal, _, pooled_prompt_embeds_normal, _ = pipe.pipeline.encode_prompt(prompt_normal)
    prompt_embeds_dark, _, pooled_prompt_embeds_dark, _ = pipe.pipeline.encode_prompt(prompt_dark)

    # interpolate embeddings
    interpolate_embeds = []
    for t in interpolants:
        int_prompt_embeds = prompt_embeds_normal + t * (prompt_embeds_dark - prompt_embeds_normal)
        int_pooled_prompt_embeds = pooled_prompt_embeds_normal + t * (pooled_prompt_embeds_dark - pooled_prompt_embeds_normal)

        interpolate_embeds.append((int_prompt_embeds, int_pooled_prompt_embeds))

    return dict(zip(ev_list, interpolate_embeds))