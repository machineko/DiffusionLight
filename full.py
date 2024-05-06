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
import os
from tqdm import tqdm
CACHE_DIR = "./temp_inpaint_iterative"
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
MAX_NEGATIVE_EV = -5
STRENGHT = 0.8
mask_generator = MaskGenerator()

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

def get_ideal_normal_ball(size, flip_x=True):
    """
    Generate normal ball for specific size 
    Normal map is x "left", y up, z into the screen    
    (we flip X to match sobel operator)
    @params
        - size (int) - single value of height and width
    @return:
        - normal_map (np.array) - normal map [size, size, 3]
        - mask (np.array) - mask that make a valid normal map [size,size]
    """
    # we flip x to match sobel operator
    x = torch.linspace(1, -1, size)
    y = torch.linspace(1, -1, size)
    x = x.flip(dims=(-1,)) if not flip_x else x

    y, x = torch.meshgrid(y, x)
    z = (1 - x**2 - y**2)
    mask = z >= 0

    # clean up invalid value outsize the mask
    x = x * mask
    y = y * mask
    z = z * mask
    
    # get real z value
    z = torch.sqrt(z)
    
    # clean up normal map value outside mask 
    normal_map = torch.cat([x[..., None], y[..., None], z[..., None]], dim=-1)
    normal_map = normal_map.numpy()
    mask = mask.numpy()
    return normal_map, mask

def get_ball_location(image_data):
    if 'boundary' in image_data:
        # support predefined boundary if need
        x = image_data["boundary"]["x"]
        y = image_data["boundary"]["y"]
        r = image_data["boundary"]["size"]
        
        # support ball dilation
        half_dilate = BALL_DILATE // 2

        # check if not left out-of-bound
        if x - half_dilate < 0: x += half_dilate
        if y - half_dilate < 0: y += half_dilate

        # check if not right out-of-bound
        if x + r + half_dilate > IMG_W: x -= half_dilate
        if y + r + half_dilate > IMG_H: y -= half_dilate   
            
    else:
        # we use top-left corner notation
        x, y, r = ((IMG_W // 2) - (BALL_SIZE // 2), (IMG_H // 2) - (BALL_SIZE // 2), BALL_SIZE)
    return x, y, r

def interpolate_embedding(pipe):
    print("interpolate embedding...")

    # get list of all EVs
    ev_list = [float(x) for x in EV.split(",")]
    interpolants = [ev / MAX_NEGATIVE_EV for ev in ev_list]

    print("EV : ", ev_list)
    print("EV : ", interpolants)

    # calculate prompt embeddings
    prompt_normal = PROMPT
    prompt_dark = PROMPT_DARK
    prompt_embeds_normal, _, pooled_prompt_embeds_normal, _ = pipe.pipeline.encode_prompt(prompt_normal)
    prompt_embeds_dark, _, pooled_prompt_embeds_dark, _ = pipe.pipeline.encode_prompt(prompt_dark)

    # interpolate embeddings
    interpolate_embeds = []
    for t in interpolants:
        int_prompt_embeds = prompt_embeds_normal + t * (prompt_embeds_dark - prompt_embeds_normal)
        int_pooled_prompt_embeds = pooled_prompt_embeds_normal + t * (pooled_prompt_embeds_dark - pooled_prompt_embeds_normal)

        interpolate_embeds.append((int_prompt_embeds, int_pooled_prompt_embeds))

    return dict(zip(ev_list, interpolate_embeds))




def endpoint(image_data: dict, denoising_step: int, num_iter: int = 2, ball_per_iteration: int = 30, algorithm: str = "normal",):
    embedding_dict = interpolate_embedding(pipe)
    normal_ball, mask_ball = get_ideal_normal_ball(size=BALL_SIZE+BALL_DILATE)


    for ev, (prompt_embeds, pooled_prompt_embeds) in embedding_dict.items():
            x, y, r = get_ball_location(image_data)
            input_image = image_data["img"]
            img_name = image_data["name"]
            mask = mask_generator.generate_single(
                input_image, mask_ball, 
                x - (BALL_DILATE // 2),
                y - (BALL_DILATE // 2),
                r + BALL_DILATE
            )
            generator = torch.Generator().manual_seed(seed=SEED)
            kwargs = {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                'negative_prompt': NEGATIVE_PROMPT,
                'num_inference_steps': denoising_step,
                'generator': generator,
                'image': input_image,
                'mask_image': mask,
                'strength': 1.0,
                'current_seed': SEED, # we still need seed in the pipeline!
                'controlnet_conditioning_scale': CONTROL_SCALE,
                'height': IMG_H,
                'width': IMG_W,
                'normal_ball': normal_ball,
                'mask_ball': mask_ball,
                'x': x,
                'y': y,
                'r': r,
                'guidance_scale': GUIDANCE_SCALE,
            }
            cache_name = f"{img_name}_seed{SEED}"

            if algorithm == "normal":
                output_image = pipe.inpaint(**kwargs).images[0]
            elif algorithm == "iterative":
                # This is still buggy
                print("using inpainting iterative, this is going to take a while...")
                kwargs.update({
                    "strength": STRENGHT,
                    "num_iteration": num_iter,
                    "ball_per_iteration": ball_per_iteration,
                    "agg_mode": "median",
                    "save_intermediate": True,
                    "cache_dir": os.path.join(CACHE_DIR, cache_name),
                })
                output_image = pipe.inpaint_iterative(**kwargs)
            else:
                raise NotImplementedError(f"Unknown algorithm {algorithm}")