import torch
torch.set_grad_enabled(False)

import random
from relighting.inpainter import BallInpainter
from relighting.mask_utils import MaskGenerator
from relighting.ball_processor import (
    get_ideal_normal_ball,
)
from relighting.argument import (
    SD_MODELS, 
    CONTROLNET_MODELS,
)
import os
import skimage
import numpy as np
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
DEVICE = "cuda"
MAX_NEGATIVE_EV = -5
STRENGHT = 0.8
SCALE = 4
GAMMA = 2.4
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
# start_time = time.time()
# pipe.pipeline.unet = torch.compile(pipe.pipeline.unet, mode="reduce-overhead", fullgraph=True)
# print("Model compilation time: ", time.time() - start_time)

class TonemapHDR(object):
    """
        Tonemap HDR image globally. First, we find alpha that maps the (max(numpy_img) * percentile) to max_mapping.
        Then, we calculate I_out = alpha * I_in ^ (1/gamma)
        input : nd.array batch of images : [H, W, C]
        output : nd.array batch of images : [H, W, C]
    """

    def __init__(self, gamma=2.4, percentile=50, max_mapping=0.5):
        self.gamma = gamma
        self.percentile = percentile
        self.max_mapping = max_mapping  # the value to which alpha will map the (max(numpy_img) * percentile) to

    def __call__(self, numpy_img, clip=True, alpha=None, gamma=True):
        if gamma:
            power_numpy_img = np.power(numpy_img, 1 / self.gamma)
        else:
            power_numpy_img = numpy_img
        non_zero = power_numpy_img > 0
        if non_zero.any():
            r_percentile = np.percentile(power_numpy_img[non_zero], self.percentile)
        else:
            r_percentile = np.percentile(power_numpy_img, self.percentile)
        if alpha is None:
            alpha = self.max_mapping / (r_percentile + 1e-10)
        tonemapped_img = np.multiply(alpha, power_numpy_img)

        if clip:
            tonemapped_img_clip = np.clip(tonemapped_img, 0, 1)

        return tonemapped_img_clip.astype('float32'), alpha, tonemapped_img
    
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

def create_envmap_grid(size: int):
    """
    BLENDER CONVENSION
    Create the grid of environment map that contain the position in sperical coordinate
    Top left is (0,0) and bottom right is (pi/2, 2pi)
    """    
    
    theta = torch.linspace(0, np.pi * 2, size * 2)
    phi = torch.linspace(0, np.pi, size)
    
    #use indexing 'xy' torch match vision's homework 3
    theta, phi = torch.meshgrid(theta, phi ,indexing='xy') 
    
    
    theta_phi = torch.cat([theta[..., None], phi[..., None]], dim=-1)
    theta_phi = theta_phi.numpy()
    return theta_phi

def get_normal_vector(incoming_vector: np.ndarray, reflect_vector: np.ndarray):
    """
    BLENDER CONVENSION
    incoming_vector: the vector from the point to the camera
    reflect_vector: the vector from the point to the light source
    """
    #N = 2(R ⋅ I)R - I
    N = (incoming_vector + reflect_vector) / np.linalg.norm(incoming_vector + reflect_vector, axis=-1, keepdims=True)
    return N

def get_cartesian_from_spherical(theta: np.array, phi: np.array, r = 1.0):
    """
    BLENDER CONVENSION
    theta: vertical angle
    phi: horizontal angle
    r: radius
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.concatenate([x[...,None],y[...,None],z[...,None]], axis=-1)

def process_image(ball_img):
    I = np.array([1,0, 0])
    env_grid = create_envmap_grid(BALL_SIZE * SCALE)   
    reflect_vec = get_cartesian_from_spherical(env_grid[...,1], env_grid[...,0])
    normal = get_normal_vector(I[None,None], reflect_vec)
    
    pos = (normal + 1.0) / 2
    pos  = 1.0 - pos
    pos = pos[...,1:]
    
    env_map = None
    ball_img = np.array(ball_img)
    # using pytorch method for bilinear interpolation
    with torch.no_grad():
        # convert position to pytorch grid look up
        grid = torch.from_numpy(pos).unsqueeze(0).float()
        grid = grid * 2 - 1 # convert to range [-1,1]

        # convert ball to support pytorch
        ball_image = torch.from_numpy(ball_img).unsqueeze(0).float() / 255.0
        ball_image = ball_image.permute(0,3,1,2) # [1,3,H,W]
        
        env_map = torch.nn.functional.grid_sample(ball_image, grid, mode='bilinear', padding_mode='border', align_corners=True)
        env_map = env_map[0].permute(1,2,0).numpy()
                
    env_map_default = skimage.transform.resize(env_map, (BALL_SIZE, BALL_SIZE*2), anti_aliasing=True)
    return env_map_default

def exposure2hdr(image):
    scaler = np.array([0.212671, 0.715160, 0.072169])
    ev = [float(i) for i in EV.split(",")]
    evs = [e for e in sorted(ev, reverse = True)]
    image0_linear = np.power(image, GAMMA)

    luminances = []
    for i in range(len(evs)):
        linear_img = np.power(image, GAMMA)
        
        linear_img *= 1 / (2 ** evs[i])
        
        lumi = linear_img @ scaler
        luminances.append(lumi)
        
    out_luminace = luminances[len(evs) - 1]
    for i in range(len(evs) - 1, 0, -1):
        maxval = 1 / (2 ** evs[i-1])
        p1 = np.clip((luminances[i-1] - 0.9 * maxval) / (0.1 * maxval), 0, 1)
        p2 = out_luminace > luminances[i-1]
        mask = (p1 * p2).astype(np.float32)
        out_luminace = luminances[i-1] * (1-mask) + out_luminace * mask
        
    hdr_rgb = image0_linear * (out_luminace / (luminances[0] + 1e-10))[:, :, np.newaxis] 
    return hdr_rgb


def endpoint(image_data: dict, denoising_step: int = 10, num_iter: int = 2, ball_per_iteration: int = 30, algorithm: str = "normal"):
    embedding_dict = interpolate_embedding(pipe)
    normal_ball, mask_ball = get_ideal_normal_ball(size=BALL_SIZE+BALL_DILATE)
    output_images, square_images = [], []
    print(embedding_dict.items())

    for ev, (prompt_embeds, pooled_prompt_embeds) in embedding_dict.items():
        print(ev)
        x, y, r = get_ball_location(image_data)
        input_image = image_data["img"]
        img_name = image_data["name"]
        mask = mask_generator.generate_single(
            input_image, mask_ball, 
            x - (BALL_DILATE // 2),
            y - (BALL_DILATE // 2),
            r + BALL_DILATE
        )
        generator = torch.Generator().manual_seed(SEED)
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
        square_image = output_image.crop((x, y, x+r, y+r))
        output_images.append(output_image)
        square_images.append(square_image)
    return output_images, square_images
    
def get_circle_mask():
    x = torch.linspace(-1, 1, BALL_SIZE)
    y = torch.linspace(1, -1, BALL_SIZE)
    y, x = torch.meshgrid(y, x)
    z = (1 - x**2 - y**2)
    mask = z >= 0
    return mask

def cropped_ball(square):
    mask = get_circle_mask().numpy()
    square = np.array(square)
    square[mask == 0] = 0
    square = np.concatenate([square, (mask*255)[...,None]], axis=2)
    square = square.astype(np.uint8)
    return square

# if __name__ == "__main__":
#     img = Image.open("input/bed.png").resize(size = (1024, 1024), resample = Image.BICUBIC)
#     img_data = {
#         "img": img,
#         "name": "bed.png",
#     }
#     img, square = endpoint(img_data)
#     env_map_default = process_image(square)
#     hdr = exposure2hdr(env_map_default)
