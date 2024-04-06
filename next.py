from diffusers import DiffusionPipeline
import torch
import os, io, time
import anthropic
import base64
from dotenv import load_dotenv
load_dotenv()

if torch.cuda.is_available():
    print("CUDA is available")
    print("CUDA VERSION: " + torch.version.cuda)  
else:
    print("CUDA is not available")
    exit()


def draw_image(prompt: str, negative_prompt: str, seed=0):
    pipeline = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16)
    pipeline.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    images = pipeline(prompt, generator=generator, guidance_scale=7.5, negative_prompt=negative_prompt, width=512, height=960)
    return images[0][0]



if __name__ == "__main__":

    prompt = '''
    - Gender: One Cute girl
BREAK
- Age: 16 years old
BREAK
- Height: 156cm, petite
BREAK
- Atmosphere: Intellectual
BREAK
- Style: Anime-style
BREAK
- Personality: Cool, intelligent, with a cute vibe
BREAK
- Face: Big, sparkling eyes
BREAK
- Expression: Smiling
BREAK
- Eyes: Round
BREAK
- Facial decoration: Black-rimmed square glasses, large size
BREAK
- (Outfit (Top): Navy blazer-type School uniform with red tie:1.3),
BREAK
-  (Outfit (Bottom): Gray knee-length skirt,Gray color:1.3),
BREAK
- Shoes: Black loafers
BREAK
- Socks: Knee-high black socks
BREAK
- (Hairstyle:Very Long, straight black hair:1.3), (bangs cut straight above the eyebrows),
BREAK
- Haircolor:Black:1.4,
BREAK
- Hair set: Not tied up,BREAK
BREAK
- Body type: Petite, slim
BREAK
- Expression: Cute smile,
BREAK
- Gaze: Looking directly at the camera,
BREAK
- Pose: Standing in upright posture and Both arms raised above the shoulder,
- Body: Erect spine, shoulders back, arms at sides
- Hands: Slightly open, hanging naturally by the sides
- Legs: Feet close together, legs straight
BREAK
- Composition: Full body
BREAK
- (Background: None:1.4),
BREAK
- Personal items: None,

masterpiece,best quality,ultra-detailed,
    '''
    np = '''
lowres, bad anatomy, text, bad face, error, extra digit, fewer digits, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, {blurry}, missing arms, missing legs, more than two legs,nsfw,''' 

    # timestamp as seed
    for i in range(5):
        seed = int(time.time())
        timestamp = time.strftime("%Y%m%d%H%M%S")
        image = draw_image(prompt, np, seed)
        image.save(f"out/{timestamp}.png")
