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

seed = 2

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16)
pipeline.to("cuda")
generator = torch.Generator(device="cuda").manual_seed(seed)

api_key = os.environ.get("ANTHROPIC_API_KEY")

def draw_image(prompt: str):
    images = pipeline(prompt, generator=generator, guidance_scale=7.5)
    return images[0][0]


claude_client = anthropic.Anthropic(api_key=api_key)
image_media_type = "image/png"

timestamp = time.strftime("%Y%m%d%H%M%S")


if __name__ == "__main__":

    image_order = "白黒漫画スタイルの画像。右手で鞄を持っている男子高校生。サッカーをしている。公園にいる。"

    message = claude_client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please create prompt of Stable Diffusion. For following image description."
                            + "Our model accepts 77 tokens at most."
                            + "Please use English for the prompt."
                            + "Please tell me only the improved prompt. DO NOT include other information."
                    },
                    {
                        "type": "text",
                        "text": f"Description: {image_order}"
                    },
                    {
                        "type": "text",
                        "text": "Prompt:"
                    }
                ],
            }
        ],
    )
    prompt = message.content[0].text
    print("============= PROMPT START ==============")
    print(prompt)
    print("============= PROMPT END ==============")

    image = draw_image(prompt)
    image.save(f"out/{timestamp}_{seed}_first.png")


    for i in range(1, 4):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        image_data = base64.b64encode(img_byte_arr.getvalue()).decode('ascii')

        message = claude_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image_media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Please improve prompt of Stable Diffusion to generate a mode suitable image."
                                + "Our model accepts 77 tokens at most."
                                + "Please use English for the prompt."
                                + "Attached image is the result of the previous prompt."
                                + "Please tell me only the improved prompt. DO NOT include other information."
                        },
                        {
                            "type": "text",
                            "text": f"Description: {image_order}"
                        },
                        {
                            "type": "text",
                            "text": f"Previous prompt: {image_order}"
                        },
                        {
                            "type": "text",
                            "text": "Improved prompt:"
                        }
                    ],
                }
            ],
        )
        prompt = message.content[0].text
        print("============= PROMPT START ==============")
        print(prompt)
        print("============= PROMPT END ==============")

        image = draw_image(prompt)
        image.save(f"out/{timestamp}_{seed}_{i}.png")