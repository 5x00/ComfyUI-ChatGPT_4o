import PIL
import numpy as np
import os
import base64
from openai import OpenAI
import PIL
import torch
from io import BytesIO
import anthropic

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def gen_openai(API_Key, Image, Prompt):
    # Initialize OpenAI client
    client = OpenAI(api_key=API_Key)
    # ChatGPT completions
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": Prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{convert_image(Image)}"}}
                ],
            }
        ],
    )
    return response.choices[0].message.content

def gen_claude(API_Key, Image, Prompt):
    #Initialize Claude client
    client = anthropic.Anthropic(api_key=API_Key)

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": convert_image(Image),
                        },
                    },
                    {"type": "text", "text": Prompt}
                ],
            }
        ],
    )

    return message.content

def convert_image(Image):
    # Convert and resize image
    pil_image = tensor_to_image(Image)
    max_dimension = 512
    pil_image.thumbnail((max_dimension, max_dimension))
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0) 
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()
    return base64_image