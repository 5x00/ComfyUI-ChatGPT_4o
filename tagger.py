from io import BytesIO
import os
import base64
from openai import OpenAI
from PIL import Image, ImageOps
import torch
import numpy as np

class tagger_node:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Caption",)
    FUNCTION = "create_caption"
    CATEGORY = "5x00/GPT4o"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                 "Image" : ("IMAGE", {}), 
                 "Prompt" : ("STRING", {}),
                 "API_Key" : ("STRING", {}),
            },
        }

    def create_caption(self, API_Key, Image, Prompt):

        # Initialize OpenAI client
        client = OpenAI(api_key=API_Key)

        # Convert and resize image
        Image = ImageOps.exif_transpose(Image)
        if Image.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        pil_image = i.convert("RGB")
        pil_image = np.array(pil_image).astype(np.float32) / 255.0
        pil_image = torch.from_numpy(pil_image)[None,]
        max_dimension = 512
        pil_image.thumbnail((max_dimension, max_dimension))
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")
        buffer.seek(0) 
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        buffer.close()

        # ChatGPT completions
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": Prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ],
                }
            ],
        )
        caption = response.choices[0].message.content 
        return caption
    
NODE_CLASS_MAPPINGS = {
    "Image Tagger" : tagger_node,
}