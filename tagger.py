from io import BytesIO
import os
import base64
from openai import OpenAI
from PIL import Image, ImageOps
from torchvision.transforms import ToPILImage

class vlmnode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "image_in" : ("IMAGE", {}) },
            "required": { "prompt_in" : ("STRING", {}) },
            "required": { "api" : ("STRING", {}) },
        }

    RETURN_TYPES = ("STRING")
    RETURN_NAMES = ("text_out",)
    CATEGORY = "5x00/GPT4o"
    FUNCTION = "create_caption"

    def create_caption(self, api, image_in, prompt_in):

        # Initialize OpenAI client
        client = OpenAI(api_key=api)

        # Convert and resize image
        transform = ToPILImage()
        pil_image = transform(image_in.squeeze(0))
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
                        {"type": "text", "text": prompt_in},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ],
                }
            ],
        )
        caption = response.choices[0].message.content 
        return caption