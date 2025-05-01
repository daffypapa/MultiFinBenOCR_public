from transformers import AutoProcessor, AutoModelForVision2Seq, BlipProcessor, BlipForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import torch
from openai import OpenAI
import base64
import io
import os

class Agent:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if "llava" in model_name:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            self.model = AutoModelForVision2Seq.from_pretrained(
                "llava-hf/llava-1.5-7b-hf",
                device_map="auto",
                quantization_config=bnb_config
            )
        elif "blip" in model_name:
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        elif "gpt-4o" in model_name:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")  # Replace with your API key
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def draft(self, image_path):
        image = Image.open(image_path).convert("RGB")

        
        prompt = "Convert this financial statement page into semantically correct HTML. Return html and nothing else. Use plain html only, no styling please."

        if "llava" in self.model_name:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt"
            ).to(self.device, torch.float16 if torch.cuda.is_available() else torch.float32)

            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=1024)

            result = self.processor.tokenizer.decode(output[0], skip_special_tokens=True)
            return result

        elif "blip" in self.model_name:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=1024)
            result = self.processor.tokenizer.decode(output[0], skip_special_tokens=True)
            return result

        elif "gpt-4o" in self.model_name:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            b64_image = base64.b64encode(img_bytes).decode("utf-8")
            
            client = OpenAI(
                # This is the default and can be omitted
                api_key=self.openai_api_key,
            )
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                max_tokens=1024
            )
            return response.choices[0].message.content
