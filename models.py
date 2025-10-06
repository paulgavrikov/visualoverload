import mimetypes
from pathlib import Path
import os
import logging
import json
import time
from collections import defaultdict
import base64
from multiprocessing import Pool
import abc
import torch
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import random
from functools import partial
import math


REGISTERED_MODELS = {}

MODEL_PARAM_SIZES = {
    "paligemma2-3b-mix-448": 3,
    "paligemma2-10b-mix-448": 10,
    "paligemma2-28b-mix-448": 28,
    "llava-1.5-7b-hf": 7,
    "llava-v1.6-vicuna-7b-hf": 7,
    "llava-onevision-qwen2-7b-ov-hf": 7,
    "llava-onevision-qwen2-72b-ov-hf": 72,
    "llava-1.5-13b-hf": 13,
    "llava-v1.6-vicuna-13b-hf": 13,
    "smolvlm": 2,
    "gemma-3n-e2b-it": 5,
    "gemma-3n-e4b-it": 5,
    "gemma-3-12b-it": 12,
    "gemma-3-27b-it": 27,
    "internlm-xcomposer2-4khd-7b": 7,
    "internlm-xcomposer2d5-7b": 7,
    "VILA-HD-8B-PS3-4K-SigLIP": 8,
    "VILA-HD-8B-PS3-1.5K-SigLIP": 8,
    "Qwen2.5-VL-3B-Instruct": 3,
    "Qwen2.5-VL-7B-Instruct": 7,
    "Qwen2.5-VL-32B-Instruct": 32,
    "Qwen2.5-VL-72B-Instruct": 72,
    "Llama-4-Scout-17B-16E-Instruct": 109,
    "InternVL3-1B": 1,
    "InternVL3-2B": 2,
    "InternVL3-8B": 8,
    "InternVL3-14B": 14,
    "InternVL3-38B": 38,
    "InternVL3-78B": 78,
    "gemini-2.0-flash": float("inf"),
    "gemini-2.5-flash": float("inf"),
    "o4-mini-2025-04-16": float("inf"),
    "o3-2025-04-16": float("inf"),
    "horizon-alpha": float("inf"),
    "deepseek-vl2-tiny": 1,
    "deepseek-vl2-small": 2.8,
    "deepseek-vl2": 4.5,
    "LFM2-VL-1.6B": 1.6,
    "LFM2-VL-450M": 0.45,
    "SmolVLM-Instruct": 2,
}

MODEL_ALIASES = {
    "paligemma2-3b-mix-448": "PaliGemma 2 3B",
    "paligemma2-10b-mix-448": "PaliGemma 2 10B",
    "paligemma2-28b-mix-448": "PaliGemma 2 28B",
    "llava-1.5-7b-hf": "LLaVA 1.5 7B",
    "llava-1.5-13b-hf": "LLaVA 1.5 13B",
    "llava-v1.6-vicuna-7b-hf": "LLaVA-NeXT 7B",
    "llava-v1.6-vicuna-13b-hf": "LLaVA-NeXT 13B",
    "llava-onevision-qwen2-7b-ov-hf": "LLaVA-OneVision 7B",
    "llava-onevision-qwen2-72b-ov-hf": "LLaVA-OneVision 72B",
    "gemma-3n-e2b-it": "Gemma 3n E2B",
    "gemma-3n-e4b-it": "Gemma 3n E4B",
    "gemma-3-12b-it": "Gemma 3 12B",
    "gemma-3-27b-it": "Gemma 3 27B",
    "internlm-xcomposer2-4khd-7b": "InternLM-XComposer2-4KHD",
    "internlm-xcomposer2d5-7b": "InternLM-XComposer2.5",
    "VILA-HD-8B-PS3-4K-SigLIP": "VILA HD 4K",
    "VILA-HD-8B-PS3-1.5K-SigLIP": "VILA HD 1.5K",
    "Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL 3B",
    "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL 7B",
    "Qwen2.5-VL-32B-Instruct": "Qwen2.5-VL 32B",
    "Qwen2.5-VL-72B-Instruct": "Qwen2.5-VL 72B",
    "Llama-4-Scout-17B-16E-Instruct": "Llama 4 Scout",
    "InternVL3-1B": "InternVL3 1B",
    "InternVL3-2B": "InternVL3 2B",
    "InternVL3-8B": "InternVL3 8B",
    "InternVL3-14B": "InternVL3 14B",
    "InternVL3-38B": "InternVL3 38B",
    "InternVL3-78B": "InternVL3 78B",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "o4-mini-2025-04-16": "o4 mini",
    "o3-2025-04-16": "o3",
    "horizon-alpha": "Horizon Alpha",
    "deepseek-vl2-tiny": "DeepSeek VL2 Tiny",
    "deepseek-vl2-small": "DeepSeek VL2 Small",
    "deepseek-vl2": "DeepSeek VL2",
    "LFM2-VL-1.6B": "LFM2 VL 1.6B",
    "LFM2-VL-450M": "LFM2 VL 450M",
    "SmolVLM-Instruct": "SmolVLM",
}    


def register_model(model_name: str):
    """
    Decorator to register a model class.
    This allows the model to be instantiated by its name.
    """
    def decorator(cls):
        REGISTERED_MODELS[model_name] = partial(cls, model_name=model_name)
        logging.info(f"Registered model: {model_name}")
        return cls
    return decorator


def load_model(model_name, device):
    """
    Load a model by its name.
    If the model is not registered, raise an error.
    """
    if model_name in REGISTERED_MODELS:
        return REGISTERED_MODELS[model_name](device=device)
    else:
        raise ValueError(f"Model {model_name} is not registered. Available models: {list(REGISTERED_MODELS.keys())}")


class KeyStorage:
    """
    A class to manage API keys for different models.
    It loads keys from a JSON file and provides a method to retrieve the key for a specific model.
    The keys are stored in a dictionary, where the a service name is the key and the API key is the value.

    The class also keeps track of the number of requests made for each service, allowing for round-robin key usage.
    If a service has multiple keys, it will cycle through them for each request.

    The keys are stored in a JSON file located at ~/api_keystore.json.
    """
    def __init__(self):
        self._api_request_count = defaultdict(int)
        self.key_storage = {}

        keystore_path = os.path.expanduser("~/api_keystore.json")

        if os.path.exists(keystore_path):
            self.key_storage = json.load(open(keystore_path, "r"))

            for key in self.key_storage:
                if type(self.key_storage[key]) == list:
                    random.shuffle(self.key_storage[key])

            logging.info(f"Loaded key storage with {list(self.key_storage.keys())}")
        else:
            logging.warning(f"Key storage not found")
    
    def get_key(self, service_name: str):
        """
        Retrieve the API key for a specific service.
        If the service has multiple keys, it will cycle through them for each request.
        Args:
            service_name (str): The name of the service for which to retrieve the API key.
        Returns:
            str: The API key for the specified service.
        Raises:
            ValueError: If the service name is not found in the key storage.    
        """

        if service_name in self.key_storage:
            key =  self.key_storage[service_name]
            if type(key) == list:
                request_count = self._api_request_count[service_name]
                key = key[request_count % len(key)]
                self._api_request_count[service_name] = request_count + 1
            return key
        else:
            raise ValueError(f"Model {service_name} not found in key storage")


class InferenceBase(abc.ABC):
    """
    Base class for inference models.
    This class defines the interface for all inference models and provides a fallback method for batch processing.
    """

    def __init__(self, device):
        super().__init__()
        if not device:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        elif device.startswith("{") and device.endswith("}"):
            # If the device is a dictionary, we assume it's a device map
            device = eval(device)
        self.requested_device = device

    def forward(self, prompt: str, image_path: str, **generation_kwargs) -> dict:
        return self.forward_batch([prompt], [image_path], **generation_kwargs)[0]

    def forward_batch(self, prompts: list[str], image_paths: list[str], **generation_kwargs) -> list[dict]:
        return list(map(partial(self.forward, **generation_kwargs), prompts, image_paths))


def hf_inference(inputs, model, processor, do_sample=False, max_new_tokens=512, **gen_args):
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, do_sample=do_sample, max_new_tokens=max_new_tokens, **gen_args)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    return [{"response": response.strip()} for response in output_texts]


@register_model("gemini-2.0-flash")
@register_model("gemini-2.5-flash")
class GeminiAPI(InferenceBase):

    def __init__(self, model_name: str, device: str):
        super().__init__(device)
        self.model_name = model_name
        self.key_storage = KeyStorage()

    def forward_batch(self, prompts, image_paths):
        with Pool(len(prompts)) as pool:
            results = pool.starmap(self.forward, zip(prompts, image_paths))
        return results

    def forward(self, prompt: str, image_path: str, _attempt: int = 0, **generation_kwargs) -> dict:
        import google
        import google.generativeai as genai

        if generation_kwargs and len(generation_kwargs) > 0:
            raise NotImplementedError("Generation kwargs are not supported for this model.")

        assert Path(image_path).exists(), f"Image path {image_path} does not exist"

        api_key = self.key_storage.get_key("GEMINI_API_KEY")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=self.model_name)

        image_parts = [
            {
                "mime_type": mimetypes.MimeTypes().guess_type(image_path)[0],
                "data": Path(image_path).read_bytes(),
            },
        ]
        prompt_parts = [image_parts[0], "\n" + prompt]

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        generation_config = genai.types.GenerationConfig(
        )

        retval = None

        try:
            response = model.generate_content(
                prompt_parts,
                safety_settings=safety_settings,
                generation_config=generation_config,
            )
            retval =  {"response": response.text.strip(), "prompt": prompt}
        except ValueError as e:
            logging.error(f"Error generating response.")
            logging.error(f"Prompt feedback: {response.prompt_feedback}")

            if response.prompt_feedback.block_reason is not None:
                retval =  {"response": None, "error": "blocked"}
            else:  # cant handle this case
                raise e
        except google.api_core.exceptions.InternalServerError:
            logging.error(f"Internal server error")
            retval =  {"response": None, "error": "internal_server_error"}

        except google.api_core.exceptions.ResourceExhausted as e:
            if _attempt > 3:
                logging.error(f"Resource exhausted for {api_key}, giving up")
                retval =  {"response": None, "error": "resource_exhausted"}
            else:
                logging.warning(f"Resource exhausted for {api_key}, retrying (Attempt {1 + _attempt})...")
                time.sleep(60) # this is a bit of a hack, but it works, ideally we would want to wait for the quota to reset
                # retry
                retval = self.forward(prompt, image_path, _attempt=_attempt + 1, **generation_kwargs)
        except Exception as e:
            logging.error(f"Error generating response. {e}")
            retval =  {"response": None, "error": "unknown_error"}

        return retval
    

@register_model("o3-2025-04-16")
@register_model("o4-mini-2025-04-16")
class OpenAIAPI(InferenceBase):

    def __init__(self, model_name: str, device: str):
        super().__init__(device)
        
        self.model_name = model_name
        self.key_storage = KeyStorage()
        

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
        
    def forward_batch(self, prompts, image_paths):
        with Pool(len(prompts)) as pool:
            results = pool.starmap(self.forward, zip(prompts, image_paths))
        return results

    def forward(self, prompt: str, image_path: str, **generation_kwargs) -> dict:

        if generation_kwargs and len(generation_kwargs) > 0:
            raise NotImplementedError("Generation kwargs are not supported for this model.")

        assert Path(image_path).exists(), f"Image path {image_path} does not exist"

        from openai import OpenAI

        self.client = OpenAI(
            api_key=self.key_storage.get_key("OPENAI_API_KEY"),
        )

        base64_image = self.encode_image(image_path)
        
        response = self.client.responses.create(
            model=self.model_name,
            input=[
                {
                    "role": "user",
                    "content": [
                        { "type": "input_text", "text": prompt },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                }
            ]
        )

        retval = None
        try:
            response_str = response.output_text
            retval =  {"response": response_str}
        except:
            logging.error(f"Error generating response. {response}")
            retval =  {"response": None, "error": "unknown_error"}

        return retval
    

@register_model("horizon-alpha")
@register_model("horizon-beta")
@register_model("kimi-vl-a3b-thinking")
class OpenRouterAPI(InferenceBase):

    def __init__(self, model_name: str, device: str):
        super().__init__(device)

        if model_name.startswith("horizon-"):
            self.model_name = "horizon/" + model_name
        elif model_name == "kimi-vl-a3b-thinking":
            self.model_name = "moonshotai/kimi-vl-a3b-thinking:free"

        self.key_storage = KeyStorage()
        

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
        
    def forward_batch(self, prompts, image_paths, **generation_kwargs):
        with Pool(len(prompts)) as pool:
            results = pool.starmap(partial(self.forward, **generation_kwargs), zip(prompts, image_paths))
        return results

    def forward(self, prompt: str, image_path: str, **generation_kwargs) -> dict:

        if generation_kwargs and len(generation_kwargs) > 0:
            raise NotImplementedError("Generation kwargs are not supported for this model.")

        assert Path(image_path).exists(), f"Image path {image_path} does not exist"

        from openai import OpenAI

        self.client = OpenAI(
            api_key=self.key_storage.get_key("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )

        base64_image = self.encode_image(image_path)
        
        response = self.client.chat.completions.create(
            extra_headers={
                # "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
                # "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
            },
            extra_body={},
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": prompt },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                    ],
                }
            ]
        )

        retval = None
        try:
            response_str = response.choices[0].message.content
            retval =  {"response": response_str}
        except:
            logging.error(f"Error generating response. {response}")
            retval =  {"response": None, "error": "unknown_error"}

        return retval


@register_model("Llama-4-Scout-17B-16E-Instruct")
class LLama4(InferenceBase):
    def __init__(self, model_name: str, device: str):
        super().__init__(device)

        from transformers import AutoProcessor, Llama4ForConditionalGeneration
        import torch

        model_id = "meta-llama/" + model_name

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Llama4ForConditionalGeneration.from_pretrained(
            model_id,
            # attn_implementation="flash_attention2",  # does not work with flash attention or flex attention
            device_map=self.requested_device,
            torch_dtype=torch.bfloat16,
        ).eval()

        print(self.model.hf_device_map)
        

    def forward(self, prompt: str, image_path: str, **generation_kwargs) -> dict:
        assert Path(image_path).exists(), f"Image path {image_path} does not exist"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": image_path},
                    {"type": "text", "text": prompt},
                ]
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, torch.bfloat16)

        return hf_inference(inputs, self.model, self.processor, **generation_kwargs)[0]


@register_model("llava-onevision-qwen2-0.5b-ov-hf")
@register_model("llava-onevision-qwen2-7b-ov-hf")
@register_model("llava-onevision-qwen2-72b-ov-hf")
class LLavaOneVision(InferenceBase):
    def __init__(self, model_name: str, device: str):
        super().__init__(device)

        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
        import torch

        self.processor = AutoProcessor.from_pretrained("llava-hf/" + model_name) 
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            "llava-hf/" + model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=self.requested_device
        ).eval()

        print(self.model.hf_device_map)
    
    def forward(self, prompt, image_path, **generation_kwargs):

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(conversation, 
                                                    add_generation_prompt=True, 
                                                    padding=True,
                                                    tokenize=True, 
                                                    return_dict=True, 
                                                    return_tensors="pt")
        
        return hf_inference(inputs, self.model, self.processor, **generation_kwargs)[0]


@register_model("llava-1.5-7b-hf")
@register_model("llava-1.5-13b-hf")
class LLava1_5(InferenceBase):
    def __init__(self, model_name: str, device: str):
        super().__init__(device)

        from transformers import AutoProcessor, LlavaForConditionalGeneration
        import torch

        self.processor = AutoProcessor.from_pretrained("llava-hf/" + model_name) 
        self.model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/" + model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=self.requested_device
        ).eval()

    
    def forward(self, prompt, image_path, **generation_kwargs):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
        ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        raw_image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=raw_image, text=text, return_tensors='pt').to(torch.float16)

        return hf_inference(inputs, self.model, self.processor, **generation_kwargs)[0]


@register_model("llava-v1.6-vicuna-7b-hf")
@register_model("llava-v1.6-vicuna-13b-hf")
class LLavaNext(InferenceBase):
    def __init__(self, model_name: str, device: str):
        super().__init__(device)

        from transformers import AutoProcessor, LlavaNextForConditionalGeneration
        import torch

        self.processor = AutoProcessor.from_pretrained("llava-hf/" + model_name) 
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/" + model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=self.requested_device,
            attn_implementation="flash_attention_2"
        ).eval()

    
    def forward(self, prompt, image_path, **generation_kwargs):
        messages = [
            [{
                "role": "user",
                "content": [
                    {"type": "image", "path": image_path},
                    {"type": "text", "text": prompt},
                ],
            },]
        ]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        return hf_inference(inputs, self.model, self.processor, **generation_kwargs)[0]


@register_model("SmolVLM-Instruct")
class SmolVLM(InferenceBase):
    def __init__(self, model_name: str, device: str):
        super().__init__(device)

        del model_name  # not used, but required by the interface

        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch

        model_id = "HuggingFaceTB/SmolVLM-Instruct"

        self.processor = AutoProcessor.from_pretrained(model_id, padding_side='left')
        self.model = AutoModelForVision2Seq.from_pretrained(model_id,
                                                        torch_dtype=torch.bfloat16,
                                                        device_map=self.requested_device,
                                                        _attn_implementation="flash_attention_2").eval()

    def forward(self, prompt, image_path, **generation_kwargs):
        from transformers.image_utils import load_image

        # Load image
        image = load_image(image_path)

        # Create input messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Prepare inputs
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
        inputs = self.processor(text=text, 
                                images=[image], 
                                padding=True,
                                return_tensors="pt")        
        
        return hf_inference(inputs, self.model, self.processor, **generation_kwargs)[0]


@register_model("Qwen2.5-VL-3B-Instruct")
@register_model("Qwen2.5-VL-7B-Instruct")
@register_model("Qwen2.5-VL-32B-Instruct")
@register_model("Qwen2.5-VL-72B-Instruct")
class QwenVL_2_5(InferenceBase):
    def __init__(self, model_name: str, device: str):
        super().__init__(device)

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
       
        model_id = "Qwen/" + model_name # Qwen2.5-VL-7B-Instruct

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=self.requested_device,
            trust_remote_code=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, padding_side='left', trust_remote_code=True)

        print(self.model.hf_device_map)

    def forward(self, prompt, image_paths, **generation_kwargs):
        from qwen_vl_utils import process_vision_info

        if type(image_paths) is str:
            image_paths = [image_paths]

        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": image_path} for image_path in image_paths],
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Prepare the input for the processor
        texts = [
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ]

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        return hf_inference(inputs, self.model, self.processor, **generation_kwargs)[0]


@register_model("InternVL3-1B")
@register_model("InternVL3-2B")
@register_model("InternVL3-8B")
@register_model("InternVL3-9B")
@register_model("InternVL3-14B")
@register_model("InternVL3-38B")
@register_model("InternVL3-78B")
@register_model("InternVL3-1B-max")
@register_model("InternVL3-2B-max")
@register_model("InternVL3-8B-max")
@register_model("InternVL3-9B-max")
@register_model("InternVL3-14B-max")
@register_model("InternVL3-38B-max")
@register_model("InternVL3-78B-max")
class InternVL3(InferenceBase):

    @staticmethod
    def build_transform(input_size):
        import torchvision.transforms as T

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    @staticmethod
    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    @staticmethod
    def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = InternVL3.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    @staticmethod
    def load_image(image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert('RGB')
        transform = InternVL3.build_transform(input_size=input_size)
        images = InternVL3.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values


    def __init__(self, model_name: str, device: str):
        super().__init__(device)

        from transformers import AutoProcessor, AutoModel

        self.max_image_slices = 12
        if model_name.endswith("-max"):
            model_name = model_name[:-4]
            self.max_image_slices = 40
            print(f"Using max image slices for {model_name}")

        model_id = "OpenGVLab/" + model_name

        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=self.requested_device).eval()
        
        self.model.config.output_hidden_states = False
        self.model.config.output_attentions = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)

        print(self.model.hf_device_map)


    def forward(self, prompt: str, image_path: str, **generation_kwargs) -> dict:
        assert Path(image_path).exists(), f"Image path {image_path} does not exist"

        pixel_values = InternVL3.load_image(image_path, max_num=self.max_image_slices).to(torch.bfloat16).to(self.model.device)
        generation_config = dict(max_new_tokens=512, do_sample=generation_kwargs.pop("do_sample", False), **generation_kwargs)

        input_prompt = "<image>\n" + prompt

        torch.cuda.empty_cache()

        with torch.inference_mode():
            try:
                response = self.model.chat(self.tokenizer, pixel_values, input_prompt, generation_config,
                                    history=None, return_history=False)
            except torch.OutOfMemoryError:
                for device_index in range(torch.cuda.device_count()):
                    occupied_bytes = torch.cuda.memory_allocated(device_index)

                    # Memory cached/reserved by the allocator (may be higher than occupied)
                    reserved_bytes = torch.cuda.memory_reserved(device_index)

                    print(f"Device {device_index} memory usage:")
                    print(f"Occupied: {occupied_bytes / 1024**2:.2f} MB")
                    print(f"Reserved: {reserved_bytes / 1024**2:.2f} MB")
                    print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(device_index) / 1024**2:.2f} MB")
            
        
        return {"response": response}


@register_model("paligemma2-3b-mix-448")
@register_model("paligemma2-10b-mix-448")
@register_model("paligemma2-28b-mix-448")
class PaliGemma2(InferenceBase):
    def __init__(self, model_name: str, device: str):
        super().__init__(device)

        from transformers import (
            PaliGemmaProcessor,
            PaliGemmaForConditionalGeneration,
        )

        model_id = "google/" + model_name

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            device_map=self.requested_device).eval()
        self.processor = PaliGemmaProcessor.from_pretrained(model_id)

    def forward(self, prompt: str, image_path: str, **generation_kwargs) -> dict:
        assert Path(image_path).exists(), f"Image path {image_path} does not exist"
        from transformers.image_utils import load_image

        image = load_image(image_path)

        input_prompt = "<image>question en " + prompt
        inputs = self.processor(text=input_prompt, 
                                images=image, 
                                return_tensors="pt").to(torch.bfloat16)
        
        return hf_inference(inputs, self.model, self.processor, **generation_kwargs)[0]


@register_model("gemma-3n-e2b-it")
@register_model("gemma-3n-e4b-it")
class Gemma3n(InferenceBase):
    def __init__(self, model_name: str, device: str):
        super().__init__(device)

        from transformers import AutoProcessor, AutoModelForImageTextToText
        import torch

        torch.set_float32_matmul_precision('high')

        model_id = "google/" + model_name # google/gemma-3n-e2b-it
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id,
                                                                device_map=self.requested_device).eval()

    def forward(self, prompt: str, image_path: str, **generation_kwargs) -> dict:
        assert Path(image_path).exists(), f"Image path {image_path} does not exist"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image_path},
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        return hf_inference(inputs, self.model, self.processor, disable_compile=generation_kwargs.pop("disable_compile", False), **generation_kwargs)[0]


@register_model("internlm-xcomposer2d5-7b")
class InternVLXComposer2_5(InferenceBase):
    def __init__(self, model_name: str, device: str):
        super().__init__(device)
        torch.set_grad_enabled(False)
        self.model = AutoModel.from_pretrained('internlm/' + model_name, 
                                               torch_dtype=torch.bfloat16, 
                                               trust_remote_code=True,
                                               device_map=self.requested_device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained('internlm/' + model_name, 
                                                       trust_remote_code=True)
        self.model.tokenizer = self.tokenizer

    def forward(self, prompt: str, image_path: str) -> dict:
        assert Path(image_path).exists(), f"Image path {image_path} does not exist"

        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            response, _ = self.model.chat(self.tokenizer, prompt + " Do not explain.", [image_path], do_sample=False, num_beams=3, use_meta=True, hd_num=15, pad_token_id=self.tokenizer.eos_token_id)
        
        return {"response": response.strip()}


@register_model("internlm-xcomposer2-4khd-7b")
class InternVLXComposer2(InferenceBase):
    def __init__(self, model_name: str, device: str):
        super().__init__(device)
        torch.set_grad_enabled(False)
        self.model = AutoModel.from_pretrained('internlm/' + model_name, 
                                               torch_dtype=torch.bfloat16, 
                                               trust_remote_code=True,
                                               device_map=self.requested_device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained('internlm/' + model_name, 
                                                       trust_remote_code=True)
        self.model.tokenizer = self.tokenizer

    def forward(self, prompt: str, image_path: str, **generation_kwargs) -> dict:
        assert Path(image_path).exists(), f"Image path {image_path} does not exist"

        if generation_kwargs and len(generation_kwargs) > 0:
            raise NotImplementedError("Generation kwargs are not supported for this model.")

        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            response, _ = self.model.chat(self.tokenizer, query="<ImageHere>" + prompt, image=image_path, do_sample=False, num_beams=3, hd_num=25, pad_token_id=self.tokenizer.eos_token_id)
        
        return {"response": response.strip()}


@register_model("VILA-HD-8B-PS3-1.5K-SigLIP")
@register_model("VILA-HD-8B-PS3-4K-SigLIP")
class NVilaHD(InferenceBase):
    def __init__(self, model_name: str, device: str):
        super().__init__(device)
        self.model_name = model_name

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def forward(self, prompt: str, image_path: str, **generation_kwargs) -> dict:
        from openai import OpenAI

        if generation_kwargs and len(generation_kwargs) > 0:
            raise NotImplementedError("Generation kwargs are not supported for this model.")

        client = OpenAI(
            base_url="http://localhost:8000",
            api_key="fake-key",
        )

        base64_image = self.encode_image(image_path)

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model=self.model_name,
        )

        return {"response": response.choices[0].message.content[0]["text"]}    
        
@register_model("deepseek-vl2")
@register_model("deepseek-vl2-small")
@register_model("deepseek-vl2-tiny")
class DeepSeekVL2(InferenceBase):

    def check_install(self):
        try:
            import deepseek_vl2
        except Exception as e:
            logging.critical(
                'Please first install deepseek_vl2 from source codes in: https://github.com/deepseek-ai/DeepSeek-VL2')
            raise e

    def __init__(self, model_name: str, device: str):
        super().__init__(device)

        self.check_install()

        from transformers import AutoModelForCausalLM
        from deepseek_vl2.models import DeepseekVLV2Processor

        # specify the path to the model
        model_path = "deepseek-ai/" + model_name
        self.processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer

        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=self.requested_device,
                                                           torch_dtype=torch.bfloat16).eval()

    def forward(self, prompt: str, image_path: str, **generation_kwargs) -> dict:
        assert Path(image_path).exists(), f"Image path {image_path} does not exist"

        if generation_kwargs and len(generation_kwargs) > 0:
            raise NotImplementedError("Generation kwargs are not supported for this model.")

        from deepseek_vl2.utils.io import load_pil_images

        conversation = [
            {
                "role": "<|User|>",
                "content": "<image>\n" + prompt,
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(self.model.device)

        inputs_embeds, past_key_values = self.model.incremental_prefilling(
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            chunk_size=512
        )
        
        # run the model to get the response
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            past_key_values=past_key_values,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512, 
            do_sample=False, 
            use_cache=True
        )

        answer = self.tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=True)

        return {"response": answer.strip()}


@register_model("gemma-3-12b-it")
@register_model("gemma-3-27b-it")
class Gemma3(InferenceBase):
    def __init__(self, model_name: str, device: str):
        super().__init__(device)

        from transformers import Gemma3ForConditionalGeneration, AutoProcessor
       
        model_id = f"google/{model_name}"

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            device_map=self.requested_device,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_id)

    def forward(self, prompt, image_path, **inference_args):


        messages = [
            # {
            #     "role": "system",
            #     "content": [{"type": "text", "text": "You are a helpful assistant."}]
            # },
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        
        return hf_inference(inputs, self.model, self.processor, **inference_args)[0]
    

@register_model("Kimi-VL-A3B-Thinking-2506")
class KimiVLThinking(InferenceBase):
    def __init__(self, model_name: str, device: str):
        super().__init__(device)

        from transformers import AutoModelForCausalLM, AutoProcessor
       
        model_id = f"moonshotai/{model_name}"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            device_map=self.requested_device,
            trust_remote_code=True
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def extract_thinking_and_summary(self, text: str, bot: str = "◁think▷", eot: str = "◁/think▷") -> str:
        if bot in text and eot not in text:
            return ""
        if eot in text:
            return text[text.index(bot) + len(bot):text.index(eot)].strip(), text[text.index(eot) + len(eot) :].strip()
        return "", text
    
    def downsample_to_pixel_limit(img: Image.Image, max_pixels: int) -> Image.Image:
        """
        Downsamples an image so that its total number of pixels does not exceed max_pixels,
        while preserving aspect ratio.

        Parameters:
            img (PIL.Image.Image): The input image.
            max_pixels (int): Maximum allowed number of pixels (width * height).

        Returns:
            PIL.Image.Image: The resized image.
        """
        # Current image size
        w, h = img.size
        current_pixels = w * h

        if current_pixels <= max_pixels or max_pixels <= 0:
            return img  # No resizing needed

        # Calculate scale factor based on pixel ratio
        scale_factor = math.sqrt(max_pixels / current_pixels)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)

        return img.resize((new_w, new_h), Image.LANCZOS)


    def forward(self, prompt, image_paths, **generation_kwargs):

        if type(image_paths) is str:
            image_paths = [image_paths]

        images = [KimiVLThinking.downsample_to_pixel_limit(Image.open(path), 1920*1080) for path in image_paths]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path} for image_path in image_paths
                ] + [{"type": "text", "text": prompt}],
            },
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = self.processor(images=images, text=text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

        response = hf_inference(inputs, self.model, self.processor, do_sample=generation_kwargs.pop("do_sample", True), 
                                temperature=generation_kwargs.pop("temperature", 0.8), **generation_kwargs)[0]["response"]
        thinking, summary = self.extract_thinking_and_summary(response)

        return {"response": summary, "thinking": thinking}
    

@register_model("LFM2-VL-1.6B")
@register_model("LFM2-VL-450M")
class LFM2(InferenceBase):
    def __init__(self, model_name: str, device: str):
        super().__init__(device)

        from transformers import AutoModelForImageTextToText, AutoProcessor
       
        model_id = f"LiquidAI/{model_name}"

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map=self.requested_device,
            torch_dtype="bfloat16",
            trust_remote_code=True
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


    def forward(self, prompt, image_path, **inference_args):
        from transformers.image_utils import load_image
        image = load_image(image_path)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # Generate Answer
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(self.model.device)
        
        return hf_inference(inputs, self.model, self.processor, **inference_args)[0]
