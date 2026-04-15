import argparse
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
import torch
import argparse
import os
import base64
import io
from PIL import Image
import itertools
import pandas as pd
from tqdm import tqdm
import random
import wandb


PROMPT = "You are an expert in image analysis. Given two images, your task is to determine which image has a higher visual density and complexity. Do not attempt to indentify the name of the painting. Respond with 'A' if the first image has higher complexity, 'B' if the second image has higher complexity."


def pil_image_to_base64(image, format="jpeg", decode_utf8=True):
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    retval = buffered.getvalue()
    retval = base64.b64encode(retval)
    if decode_utf8:
        retval = f"data:image/{format.lower()};base64," + retval.decode('utf-8')
    return retval


class Qwen():

    def __init__(self, model_name: str):
        super().__init__()

        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )

        if hasattr(self.model, "hf_device_map"):
            print(self.model.hf_device_map)

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    
    def inference(self, prompt, img1, img2):

        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": img1},
                    {"type": "image", "image": img2},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device, torch.bfloat16)
        inputs_len = inputs["input_ids"].shape[1]

        generation_config = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, repetition_penalty=1.0, max_new_tokens=8192)

        with torch.inference_mode():
            response = self.model.generate(**inputs, **generation_config, return_dict_in_generate=True)
                   
        return self.processor.tokenizer.decode(response.sequences[0, inputs_len:-1])


def process_results(result):
    reason = result.split("</think>")[0].replace("<think>", "").strip()
    choice = result.split("</think>")[1].strip()
    return choice, reason


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-27B", help="Qwen 3.5 model ID (non-MoE)")
    parser.add_argument("--path", type=str, help="Path to the dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="qwen_complexity.csv", help="Output file")
    parser.add_argument("--range", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    output_file = args.output
    if args.range:
        output_file = output_file.replace(".csv", f"_{args.range}.csv")

    wandb.init(project="visualoverload_complexity", config=args)

    random.seed(args.seed)

    image_store = {}

    print("Loading images..")
    for image in os.listdir(args.path):
        if image.endswith(".jpg") or image.endswith(".jpeg") or image.endswith(".png"):
            image_store[image] = pil_image_to_base64(Image.open(os.path.join(args.path, image)))
    print(f"Found {len(image_store)} images.")

    tasks = list(itertools.combinations(image_store.keys(), 2))
    random.shuffle(tasks)

    print(f"Total tasks: {len(tasks)}")

    if args.resume:
        df_resume = pd.read_csv(args.resume)
        resume_set = set(tuple(x) for x in df_resume[["img_A", "img_B"]].values)
        resume_set.update((x[1], x[0]) for x in df_resume[["img_A", "img_B"]].values)
        tasks = [task for task in tasks if task not in resume_set]
        print(f"Resuming. Remaining tasks: {len(tasks)}")

    if args.range:
        start, end = tuple(map(int, args.range.split("-")))
        tasks = tasks[start:end]
        print(f"Truncating. Remaining tasks: {len(tasks)}")

    if len(tasks) == 0:
        print("No tasks to run.")
        exit()

    print("Loading model..")
    model = Qwen(args.model)

    print(f"Running inference on {len(tasks)} tasks.")

    rows = []
    for img1, img2 in tqdm(tasks):
        try:
            result = model.inference(PROMPT, image_store[img1], image_store[img2])
            print(f"\n{img1} vs {img2}")
            print(result)
            choice, reason = process_results(result)
            rows.append([img1, img2, choice, reason])
            wandb.log({"img_A": img1, "img_B": img2, "choice": choice, "reason": reason})
            df = pd.DataFrame(rows, columns=["img_A", "img_B", "choice", "reason"])
            df.to_csv(output_file, index=False)
        except Exception as e:
            print(f"Error processing {img1} and {img2}: {e}")
            
    del model
    del image_store
    wandb.finish()

