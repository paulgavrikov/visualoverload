from datasets import load_dataset
import json
from argparse import ArgumentParser
from models import load_model
from tqdm import tqdm


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name", default="InternVL3-1B")
    args = parser.parse_args()

    model = load_model(args.model, device="auto")

    # load the visualoverload dataset from the hub
    vol_dataset = load_dataset("paulgavrikov/visualoverload")
    responses = []

    for question in tqdm(vol_dataset["test"]):

        question["image"].save("image.jpeg")  # this is a PIL image and we save it to a file for our script, 
                                              # you can also modify the model to accept PIL images directly 
        
        # you have two options here, either use our default prompts
        # or create your own prompt from the available fields:
        # item["question"], item["options"], item["question_type"] (which is either "choice", "ocr", or "count")
        prompt = question["default_prompt"]
        
        response = model.forward(prompt, "image.jpeg")["response"]

        response = {
            "question_id": question["question_id"],
            "response": response,
        }
        responses.append(response)

    with open("my_prediction.json", "w") as f:
        json.dump(responses, f)
