<p align="center">
<img src="./assets/logo.jpg" width="400"> <br>
</p>

# VisualOverload

<p align="center">
[<a href="">📚 Paper (coming soon)</a>] 
[<a href="https://huggingface.co/datasets/paulgavrikov/visualoverload">🤗 Dataset on HuggingFace</a>]
[<a href="https://huggingface.co/spaces/paulgavrikov/visualoverload-submit">🏆 Leaderboard</a>]
[<a href="https://huggingface.co/spaces/paulgavrikov/visualoverload-submit">🎯 Online Evaluator</a>]
</p>


Is basic image understanding really solved in state-of-the-art VLMs? We present VisualOverload, a slightly different visual question answering (VQA) benchmark comprising 2,720 question–answer pairs, with privately held ground-truth responses. Unlike prior VQA datasets that typically focus on near global image understanding, VisualOverload challenges models to perform simple, knowledge-free visual understanding and reasoning of details in densely populated (or, *overloaded*) scenes. Our dataset consists of high-resolution scans of public-domain paintings that are populated with multiple figures, actions, and unfolding subplots set against elaborately detailed backdrops. Questions were handcrafted to probe for a thorough understanding of the scene.

## 📂 Load the dataset

The easiest way to load the dataset is to use HuggingFace's `datasets`.

```python
from datasets import load_dataset

vol_dataset = load_dataset("paulgavrikov/visualoverload")
```

Each sample contains the following fields

- `question_id`: Unique identifier of each question. 
- `image`: A PIL JPEG image. Most of our images match the total pixel count of 4k (3840x2160 px) in different aspect ratios. 
- `question`: A question about the image.
- `question_type`: Type of question. Will be one of `choice` (response expected to be "A", "B", "C", or "D"), `counting` (freeform), or `ocr` (freeform). You can use this information to request a suitable output format. 
- `options`: This is the list of options for `question_type=choice` and empty otherwise. Please treat the options as answers options `A, B, C, D` (4 options) or `A, B` (2 options).
- `difficulty`: Meta-data about the difficulty of the question. One of `easy`, `medium`, or `hard`.
- `category`:  Meta-data about the question task. One of `activity`, `attributes`, `counting`, `ocr`, `reasoning`, or `scene`.
- `default_prompt`: You can use this prompt to stay compliant with our results. It is a simple combination of the question and answers, with some additional output format constraints. This should work well for most models.

## 🎯 Evaluate your model

Please see eval.py for an example evaluation script that generates a correct submission JSON.

All of our ground truth labels are private. The only way to score your submission is to use the [evaluation server](https://huggingface.co/spaces/paulgavrikov/visualoverload-submit). You will need to sign in with a HuggingFace account.  

Your predictions should be a list of dictionaries, each containing an `question_id` field and a `response` field. For multiple choice questions, the `response` field should contain the predicted answer choice. For open-ended questions, the `response` field should contain the option letter (A-D). We will apply simple heuristics to clean the responses, but please ensure they are as accurate as possible.


## 🏆 Submit to the leaderboard
We welcome all submissions for model *or* method (including prompting-based) to our dataset. Please create an issue following the template and include your predictions as JSON. 


## 📝 License

Our dataset is licensed under CC BY-SA 4.0. All images are based on artwork that is royalty-free public domain (CC0).

## 📚 Citation

```latex
Coming soon.
```