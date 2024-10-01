<h1 align="center">
Visual Perception in Text Strings
</h1>

Understanding visual semantics embedded in consecutive characters is a crucial capability for both large language models (LLMs) and multi-modal large language models (MLLMs). This type of artifact possesses the unique characteristic that identical information can be readily formulated in both texts and images, making them a significant proxy for analyzing how capable modern LLMs and MLLMs are in modality-agnostic vision understanding. In this project, we select ASCII art as a representative artifact, where the lines and brightness used to depict each concept are rendered by characters, and we frame the problem as an ASCII art recognition task. The data and codes for fine-tuning and evaluate models' performance are presented in this repository.

## Installation

Clone this repo into your working directory and setup the environment:

```python
git clone xxx
cd ASCIIEval
conda create -n ascii python=3.10
conda activate ascii
pip install -r requirements.txt
```

Major requirements are listed in `requirements.txt`. The specific version of these packages may be varied based on the specific LLMs or MLLMs to be trained or evaluated.


## Data

Some examplified training (ASCIITune) and test data (ASCIIEval) are provided in `./data`. 

To begin with, please collect the images for each ascii art by:
```python
bash script/prepare_data.sh
```

Here are descriptions for some important properties of each sample:
* `url`: the source website.
* `ascii_art`: the text string of the ASCII art.
* `category-1/2/3`: the class/group/concept depicted in the ASCII art.
* `choices`: the positive and negative choices for the ascii art recognition task.
* `labels`: the corresponding label for each choice.
* `image_path`: the path to the image modality of the ASCII art.

We appreciate the fantanstic art created by ASCII artists. To focus on measuring the visual perception abilities of models, we made slight modifications to the original ASCII art crawled online. We also keep the original ASCII art (`original_ascii_art` ) in ASCIIEval. The released dataset is restricted to academic use.


## Evaluation

To evaluate LLMs on ASCIIEval locally:
```bash
CUDA_VISIBLE_DEVICES=xxx python3 src/evaluation.py --model_dir /path/to/the/model --model_template xxx --output_file_path xxx.jsonl

```

To evaluate MLLMs on ASCIIEval locally:
```bash
CUDA_VISIBLE_DEVICES=xxx python3 src/evaluation_mm.py --model_dir /path/to/the/model --output_file_path xxx.jsonl --mode both
```
`mode` determines the input modality for MLLMs, including `text-only`, `image-only`, `both`.

To evaluate models through API:
```bash
export API_KEY=xxx
python3 src/evaluation_by_api.py --api_key $API_KEY --model_name xxx --base_url https://xxxxxxxx/v1 --template_name xxx --output_file_path xxx.jsonl --mode text-only
```

## Fine-tuning

To fine-tune an LLM on ASCIITune:
```bash
bash script/train_LLM.sh 
```

To fine-tune an MLLM on ASCIITune:
```bash
bash script/train_MLLM.sh
```
Here, `mode` represents different input modality setting, including `text-only`, `image-only`, `both` and `random`.



