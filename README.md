<div align= "center">
    <h1> Visual Perception in Text Strings </h1>
</div>


Perceiving visual semantics embedded within consecutive characters is a crucial yet under-explored capability for both Large Language Models (LLMs) and Multi-modal Large Language Models (MLLMs). In this work, we select ASCII art as a representative artifact. It depicts concepts through careful arrangement of characters, which can be formulated in both text and image modalities. We frame the problem as a recognition task, and construct a novel benchmark, ASCIIEval. It covers over 3K samples with an elaborate categorization tree, along with a training set for further enhancement. Encompassing a comprehensive analysis of tens of models through different input modalities, our benchmark demonstrate its multi-faceted diagnostic power. 
* For textual input, proprietary models exhibit the ability for recognizing ASCII art concepts with over 70% accuracy on certain categories, but open-source LLMs lags far behind. We propose rationale-assisted fine-tuning to bridge this gap, which elevates the open-source LLMs performance relatively by 26.10%. 
* For image inputs, we reveal that the latest open-source MLLMs over-emphasize fine-grained text recognition at the expense of perceiving collective visual information, leading to the dramatic gap of over 37% accuracy compared with GPT-5. To address this, we introduce a test-time, low-resolution prompting strategy and a vision-focused fine-tuning approach to activate models' perception ability.
* Another critical finding is that model performance is sensitive to the length of the ASCII art, with this sensitivity varying across input modalities. Unfortunately, none of the models could successfully benefit from the simultaneous provision of both modalities, highlighting the need for more flexible modality-fusion approaches.


<h1 align="center">
<img src="./illustration.png" alt="Overview of ASCIIEval"/>
<br>
</h1>


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

We express our gratitude to the ASCII artists whose fantastic creations underpin our research. In order to assess the visual perception abilities of models, we made slight modifications to the original ASCII art for the test set ASCIIEval. Meanwhile, we retained the original ASCII art (`original_ascii_art`) and provided the URL (`url`) to the data source. It is important to note that our data is licensed under CC BY NC 4.0, which permits only non-commercial use and is intended exclusively for research purposes.

Some examplified training (ASCIITune) and test data (ASCIIEval) are provided in `./data`. 

To begin with, please collect the images for each ASCII art by:
```python
bash script/prepare_data.sh
```

Here are descriptions for some important properties of each sample:
* `url`: the source webpage.
* `ascii_art`: the text string of the ASCII art.
* `category-1/2/3`: the class/group/concept depicted in the ASCII art.
* `choices`: the positive and negative choices for the ascii art recognition task.
* `labels`: the corresponding label for each choice.
* `image_path`: the path to the image modality of the ASCII art.

Statistics of ASCIIEval and ASCIITune are as follows:
<table><thead>
  <tr>
    <th></th>
    <th>#Samples</th>
    <th>#Concepts</th>
    <th>#Characters<br>(Min / Max / Avg)</th>
    <th>#Lines<br>(Min / Max / Avg)</th>
  </tr></thead>
<tbody>
  <tr>
    <td>ASCIIEval</td>
    <td>3,526</td>
    <td>359</td>
    <td>4 / 15,282 / 63,553</td>
    <td>1 / 100 / 16.97</td>
  </tr>
  <tr>
    <td>ASCIITune</td>
    <td>11,836</td>
    <td>2,307</td>
    <td>1 / 13,569 / 62,238</td>
    <td>1 / 97 / 15.22</td>
  </tr>
</tbody>
</table>


## Evaluation

To evaluate LLMs on ASCIIEval locally:
```bash
CUDA_VISIBLE_DEVICES=xxx python3 src/evaluation.py --model_dir /path/to/the/model --output_file_path xxx.jsonl

```

To evaluate MLLMs on ASCIIEval locally:
```bash
CUDA_VISIBLE_DEVICES=xxx python3 src/evaluation_mm.py --model_dir /path/to/the/model --output_file_path xxx.jsonl --mode both
```
`mode` determines the input modality for MLLMs, including `text-only`, `image-only`, `both`.

To evaluate models through API:
```bash
export API_KEY=xxx
python3 src/evaluation_by_api.py --api_key $API_KEY --model_name xxx --base_url https://xxxxxxxx/v1 --output_file_path xxx.jsonl --mode text-only
```

## Fine-tuning

To fine-tune an LLM on ASCIITune:
```bash
bash script/train_LLM.sh 
bash script/train_LLM_w_rational.sh
```

To fine-tune an MLLM on ASCIITune:
```bash
bash script/train_MLLM.sh
```
Here, `mode` represents different input modality setting, including `text-only`, `image-only`, `both` and `random`.
