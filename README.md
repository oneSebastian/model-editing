# Model Editing
This repository contains a unified framework to run model editing datasets alongside with [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) control tasks on various model editors. It is designed to be easily extendable with support for additional models, model editors, and evaluation datasets. The *MEMIT* model editor was developed by [Meng et al. (2023)](https://arxiv.org/abs/2210.07229) and the implementation is adapted from [here](https://github.com/kmeng01/memit).

## Installation
Install the `model_editing` package with `pip install -e .` together with the dependencies in `requirements.txt` or use use the code directly. Before usage create `config/default_config.yaml` following the lines of the example config.

## Datasets
This model editing benchmark by default includes a number of knowledge editing datasets. We distribute a slightly augmented version of *RippleEdits* with this repository. The other datasets are downloaded as they are needed.

- **RippleEdits:**  
  The *RippleEdits* dataset was released by [Cohen et al. (2023)](https://arxiv.org/abs/2307.12976). The original data can be found [here](https://github.com/edenbiran/RippleEdits). We use and distribute here an augmented version of the dataset. The authors' implementation contains calls to the wikidata API to resolve entitiy codes into entitiy labels. We resolved the API calls and added the entity labels where possible, thus creating the extended static dataset distributed here.

- **zsRE and CounterFact:**
  We use the same dataset splits as [Meng et al. (2023)](https://arxiv.org/abs/2202.05262). We use the datasets distributed by them [here](https://rome.baulab.info/data/). The *CounterFact* dataset was created by them. The *zsRE* dataset was first created by [Levy et al. (2017)](https://aclanthology.org/K17-1034/).

- **MQuAKE:**
  The *MQuAKE* dataset was released by [Zhong et al. (2024)](https://arxiv.org/abs/2305.14795). The data can be found [here](https://github.com/princeton-nlp/MQuAKE).

## Usage
The scripts `example_evaluation.sh` and `example_analysis.sh` give examples of how to evaluate a model on a given set of editors, model editing datasets and control tasks. And how to get a basic overview over the evaluation results. The EvalResult class from `model_editing/analysis.py` can be used to load pandas dataframes of the evaluation results for a more fine-grained analysis using the methods *load_editing_data*, *aggregate_editing_data* and *load_aggregated_control_data* respectively.

## Extensions
This benchmark can be extended in several ways:

- **Adding a new model**  
  Simply add a new case to *load_model* in `model_editing/models.py`. Note, however, that some editors such as *MEMIT* require specific hyperparmaters for each model. In the case of *MEMIT* these can be found in `model_editing/modeleditor/rome_style/hparams/MEMIT/`.

- **Adding an additional model editing dataset**  
  Support for additional datasets can be added to `model_editing/editing_tasks/datasets/`. Follow the example of existing datasets in parsing the new datset, extracting a list of examples consisting of edit facts and test cases containung (multiple) queries.

- **Adding a new model editor**  
  Model editors are implemented in `model_editing/modeleditor/`. Each editor must implement the methods *edit_model* and *restore_model* to inject a list of edits and to undo any edits respectively. For editors that update model parameters these methods suffice. The evaluation is automatically run on the updated model. Editors that inject statements into the prompts put to the model, such as *in-context* and *context-retriever*, must additionaly implement the methods used to prepare these prompts. See `model_editing/modeleditor/util.py` or one of those editors for more details.
