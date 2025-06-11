# Remarkable Robustness of LLMs: Stages of Inference?
## Overview

This repository contains the codebase for the "Remarkable Robustness of Large Language Models" paper. The project aims to investigate the robustness of large language models (LLMs) by swapping and ablation experiments. All supporting experiments in the paper, such as prediction and suppression neuron counting, entropy calculation, and attention visualization, are included in this repository. The codebase is written in Python and uses Jupyter Notebooks for data analysis and visualization.

## Repository Structure

```sh
└── /
    ├── LICENSE
    ├── README.md
    ├── model_intervention.py
    ├── notebooks
        ├── attention_prev5.ipynb
        ├── casestudies
        ├── entropy_calculation.ipynb
        └── neuron_counter.ipynb
    └──requirements.txt
```

## Modules

<details closed><summary>Repository Summary</summary>

| File                                                                                                                                    | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|-----------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [model_intervention.py](https://github.com/vdlad/Remarkable-Robustness-of-LLMs/blob/master/model_intervention.py)                       | Carry out layer swapping and ablation experiments on any model supported by TransformerLens. Computes metrics and conducts interventions to study model behavior and performance and saves to dataframe.                                                                                                                                                                                                                                                                          |
| [requirements.txt](https://github.com/vdlad/Remarkable-Robustness-of-LLMs/blob/master/requirements.txt)                                 | Package requirements for the repository                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| [neuron_counter.ipynb](https://github.com/vdlad/Remarkable-Robustness-of-LLMs/blob/master/notebooks/neuron_counter.ipynb)               | Determine the number of prediction and suppression neurons in any model supported by TransformerLens                                                                                                                                                                                             |
| [entropy_calculation.ipynb](https://github.com/vdlad/Remarkable-Robustness-of-LLMs/blob/master/notebooks/entropy_calculation.ipynb)     | Use the [LogitLens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) technique but then takes the entropy to see the entropy of the model change through the layers.                                                                                                                                                                                              |
| [attention_prev5.ipynb](https://github.com/vdlad/Remarkable-Robustness-of-LLMs/blob/master/notebooks/attention_prev5.ipynb)             | Uses TransformerLens to determine the mean attention on the previous 5 tokens of any input. |
| [subjoiner_heads.ipynb](https://github.com/vdlad/Remarkable-Robustness-of-LLMs/blob/master/notebooks/casestudies/subjoiner_heads.ipynb) | Code for discovering subjoiner heads in language models. A subjoiner head is an attention head responsible for predicting the next token in multi-token words. |
| [probe_neurons.ipynb](https://github.com/vdlad/Remarkable-Robustness-of-LLMs/blob/master/notebooks/casestudies/probe_neurons.ipynb)     |  Probe individual neurons (which you can determine by find_neurons) by training a probe on the activations of the MLP output. It compares individual probes against an ensemble of probes to show that neurons work together to achieve their accuracy, even outperforming the mean model accuracy with the right ensemble.|
| [find_neurons.ipynb](https://github.com/vdlad/Remarkable-Robustness-of-LLMs/blob/master/notebooks/casestudies/find_neurons.ipynb)       | Find the relevant neurons to probe by looking at the product of the unembedding matrix and the output weights of the MLPs.                                                                                                            |

</details>

## Getting Started


### Installation

#### From `source`

1. Clone the repository:

```console
$ git clone https://github.com/vdlad/Remarkable-Robustness-of-LLMs/
```

2. Change to the project directory:

```console
$ cd Remarkable-Robustness-of-LLMs
```

3. Install the dependencies:

```console
$ pip install -r requirements.txt
```

## Cite Us 
```console
@article{lad2024remarkable,
  title={The Remarkable Robustness of LLMs: Stages of Inference?},
  author={Lad, Vedang and Gurnee, Wes and Tegmark, Max},
  journal={arXiv preprint arXiv:2406.19384},
  year={2024}
}
```




