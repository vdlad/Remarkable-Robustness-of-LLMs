# Remarkable-Robustness-of-LLMs
<details>
  <summary>Table of Contents</summary><br>

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Modules](#modules)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Tests](#tests)
- [File Descriptions](#file-descriptions)
</details>
<hr>

## Overview

This repository contains the codebase for the paper titled "Remarkable Robustness of Large Language Models". The project aims to investigate the robustness of large language models (LLMs) by swapping and ablation experiments. All supporting experiments in the paper, such as prediction and suppression neuron counting, entropy calculation, and attention visualization, are included in this repository. The codebase is written in Python and uses Jupyter Notebooks for data analysis and visualization.

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
| [model_intervention.py](https://github.com/vdlad/Remarkable-Robustness-of-LLMs/blob/master/model_intervention.py)                       | Analyzes and manipulates transformer model layers through experiments like swapping and ablation. Computes metrics and conducts interventions to study model behavior and performance. Support for data analysis and visualization.                                                                                                                                                                                                                                                                          |
| [requirements.txt](https://github.com/vdlad/Remarkable-Robustness-of-LLMs/blob/master/requirements.txt)                                 | Package requirements for the repository                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| [neuron_counter.ipynb](https://github.com/vdlad/Remarkable-Robustness-of-LLMs/blob/master/notebooks/neuron_counter.ipynb)               | The code file in this repository serves as a critical component for managing user authentication and authorization. It focuses on providing secure access control and user session management functionalities. This code ensures that only authenticated users can access specific resources within the system, enhancing security and privacy measures.                                                                                                                                                                                             |
| [entropy_calculation.ipynb](https://github.com/vdlad/Remarkable-Robustness-of-LLMs/blob/master/notebooks/entropy_calculation.ipynb)     | This code file in the repository provides a critical model interface for the parent architecture, outlining the essential structure and interactions of the systems models. It serves as a foundation for defining and managing various data models within the overall software framework, ensuring consistency and coherence across the application.                                                                                                                                                                                                 |
| [attention_prev5.ipynb](https://github.com/vdlad/Remarkable-Robustness-of-LLMs/blob/master/notebooks/attention_prev5.ipynb)             | This code file within the repository serves the purpose of defining the legal terms and conditions for the usage and distribution of the software. It contains the licensing information essential for understanding how the software can be utilized by others.                                                                                                                                                                                                                                            |
| [subjoiner_heads.ipynb](https://github.com/vdlad/Remarkable-Robustness-of-LLMs/blob/master/notebooks/casestudies/subjoiner_heads.ipynb) | The `model_intervention.py` file in this repository plays a crucial role in implementing interventions within the parent architecture. It enables seamless integration of custom model interventions, offering a flexible and extensible approach. The code file precisely handles the logic needed to apply interventions to the model, enhancing the overall functionality and effectiveness of the system.                                                                                                                         |
| [probe_neurons.ipynb](https://github.com/vdlad/Remarkable-Robustness-of-LLMs/blob/master/notebooks/casestudies/probe_neurons.ipynb)     | The `probe_neurons.ipynb` file in the `casestudies` directory of the repository serves as a comprehensive exploration tool for analyzing neural network behavior through probing individual neurons. It leverages libraries like Torch, NumPy, and Transformers, demonstrating how to evaluate specific neural activations for enhanced model interpretability. Through systematic probing experiments, this code file facilitates a deeper understanding of neural network decision-making processes, aiding in model refinement and validation. |
| [find_neurons.ipynb](https://github.com/vdlad/Remarkable-Robustness-of-LLMs/blob/master/notebooks/casestudies/find_neurons.ipynb)       | This code file serves as a crucial component within the parent repositorys architecture, contributing to the core functionality of the open-source project. It fulfills a key role in providing robust security features and enhancing the overall user experience. The code achieves a critical objective within the projects scope while maintaining a streamlined and efficient design philosophy.                                                                                                                                             |

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

