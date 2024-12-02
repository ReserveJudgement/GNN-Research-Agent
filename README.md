![bgd6](https://github.com/user-attachments/assets/d6cd6fd0-b43e-4e8a-ac94-0ac2cac36347)


## GNN-Research-Agent
Experiment using an LLM agent as an autonomous research assistant to analyze neurological data in graph format.
The dataset being analyzed is "flywire", the connectome of an adult female fly, as described in: 

https://www.biorxiv.org/content/10.1101/2023.06.27.546656v2

The task of the agent is to train and evaluate a GNN over part of the connectome using link predicition.

## Agent
The agent is either commercial (google API) or open source (using hugging-face transformers).
There are two fixed predefined prompts: a description of the dataset and a description of the task.
The agent loop:

- The agent is prompted with the dataset description and the task, and asked to generate code.
- The code is run. If there are errors, they are fed back to the agent together with its generated code, with a prompt to fix the code. Three errors in a row exits the process.
- If the code runs without errors, the evaluation of the resulting GNN is returned to the agent, with a prompt to write a summary report.

Agent process can be found in LLM-researcher.py

## GNN
The agent is prompted to use the NetworkX and PyTorchGeometric (PyG) packages.

NetworkX: https://networkx.org/

PyG: https://pytorch-geometric.readthedocs.io/en/latest/

Working human-made baseline code can be found in GNN-human-implemented.py

## Flywire data
The dataset can be downloaded from: https://codex.flywire.ai/

The agent does not have to deal with downloading the dataset or the packages, these are prepared in advance in the environment.

Connectome tools package is required for loading the data: https://github.com/alitwinkumar/connectome_tools

Example code for analyzing the dataset can be found at: https://github.com/alitwinkumar/connectome_examples

## Intermediate Results
Gemini manages to produce working code on the third trial.

The produced code can be found in gemini/LLM-code.txt

The results of the trained model can be found in gemini/results.csv

Outcomes are not good. Compare:

Human implemented model performance:

![human_losses](https://github.com/user-attachments/assets/3454a1ae-f903-4fa5-87b0-55b25a91553d)

Gemini implemented model performance:

![gemini_losses1](https://github.com/user-attachments/assets/ae85785f-750a-4bb7-ad78-2ce385c90fd5)


The model's summary report can be found in gemini/report.txt

The model correctly concludes that "The training process revealed significant instability and poor performance". It makes recommendations for improvement.

## Next steps

- Expand the agent cycle to improve on performance, not just code errors
- Evaluate additional models
- Add further tasks


