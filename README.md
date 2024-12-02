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

Working example code can be found in GNN.py

## Flywire data
The dataset can be downloaded from: https://codex.flywire.ai/

The agent does not have to deal with downloading the dataset or the packages, these are prepared in advance.

Connectome tools package is required for loading the data: https://github.com/alitwinkumar/connectome_tools

Example code for analyzing the dataset can be found at: https://github.com/alitwinkumar/connectome_examples
