import connectome_tools
from connectome_tools.connectome_loaders import load_flywire
import textwrap
import torch
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm
import pandas as pd
import json
import os
import google.generativeai as genai
import time
import re
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

device = "cuda" if torch.cuda.is_available() else "cpu"


data_description = """FlyWire is a connectome of a female drosophila brain.

The FlyWire neurons can be loaded using the python script:

neurons, J, nts_Js = load_flywire('data/flywire', by_nts=True, include_spatial=False)

Data description:
neurons is a dataframe containing information for each of the approximately 140,000 neurons in the dataset.
It has the following columns:
• root_id: A unique numerical identifier for each neuron. You can search using this identifier in the
codex web interface.
• group: auto-generated group (based on primary input and output brain regions)
• nt_type: the predicted neurotransmitter type
• nt_type score: the probability associated with the predicted neurotransmitter type (i.e., the maximum
of da_avg, ser_avg, gaba_avg, glut_avg, ach_avg, oct_avg)
• da_avg, ser_avg, gaba_avg, glut_avg, ach_avg, oct_avg are the neurotransmitter predictions (expressed as probabilities for each neurotransmitter type)
• flow, super_class, class, sub_class, cell_type, hemibrain_type, hemilineage, side, nerve: From
classification.csv. These are anatomical properties that tend to be the most useful for selecting subsets of neurons.
• x, y, z: Information on neuronal location, but imprecise.
• x_presyn, y_presyn, z_presyn,x_postsyn, y_postsyn, z_postsyn: The x,y,z locations of each neuron’s synapses, averaged either over pre-synaptic sites or over post-synaptic sites. Calculated from
synapse coordinates.csv. presyn denotes the average location of outgoing synapses, and postsyn
denotes the average location of incoming synapses.
• rho_presyn and rho_postsyn are the standard deviations of outgoing and incoming synapse locations
respectively.
• J_idx, J_idx_post, J_idx_pre: The correspondence between the index in the connectivity matrix J
and the entries in the dataframe. When initially loaded, these three columns are identical and just
contain their corresponding row index, but this may change if the filter connectivity function is
called later.

The connectivity matrix J contains nonnegative integer entries Jij equal to the number of synapses
from the jth neuron onto the ith neuron in the dataset (where i, j use the same indexing as the neurons
dataframe).
J_nts is a python dict with six keys:
• J_nts[’ACH’] contains cholinergic synapse counts. Acetylcholine is the main excitatory neurotransmitter in flies.
• J_nts[’GABA’] contains GABAergic synapse counts. GABA is an inhibitory neurotransmitter in flies.
• J_nts[’GLUT’] contains glutamatergic synapse counts. Glutamate is thought to be mostly inhibitory
in the fly central brain.
• J_nts[’DA’], J_nts[’OCT’], J_nts[’SER’] contains dopaminergic, octopaminergic, and serotonergic
synapse counts. These are typically modeled as neuromodulators, with dopamine for instance being
required for mushroom body-dependent learning.
J is a sum of these six matrices: J = JACH + JGABA + JGLUT + JDA + JOCT + JSER.

Modeling considerations:
Connections involving fewer than 5 or 10 synapses are often treated as unreliable and discarded. Additionally,
models sometimes use normalized synaptic counts; i.e. Jij ← Jij/ Pk Jik, as a measure of connection
strength, since fly neurons have a wide range in number of incoming synapses.

Connection data can be loaded using the python script:
connections = pd.read_csv('data/flywire/connections.csv')
In this dataframe, each row represents a connection between a pair of FlyWire neurons, or an edge between this pair in our input graph. 
The strength of the connection is taken to be the number of synapses between this pair of neurons, which is given in the 'syn_count' column.
Columns in this dataframe:
• pre_root_id: root_id of the presynaptic neuron
• post_root_id: root_id of the postsynaptic neuron
• neuropil: brain region
• syn_count: count of synapses
• nt_type: neurotransmitter type

"""


task_description = """We are interested in neurons in the left lateral horn of the fly, a region implicated in innate odor behavior.
This region is labeled LH_L in the neuropil column of the connections dataset.

Write code that trains a Graph Neural Network (GNN) to predict links in the relevant neuropil.

The nodes of our graph input will be the neurons, and the edges will be the connections between them. We will also include the following attributes:

• Node attributes: The cell type of each neuron (represented as a one-hot encoding)
• Edge attributes: The strength of connectivity (synaptic count) of each connection

Use the networkx package and PyTorch Geometric (PyG) libarary for implementation. 

Use Graph Convolution layers for the network.

Use a train-test split with test size 0.2

Use up to 100 epochs of training.

At each epoch, record the training loss, and evaluate the model on the test set using ROC AUC score.

Save the trained model in the working directory as trained-GNN.pt

Save the evaluation results in the working directory as results.csv
"""


code_instruct = """\nProduce ONLY executable code in python. 
Provide properly denoted comments within the code, but DO NOT give any free-text explanations before or after."""


class Agent:
    def __init__(self, agentype):
        self.agentype = agentype
        if agentype == "gemini":
            GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY1')
            genai.configure(api_key=GOOGLE_API_KEY)
            # gemini_solver.list_models()

            safesettings = {"HARM_CATEGORY_HARASSMENT": "block_none",
                            "HARM_CATEGORY_DANGEROUS": "block_none",
                            "HARM_CATEGORY_HATE_SPEECH": "block_none",
                            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "block_none"}

            modelname = "gemini-1.5-flash-latest"
            self.model = genai.GenerativeModel(modelname, safety_settings=safesettings)
        else:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.agentype,
                                                              device_map="auto",
                                                              attn_implementation="flash_attention_2",
                                                              quantization_config=quantization_config).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.agentype)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, p):
        if self.agentype == "gemini":
            response = self.model.generate_content(p).text
        else:
            model_input = self.tokenizer([p], return_tensors="pt").to(device)
            generated_ids = self.model.generate(model_input.input_ids,
                                                pad_token_id=self.tokenizer.eos_token_id,
                                                max_new_tokens=2000)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_input.input_ids, generated_ids)]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        parsed = response.split("python")
        if len(parsed) > 1:
            response = parsed[1][:-3]
        return response


if __name__ == '__main__':

    assistant = Agent("gemini")
    """

    prompt = "Data description: " + data_description + "\n\nTask description: " + task_description + code_instruct

    success = False

    for i in range(3):
        print("\nTrial ", i+1)
        code = assistant.generate(prompt)
        print("\nCode: \n", code)
        try:
            print("\nExecuting")
            exec(code)
        except Exception as error:
            print("\nError: ", error)
            prompt = f"\n\nCode: {code}\n\nError message: {error}\n\nFix the error in the python code to complete the task." + code_instruct
            code = assistant.generate(prompt)
        else:
            if os.path.exists("results.csv"):
                print("GNN model trained")
                success = True
                # save agent code
                with open("LLM-code.txt", "w") as f:
                    f.write(code)
                break
            else:
                error = "results.csv file not found"
                print("\nError: ", error)
                prompt = f"\n\nCode: {code}\n\nError message: {error}\n\nFix the error in the python code to complete the task." + code_instruct
    """
    success = True
    
    if success is True:
        df = pd.read_csv("results.csv")
        cols = df.columns
        results = []
        for col in cols:
            results.append(col + ": ")
            line = df["col"].astype(str).tolist()
            results.append("; ".join(line))
        results = "\n".join(results)
        prompt = "This was the task given: " + task_description
        prompt += "This was the code produced: \n\n" + code
        prompt += "\n\nHere are the results of the evaluated GNN model: \n\n" + results
        prompt += "\n\nWrite a report summarizing the results."
        report = assistant.generate(prompt)
        print(report)
        with open("LLM-code.txt", "w") as f:
            f.write(code)
    else:
        print("failed")

