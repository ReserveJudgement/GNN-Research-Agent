import connectome_tools
from connectome_tools.connectome_loaders import load_flywire
# GNN
import torch
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx

# Data handling
import networkx as nx
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt

# Link prediction model - code adapted from https://gist.github.com/tomonori-masui/144c2057a64ec892a0a88066607eb3d2#file-link_predictor-py
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling

import torch_geometric.transforms as T


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels) # Graph convolution layer 1
        self.conv2 = GCNConv(hidden_channels, out_channels) # Graph convolution layer 2

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(
            dim=-1
        )  # product of a pair of nodes on each edge

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

def train_link_predictor(
    model, train_data, val_data, optimizer, criterion, n_epochs=300
):

    train_losses = []
    val_losses = []
    for epoch in range(1, n_epochs + 1):

        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)

        # sampling training negatives for every training epoch
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()

        val_auc, _, _ = eval_link_predictor(model, val_data)

        train_losses.append(loss.detach().numpy())
        val_losses.append(val_auc)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d}")

    return model, train_losses, val_losses


@torch.no_grad()
def eval_link_predictor(model, data):

    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()

    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy()), data.edge_label.cpu().numpy(), out.cpu().numpy()




if __name__ == '__main__':
    data_path = "data/"
    neurons, J, J_nts = load_flywire(data_path, by_nts=True, include_spatial=False)

    # Import FlyWire connections as an edgelist
    connections = pd.read_csv(f'{data_path}/connections.csv')

    # Restrict dataset to a specific neuropil
    neuropil_of_interest = 'LH_L'
    connections = connections[
        connections['neuropil'] == neuropil_of_interest]  # Filter connections in edgelist by neuropil of interest

    lh_neurons = list(np.unique(list(connections['pre_root_id']) + list(
        connections['post_root_id'])))  # Subselect neurons in neuropil of interest
    neurons = neurons[neurons['root_id'].isin(lh_neurons)]

    print('Number of neurons in ' + neuropil_of_interest + ': ' + str(len(neurons)))

    print(f'Baseline: {len(connections.index)/(len(neurons)*(len(neurons) - 1)/2)}')

    all_neurons = list(neurons['root_id'])  # List of neurons in graph
    all_neuron_types = list(neurons['cell_type'])  # List of types for each neuron in graph
    all_nt_types = list(neurons['nt_type'])

    # %%
    all_unique_types = np.array((neurons['cell_type'].unique()))  # List of unique neuron types in graph
    all_unique_nts = np.array((neurons['nt_type'].unique()))

    # Node attributes: The cell type of each neuron (this will be represented as a one-hot encoding)
    # (num_nodes, num_feats)
    # predict using cell type
    #node_types = np.zeros((len(all_neurons), len(all_unique_types)))
    # predict using neurotransmitter type
    node_types = np.zeros((len(all_neurons), len(all_unique_nts)))
    for i_neuron, neuron in enumerate(all_neurons):
        #i_type = np.where(all_unique_types == all_neuron_types[i_neuron])
        i_type = np.where(all_unique_nts == all_nt_types[i_neuron])
        assert len(i_type) == 1
        node_types[i_neuron, i_type] = 1

    # Edge attributes: The strength of connectivity (synaptic count) of each connection
    connections = connections.astype({"syn_count": 'float32'})

    # networkx graph -> PyTorch geometric object transformation is easy
    #G = nx.DiGraph()  # Initialize a networkx directed graph
    G = nx.from_pandas_edgelist(connections, 'pre_root_id', 'post_root_id', ['syn_count'],
                                create_using=nx.DiGraph())  # Create a graph with our node and edge information

    # Convert graph into PyTorch geometric object
    graph = from_networkx(G, group_edge_attrs=['syn_count'])  # Add edge attributes from networkx graph

    graph.x = torch.from_numpy(node_types).float()  # Add node attributes (one-hot encoded cell type)

    split = T.RandomLinkSplit(
        num_val=0.05,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0,
    )
    train_data, val_data, test_data = split(graph)

    print('train data: ' + str(train_data))
    print('val data:' + str(val_data))
    print('test data:' + str(test_data))

    #model = Net(len(all_unique_neurons), 128, 64).to('cpu')
    model = Net(len(all_unique_nts), 128, 64).to('cpu')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    model, train_losses, val_losses = train_link_predictor(model, train_data, val_data, optimizer, criterion)

    test_auc, _, _ = eval_link_predictor(model, test_data)
    print(f"Final validation accuracy: {test_auc:.3f}")

    # Plot losses
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(train_losses, label='train loss')
    ax[1].plot(val_losses, label='validation accuracy (auc)')
    ax[0].set_title('Training loss')
    ax[1].set_title('Validation accuracy')

    ax[0].set_xlabel('Epoch')
    ax[1].set_xlabel('Epoch')

    ax[0].set_ylabel('Loss')
    ax[1].set_ylabel('AUC')

    plt.show()

    # Plot predicted links and actual edges in held out data
    _, actual_links, predicted_links, = eval_link_predictor(model, test_data)

    split = int(len(actual_links) / 2)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].hist(np.sort(predicted_links[split:]))
    ax[1].hist(np.sort(predicted_links[:split]))

    ax[0].set_title('Nonexistent edges')
    ax[1].set_title('Existent edges')

    ax[0].set_ylabel('Count')
    ax[1].set_ylabel('Count')

    ax[0].set_xlabel('Predicted link probabilities')
    ax[1].set_xlabel('Predicted link probabilities')

    plt.show()
