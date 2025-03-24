"""
A library of functions used in the seventh tutorial of the course:

T7 From Graphs to Similarity (ed. 24 - 25)

Contains two parts:

1. Functions in general
- TODO: Add list

2. Functions pertaining to the graph-based neural network
- TODO: Add list (necessary?)
"""

## Imports

import pickle
import random
from tqdm import tqdm
import copy

import matplotlib.pyplot as plt

import numpy as np

import networkx as nx
from shapely.geometry import Polygon

import omegaconf

import torch
from torch_geometric.utils import from_networkx

from umap import UMAP

## Constants

room_colors = [
    '#e6550d',  # living room
    '#1f77b4',  # bedroom
    '#fd8d3c',  # kitchen
    '#6b6ecf',  # bathroom
    '#fdae6b',  # dining
    '#5254a3',  # store room
    '#2ca02c',  # balcony
    '#fdd0a2'   # corridor
]


## Functions

# ----------------------------------------------------------------------
# --- Utilities: Drawing, UMAP, Embeddings, Loading and saving, etc. ---
# ----------------------------------------------------------------------

def load_pickle(filename):
    """
    Loads a pickled file.
    """
    with open(filename, 'rb') as f:
        object = pickle.load(f)
        f.close()
    return object


def draw_polygon(ax, poly, label=None, **kwargs):
    """Plots a polygon by filling it up. Edges of shapes are avoided to show exactly the area that
    the elements occupy."""
    x, y = poly.exterior.xy
    ax.fill(x, y, label=label, **kwargs)
    return


def draw_rooms(ax, polygons, colors, lw=None):
    """Draws the rooms of the floor plan layout."""

    # Simultaneously extract geometries and categories
    # And directly plot them int the correct color
    for poly, color in zip(polygons, colors):
        draw_polygon(ax, poly, facecolor=color, edgecolor='white', linewidth=lw)


def draw_graph(ax, G, fs, lw=0, s=20, w=2, 
               node_color='black', edge_colors=['black', 'white'], 
               viz_rooms=True,
               polygons = None,
               pos = None):

    # Extract information
    if polygons is None: 
        polygons = [Polygon(d) for _, d in G.nodes('polygon')]
    colors = [room_colors[d] for _, d in G.nodes('category')]
    if pos is None: 
        pos = {n: np.array(
            [Polygon(d).representative_point().x,
            Polygon(d).representative_point().y])
            for n, d in G.nodes('polygon')}

    # Draw room shapes
    if viz_rooms:
        draw_rooms(ax, polygons, colors, lw=lw)

    # Draw nodes
    if isinstance(s, list):
        nx.draw_networkx_nodes(G, pos, node_size=s, node_color=node_color, ax=ax)
    else:
        nx.draw_networkx_nodes(G, pos, node_size=s, node_color=node_color, ax=ax)

    # Draw door edges
    edges = [(u, v) for u, v, d in G.edges(data="connectivity") if d == 1]
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors[0],
                           width=w, ax=ax)

    # Draw door edges
    edges = [(u, v) for u, v, d in G.edges(data="connectivity") if d == 0]
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors[1],
                           width=w, ax=ax)


def remove_attributes_from_graph(graph, list_attr):
    """
    Removes attributes from graph.
    :param graph: Input topological graph.
    :param list_attr: Attributes to-be removed.
    :return: Output topological graph with removed attributes.
    """

    for attr in list_attr:
        for n in graph.nodes(): # delete irrelevant nodes
            del graph.nodes[n][attr]
    return graph


def prepare_data(data, device='cpu'):
    """Prepares the graph data from DataBatch of the PyG Dataloader"""

    edge_index = data['edge_index'].to(device)
    x_geom = data['geometry'].float().to(device)
    x_cats = data['category'].long().to(device)
    edge_feats = data['connectivity'].long().to(device)
    batch = torch.zeros(x_geom.size()[0], dtype=torch.int64).to(device)

    return edge_index, x_geom, x_cats, edge_feats, batch


def get_embeddings(graphs, model, device='cpu'):

    # set model to eval mode; no randomness
    model.eval()

    # initialize list of names and embeddings
    names, embeddings = [], []

    for graph in tqdm(graphs, total=len(graphs)):

        # Copy graph and get ID
        G = copy.deepcopy(graph)
        id = G.graph["ID"]

        # Convert to PyG graph
        G = remove_attributes_from_graph(G, ["polygon"])
        G = from_networkx(G)

        # Prepare the data
        edge_index, x_geom, x_cats, edge_feats, batch = prepare_data(G)

        # Feedforward to get graph-level feature vectors
        with torch.no_grad():
            _, graph_feats = model(edge_index, x_cats, x_geom, edge_feats, batch)

        # Append to list
        embeddings.append(graph_feats)
        names.append(id)

    # Concatenate embeddings into a vector
    embeddings = torch.cat(embeddings, dim=0)

    return names, embeddings


def normalize(mat):
    mat_n = mat - np.min(mat, axis=0)
    mat_n /= np.max(mat_n, axis=0)
    return mat_n


def get_umap_projections(rs, dim=2, norm=True, random_state=None):

    # Get projections (unnormalized)
    proj = UMAP(n_components=dim, random_state=random_state).fit_transform(rs)

    # Normalize if wanted
    if norm: proj = normalize(proj)

    return proj


def get_grid_embeddings(embeds_2d, names, w=60, h=60):

    # Initialize the grid
    x = np.linspace(1, w-1, w-1) / w
    y = np.linspace(1, h-1, h-1) / h
    xx, yy = np.meshgrid(x, y)
    coords = np.array((xx.ravel(), yy.ravel())).T

    # Find the nearest neighbor for every grid point:
    # - The nearest neighboring embedding will take the value of the grid point
    # - The nearest neighbor should be close enough to the poit (min_dist),
    # which means that not every grid point will have a floor plan associated with it.
    min_dist = 1 / (2*h)
    embeds_grid = []
    names_grid = []

    for xy in tqdm(coords, total=coords.shape[0]):

        cost = np.sqrt(np.sum(np.power(embeds_2d - xy, 2), axis=1))
        min_pos = cost.argmin()
        if cost[min_pos] < min_dist:
            embeds_grid.append(xy)
            names_grid.append(names[min_pos])
        else:
            continue

    embeds_grid = np.array(embeds_grid)

    return embeds_grid, names_grid


def draw_dataset(graphs, ids, embeds_grid, names_grid, w, fs=50, stop=-1):

    # Set sizing of the floor plans based on the grid and original sizes
    size = (1/w) * (1.1 / 2)

    # Initialize figure
    fig, ax = plt.subplots(1, 1, figsize=(fs, fs))
    ax.set_aspect('equal')

    # Ax set limits
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    # Make up spines
    ax.spines['bottom'].set_linewidth(fs/5)
    ax.spines['left'].set_linewidth(fs/5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Make up ticks
    ax.tick_params(axis='both', width=fs/5, labelsize=fs*2)

    # Make up labels
    _ = ax.set_ylabel("Hidden Feature 1", fontsize=fs*3)
    _ = ax.set_xlabel("Hidden Feature 2", fontsize=fs*3)

    for id, feat in tqdm(zip(names_grid[:stop], embeds_grid[:stop])):

        # Find G by indexing the name
        G = graphs[ids.index(id)]

        # Translate polygons and centers based on the embeddings
        polygons = [Polygon(np.array(d) * size - size / 2 + feat) for _, d in G.nodes('polygon')]
        pos = {n: np.array(
            [Polygon(d).representative_point().x,
            Polygon(d).representative_point().y]) * size - size / 2 + feat
            for n, d in G.nodes('polygon')}

        # Draw floor plan and graph
        draw_graph(ax, G, polygons=polygons, pos=pos, fs=fs, s=fs/6, w=fs/80, lw=fs/100)

# -----------------------------------------------------------------------
# --- Graph embedding network (GEN) and graph matching networks (GMN) ---
# -----------------------------------------------------------------------

import torch
from torch import nn
from torch_geometric.nn import MessagePassing, BatchNorm
from torch_scatter import scatter_mean

def mlp(feat_dim):
    "Outputs a multi-layer perceptron of various depth and size, with ReLU activation."
    layer = []
    for i in range(len(feat_dim) - 1):
        layer.append(nn.Linear(feat_dim[i], feat_dim[i + 1]))
        layer.append(nn.ReLU())
    return nn.Sequential(*layer)


def embed_layer(vocab_size, dim, drop=0.5):
    return nn.Sequential(nn.Embedding(vocab_size, dim),
                         nn.ReLU(),
                         nn.Dropout(drop))


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    """Computes pairwise similarity between two vectors x and y"""
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def cross_attention(x, y, sim=cosine_distance_torch):
    """Computes attention between x an y, and vice versa"""
    a = sim(x, y)
    a_x = torch.softmax(a, dim=1)  # i->j
    a_y = torch.softmax(a, dim=0)  # j->i
    attention_x = torch.mm(a_x, y)
    attention_y = torch.mm(torch.transpose(a_y, 1, 0), x)
    return attention_x, attention_y


def batch_pair_cross_attention(feats, batch, **kwargs):
    """Computes the cross graph attention between pairs of graph for a whole batch."""

    # find number of blocks = number of individual graphs in batch
    n_blocks = torch.unique(batch).size()[0]

    # create partitions
    block_feats = []
    for block in range(n_blocks):
        block_feats.append(feats[batch == block, :])

    # loop over all block pairs
    outs = []
    for i in range(0, n_blocks, 2):
        x = block_feats[i]
        y = block_feats[i + 1]
        attention_x, attention_y = cross_attention(x, y, **kwargs)
        outs.append(attention_x)
        outs.append(attention_y)
    results = torch.cat(outs, dim=0)

    return results


def init_weights(m, gain=1.0, bias=0.01):
    """Initializes weights of a learnable layer."""

    # Linear
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        m.bias.data.fill_(bias)

    # GRU
    if isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


class GraphEncoder(nn.Module):
    """Encoder module that projects node and edge features to learnable embeddings."""

    def __init__(self, cfg):  # 1 for binary edge attributes (*eg* presence of door or not)
        super(GraphEncoder, self).__init__()


        # Node and edge encoders are MLPs

        self.cats_one_hot = embed_layer(cfg.cats_dim, cfg.node_dim)
        self.geom_encoder = mlp([cfg.geom_dim, cfg.node_dim])
        self.node_encoder = mlp([2 * cfg.node_dim, cfg.node_dim])

        if cfg.edge_type == "door":
            self.edge_encoder = embed_layer(cfg.inter_geom_dim, cfg.edge_dim)
        else:
            self.edge_encoder = mlp([cfg.inter_geom_dim, cfg.edge_dim])

        # Initialize weights
        init_weights(self.geom_encoder)
        init_weights(self.node_encoder)
        init_weights(self.edge_encoder)

    # Forward method:
    # Room graphs: x1 = geometry, x2 = category
    # GED: x1 = ones
    def forward(self, x1, x2, edge_feat):

        x1 = self.cats_one_hot(x1)  # one-hot categorical encoding // 5  -> D_v
        x2 = self.geom_encoder(x2)  # geometry encoding // 12 -> D_v
        node_encod = torch.cat((x2, x1.squeeze(1)), -1)  # stack // D_v x D_v -> 2*Dv
        node_encod = self.node_encoder(node_encod)  # full node embedding // 2*D_v -> D_v

        # edge embedding
        edge_encod = self.edge_encoder(edge_feat)  # edge encoding // 8 -> D_e

        return node_encod, edge_encod  # D_v, D_e


# Graph (matching) convolutional layer
class GConv(MessagePassing):
    """Propagation layer for a graph convolutional or matching network."""

    def __init__(self, cfg):
        super(GConv, self).__init__(aggr=cfg.aggr)

        # Hyper settings
        self.matching = cfg.matching

        # Message passing: MLP
        self.f_message = torch.nn.Linear(cfg.node_dim*2+cfg.edge_dim, cfg.node_dim)

        # Node update: GRU
        if cfg.matching:
            # A GMN layer takes as input the intra- and inter-graph messages, by concatenation
            self.f_node = torch.nn.GRU(cfg.node_dim*2, cfg.node_dim)
        else:
            # A GEN layer takes as input the inter-graph messages only
             self.f_node = torch.nn.GRU(cfg.node_dim, cfg.node_dim)

        # Batchnorm
        self.batch_norm = BatchNorm(cfg.node_dim)

        # Initialize
        init_weights(self.f_message, gain=cfg.message_gain)  # default: small gain for message apparatus
        init_weights(self.f_node)

    def forward(self, edge_index, x, edge_attr, batch):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, original_x=x, batch=batch)

    # Message on concatenation of 1) itself, 2) aggr. neighbors, and 3) aggr. edges
    def message(self, x_i, x_j, edge_attr):
        x = torch.cat([x_i, x_j, edge_attr], dim=1)
        x = self.f_message(x)
        return x

    # Node update: GRU([])
    def update(self, aggr_out, original_x, batch):

        # Find node inputs
        if self.matching:
            # Cross-graph messages
            cross_attention = batch_pair_cross_attention(original_x, batch)
            attention_input = original_x - cross_attention
            # Concatenate intra- with inter-graph messages
            node_input = torch.cat([aggr_out, attention_input], dim=1)
        else:
            node_input = aggr_out

        # Node update
        _, out = self.f_node(node_input.unsqueeze(0), original_x.unsqueeze(0))
        out = out.squeeze(0)

        # Batch norm
        out = self.batch_norm(out)
        return out

# Graph aggregation / readout function(s)
class GraphAggregator(torch.nn.Module):
    """Computes the graph-level embedding from the final node-level embeddings."""

    def __init__(self, cfg):
        super(GraphAggregator, self).__init__()
        self.lin = torch.nn.Linear(cfg.node_dim, cfg.graph_dim)
        self.lin_gate = torch.nn.Linear(cfg.node_dim, cfg.graph_dim)
        self.lin_final = torch.nn.Linear(cfg.graph_dim, cfg.graph_dim)

    def forward(self, x, batch):
        x_states = self.lin(x)  # node states // [V x D_v] -> [V x D_F]
        x_gates = torch.nn.functional.softmax(self.lin_gate(x), dim=1)  # node gates // [N_v x D_v] -> [N_v x D_F]
        x_states = x_states * x_gates  # update states based on gate "gated states" // [N_v x D_g]
        x_states = scatter_mean(x_states, batch, dim=0)  # graph-level feature vectors // [N_v x D_g] -> [N_g x D_g]
        x_states = self.lin_final(x_states)  # final graph-level embedding // [N_g x D_g] -> [N_g x D_g]
        return x_states


# Graph Siamese network
class GraphSiameseNetwork(torch.nn.Module):
    """Graph embedding network."""

    def __init__(self, cfg):
        super(GraphSiameseNetwork, self).__init__()

        # Graph encoder
        self.encoder = GraphEncoder(cfg)

        # Graph propagation layers
        self.prop_layers = torch.nn.ModuleList()
        for _ in range(cfg.num_layers):
            self.prop_layers.append(GConv(cfg))

        # Graph aggregation layer
        self.aggregation = GraphAggregator(cfg)

    def forward(self, edge_index, x1, x2, edge_feats, batch):

        # Graph encoder
        node_feats, edge_feats = self.encoder(x1, x2, edge_feats)

        # Graph propagation
        for i in range(len(self.prop_layers)):
            node_feats = self.prop_layers[i](edge_index, node_feats, edge_feats, batch)

        # Graph aggregation
        graph_feats = self.aggregation(node_feats, batch)

        return node_feats, graph_feats
