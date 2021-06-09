import torch

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WL(MessagePassing):

	def __init__(self, normalization='wasserstein'):
		super(WL, self).__init__(aggr='add')
		self.normalization = normalization

	def forward(self, x, edge_index):
		# x has shape [N, in_channels], edge_index has shape [2, E]
		# Add self-loops to the adjacency matrix
		edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
		# edge_index has shape [2, E + N] (added at the end of the edge indices)
		# Compute normalization
		row, col = edge_index
		deg = degree(col, x.size(0), dtype=x.dtype)
		if self.normalization == 'GCN':
			deg_inv_sqrt = deg.pow(-0.5)
			norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
		else: # elif self.normalization == 'wasserstein'
			deg_inv = deg.pow(-1)
			norm = deg_inv[col]
			norm[-x.size(0):] = torch.ones(x.size(0), dtype=x.dtype)
			norm = 0.5 * norm
		# Start propagating messages
		return self.propagate(edge_index, x=x, norm=norm)

	def message(self, x_j, norm):
		# x_j has shape [E, out_channels]
		# Normalize node features.
		return norm.view(-1, 1) * x_j


def computeNodeRepresentations(G, depth=3, normalization='wasserstein'):
	'''
	Compute the WL kernel nodde representations for the continuous attributed case.

	Parameters:
		- G: (torch_geometric.Data) Graph in torch geometric format.
		- depth: (int) Number of aggregation steps to perform.
		- normalization: (str) Type of normalization to use in the aggregation step.

	Returns:
		- (np.ndarray) Node representations for every node in the input graph.
	'''
	model = WL(normalization=normalization).to(device)
	G = G.to(device)
	x, edge_index = G.x, G.edge_index
	for _ in range(depth):
		x = model(x, edge_index)
	return x.detach().numpy()
