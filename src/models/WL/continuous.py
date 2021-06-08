import torch

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree



class WL(MessagePassing):

	def __init__(self, in_channels=1):
		super(WL, self).__init__(aggr='add')
		# self.lin = torch.nn.Linear(in_channels, out_channels)

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		# x has shape [N, in_channels], edge_index has shape [2, E]

		# Add self-loops to the adjacency matrix
		edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
		# edge_index has shape [2, E + N] (added at the end of the edge indices)

		# # Linearly transform node feature matrix
		# x = self.lin(x)

		# Compute normalization
		row, col = edge_index
		deg = degree(col, x.size(0), dtype=x.dtype)
		# deg_inv_sqrt = deg.pow(-0.5)
		# norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
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
