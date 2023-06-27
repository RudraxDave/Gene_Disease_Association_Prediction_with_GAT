from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
import torch
from extract_embeddings import get_embeddings
from data_extraction import get_pyg
from torch_geometric.utils.convert import from_networkx, to_networkx


SEED = 42
class GCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super().__init__()
        torch.manual_seed(SEED)
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_channels)


    def forward(self, data):
        edge_index = data['edge_index']
        x = data['x']
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super().__init__()
        torch.manual_seed(SEED)
        self.conv1 = GATConv(input_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, output_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    # out = model(data.x, data.edge_index)
    loss = torch.nn.BCEWithLogitsLoss(out, data['edge_index'].y)
    loss.backward()
    optimizer.step()
    return loss
def process_data():
    # raw_graph = get_embeddings()
    data = get_pyg()

    model = GCN(input_channels=5, hidden_channels=16, output_channels=1)

    # model = GAT(input_channels=embeddings.shape[1], hidden_channels=16, output_channels=1)
    # print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 11):
        loss = train(model, data, optimizer)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")


if __name__ == "__main__":
    process_data()