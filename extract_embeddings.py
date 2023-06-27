import pickle
import networkx as nx
from karateclub.node_embedding.neighbourhood import DeepWalk, Node2Vec, GLEE
from karateclub.graph_embedding import Graph2Vec, WaveletCharacteristic
from sklearn.ensemble import RandomForestClassifier


def get_embeddings():

    # Load the graph
    graph = pickle.load(open("GDA_graph.pickle", "rb"))
    graph = nx.convert_node_labels_to_integers(graph, label_attribute="revert")

    # #Plotting Graph
    # nx.draw(H, with_labels=True, node_color=colors, node_size=sizes)

    # Init embedding method
    # model = DeepWalk()
    # model = Node2Vec()
    # model = GLEE()
    # # model = WaveletCharacteristic()

    # # Fit embedding model to the graph
    # model.fit(graph)
    # # Get the embedding vectors
    # embeddings = model.get_embedding()
    # print(f"Embeddings shape: {embeddings.shape}")
    return graph

if __name__ == "__main__":
    _, _ = get_embeddings()
