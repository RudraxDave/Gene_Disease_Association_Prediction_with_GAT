# -*- coding: utf-8 -*-
"""EE638 project.ipynb
"""

import pickle
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from karateclub.graph_embedding import Graph2Vec, WaveletCharacteristic
from karateclub.node_embedding.neighbourhood import DeepWalk, Node2Vec, GLEE
import pandas as pd
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils.convert import to_networkx

"""# Gene Node Features"""

def create_graph():

  training_data = pd.read_csv('GDA_associations.csv')
  training_data = training_data.dropna()
  gene_data = training_data[['geneid', 'gene_dsi', 'gene_dpi', 'gene_pli', 'protein_class']].drop_duplicates()

  protein_class_cat_codes, protein_class_category_map = pd.factorize(gene_data['protein_class'])
  gene_data['protein_class'] = protein_class_cat_codes

  print(f"Number of rows with empty values in gene_dsi:\n{gene_data.isnull().sum()}")
  # Deprecated!
  # gene_data = gene_data.fillna(0)

  gene_node_attr = gene_data.set_index('geneid').to_dict('index')

  """# Disease Node Feature Vector"""

  disease_class_list = ['C01', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'F01', 'F02', 'F03']

  disease_data = training_data[['diseaseid',	'disease_class',	'disease_type',	'disease_semantic_type']].drop_duplicates()
  disease_data = disease_data.reset_index(drop=True)

  # disease_id_cat_codes, disease_id_category_map = pd.factorize(disease_data['diseaseid'])
  # disease_data['diseaseid'] = disease_id_cat_codes

  disease_type_cat_codes, disease_type_category_map = pd.factorize(disease_data['disease_type'])
  disease_data['disease_type'] = disease_type_cat_codes

  disease_semantic_type_cat_codes, disease_semantic_type_category_map = pd.factorize(disease_data['disease_semantic_type'])
  disease_data['disease_semantic_type'] = disease_semantic_type_cat_codes


  for each_class in disease_class_list:
    disease_data[each_class] = 0


  for idx, row in disease_data.iloc[:-1].iterrows():
    try:
      list_of_disease_classes = row['disease_class'].split(';')
      for each_class in list_of_disease_classes:
        disease_data.at[idx, each_class] = 1
    except Exception as e:
      print(f"Issue while processing {idx}th row with disease class {row['disease_class']}")
      print(e)

  disease_data = disease_data.drop(['disease_class'], axis = 1)
  disease_node_attr = disease_data.set_index('diseaseid').to_dict('index')


  """# Graph"""
  # G = nx.Graph()
  # G = nx.from_pandas_edgelist(training_data, 'geneid', 'diseaseid', ['score'])
  # G = nx.convert_node_labels_to_integers(G, label_attribute="revert")

  Gu = nx.Graph()
  Gu = nx.from_pandas_edgelist(training_data, 'geneid', 'diseaseid', ['score'])
  Gu = nx.convert_node_labels_to_integers(Gu, label_attribute="revert")

  # TODO: Add PPI and other info as node attributes
  nx.set_node_attributes(Gu, gene_node_attr)
  nx.set_node_attributes(Gu, disease_node_attr)
  print(f"Number of nodes in the graph: {Gu.number_of_nodes()}")

  ## Plotting Graph - issue with axial stack in numpy
  # nx.draw(Gu, with_labels=True, node_color='skyblue', node_size=1500)

  # G.nodes[1] To access the gene node
  # G.nodes['C0036341'] To access the disease node
  # G[1]['C0036341'] To access the edge of the graph

  # #Plotting Graph
  # nx.draw(H, with_labels=True, node_color=colors, node_size=sizes)

  # save graph object to file - already dumped
  # pickle.dump(Gu, open('GDA_graph.pickle', 'wb'))
  return Gu

def get_pyg():

  training_data = pd.read_csv('GDA_associations.csv').dropna()

  offset = training_data["geneid"].nunique()
  unique_targets = training_data["diseaseid"].unique()
  target_to_id = {t: i + offset for i, t in enumerate(unique_targets)}

  # Replace the alphanumeric targets with their integer mappings
  training_data["diseaseid"] = training_data["diseaseid"].apply(lambda x: target_to_id[x])

  # Convert other features to numeric values
  diseaseclass_cat_codes, diseaseclass_category_map = pd.factorize(training_data['disease_class'])
  training_data['disease_class'] = diseaseclass_cat_codes

  diseaseclass_cat_codes, diseaseclass_category_map = pd.factorize(training_data['protein_class'])
  training_data['protein_class'] = diseaseclass_cat_codes

  diseaseclass_cat_codes, diseaseclass_category_map = pd.factorize(training_data['disease_semantic_type'])
  training_data['disease_semantic_type'] = diseaseclass_cat_codes

  diseaseclass_cat_codes, diseaseclass_category_map = pd.factorize(training_data['disease_type'])
  training_data['disease_type'] = diseaseclass_cat_codes

  # create feature matrices for genes and diseases
  gene_features = ['gene_dsi', 'gene_dpi', 'protein_class']
  gene_x = training_data[gene_features].values
  gene_x = torch.tensor(gene_x, dtype=torch.float)

  disease_features = ['disease_type', 'disease_semantic_type']
  disease_x = training_data[disease_features].values
  disease_x = torch.tensor(disease_x, dtype=torch.float)

  # create edge_index and y (labels)
  edge_index = torch.tensor(training_data[['geneid', 'diseaseid']].values.T)
  # Create the labels 'Y'
  y = torch.tensor(training_data['score'].values, dtype=torch.float)

  # create HeteroData object
  data = HeteroData()

  # add nodes and features to HeteroData object
  data['geneid'].x = gene_x
  data['diseaseid'].x = disease_x

  # add edges and labels to HeteroData object
  data['edge_index'] = edge_index
  data['edge_index'].edge_type = torch.tensor([0], dtype=torch.long)  # edge type between gene and disease nodes
  data['edge_index'].y = y

  # Combine the gene and disease features into a single feature matrix
  X = torch.cat([gene_x, disease_x], dim=1)
  data['x'] = X

  # # Generate embeddings using KarateClub
  # model = Graph2Vec(dimensions=128)
  # model.fit(data)
  # embeddings = model.get_embedding()
  # num_node_features = embeddings.shape[1]

  # # create PyTorch Geometric Data object
  # pyg_graph = Data(edge_index=edge_index, y=y)
  # num_nodes = int(torch.max(pyg_graph.edge_index)) + 1
  #
  # pyg_graph.x = X
  # pyg_graph.y = y
  # pyg_graph.num_nodes = num_nodes
  # pyg_graph.num_node_features = X.shape[1]
  # pyg_graph.num_edge_features = 1

  return data


  ### Target = Disease, Source = Gene
  # # Extract the edge index and edge attributes from the DataFrame
  # offset = training_data["geneid"].nunique()
  # unique_targets = training_data["diseaseid"].unique()
  # target_to_id = {t: i + offset for i, t in enumerate(unique_targets)}

  # # Replace the alphanumeric targets with their integer mappings
  # training_data["diseaseid"] = training_data["diseaseid"].apply(lambda x: target_to_id[x])
  # edge_index = torch.tensor(training_data[["geneid", "diseaseid"]].values).t().contiguous()
  # edge_attr = torch.tensor(training_data["score"].values, dtype=torch.float).view(-1, 1)

  # # Determine the number of nodes
  # num_nodes = edge_index.max().item() + 1

  # print(f"Number of unique genes: {offset}")
  # print(f"Number of unique diseases: {len(unique_targets)}")
  #
  # # Create a PyTorch Geometric Data object
  # graph = Data(edge_index=edge_index,
  #              edge_attr=edge_attr,
  #              num_nodes=num_nodes,
  #              num_classes=unique_targets.shape[0],
  #              node_attr=training_data[['gene_dsi', 'gene_dpi', 'gene_pli', 'protein_class']].drop_duplicates(),
  #              y=training_data['diseaseid'].drop_duplicates(),
  #              x = training_data['geneid'].drop_duplicates()
  #              )

  return graph
