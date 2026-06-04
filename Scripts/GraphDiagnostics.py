import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import os
import networkx as nx

# Change the working directory
os.chdir('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Scripts/')

# Step 1: Load nodes and edges CSV
nodes = pd.read_csv('../CuratedData/protein_node_list_PAD.csv')
edges = pd.read_csv('../CuratedData/FilteredProteinEdges.csv')

# Step 2: Create a graph using NetworkX
G = nx.Graph()

# Add nodes to the graph
G.add_nodes_from(nodes['NodeIndex'].values)  # Adjust based on the actual node column name

# Add edges to the graph
for index, row in edges.iterrows():
    G.add_edge(row['Index1'], row['Index2'])  # Adjust based on your edge column names

# Step 3: Check for disconnected components
connected_components = list(nx.connected_components(G))
num_components = len(connected_components)

# Print the results
if num_components > 1:
    print(f"The graph has {num_components} disconnected components.")
    print("Sizes of each component:", [len(comp) for comp in connected_components])
else:
    print("The graph is fully connected.")

# Step 4: Optionally, you can extract and view the disconnected components
for i, component in enumerate(connected_components):
    print(f"Component {i + 1} size: {len(component)} nodes")
    print(component)  # List of nodes in each disconnected component

# Find the largest connected component
largest_component = max(nx.connected_components(G), key=len)

# Filter nodes for the largest component
filtered_nodes = nodes[nodes['NodeIndex'].isin(largest_component)]
# Update the NodeIndex to be sequential starting from 1
filtered_nodes.loc[:, 'NodeIndex'] = range(0, len(filtered_nodes))
# Create a mapping from old node indices to new indices
node_mapping = {old_idx: new_idx for old_idx, new_idx in zip(
    nodes[nodes['NodeIndex'].isin(largest_component)]['NodeIndex'],
    filtered_nodes['NodeIndex']
)}

# Filter edges for the largest component and reindex

filtered_edges = edges[edges['Index1'].isin(largest_component) & edges['Index2'].isin(largest_component)].copy()

filtered_edges['Index1'] = filtered_edges['Index1'].map(node_mapping)
filtered_edges['Index2'] = filtered_edges['Index2'].map(node_mapping)
# Remove edges with NaN values after mapping
#filtered_edges = filtered_edges.dropna(subset=['Index1', 'Index2'])
# Save the filtered nodes and edges
filtered_nodes.to_csv('../CuratedData/LargestComponents/Nodes_with_properties_FC.csv', index=False)
filtered_edges.to_csv('../CuratedData/LargestComponents/Edges_with_nodeproperties_FC.csv', index=False)
#
G = nx.Graph()
#
# Add nodes to the graph
G.add_nodes_from(filtered_nodes['NodeIndex'].values)  # Adjust based on the actual node column name
#
# Add edges to the graph
for index, row in filtered_edges.iterrows():
    G.add_edge(row['Index1'], row['Index2'])  # Adjust based on your edge column names

# Step 3: Check for disconnected components
connected_components = list(nx.connected_components(G))
num_components = len(connected_components)

# Print the results
if num_components > 1:
    print(f"The graph has {num_components} disconnected components.")
    print("Sizes of each component:", [len(comp) for comp in connected_components])
else:
    print("The graph is fully connected.")