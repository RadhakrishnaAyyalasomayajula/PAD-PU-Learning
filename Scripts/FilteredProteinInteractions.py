import pandas as pd
import numpy as np
import os

# Change the working directory
os.chdir('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/PAD/Scripts/')

import pandas as pd

# Load the CSV file into a pandas DataFrame
file_path = "../data/interactions_with_HGNC_symbols.csv"  # Replace with your file path
df = pd.read_csv(file_path)
#
#
# Filter the DataFrame where 'association_score' is greater than 750
filtered_df = df[df['score'] > 000]
print(filtered_df.shape[0])
#
# Extract proteins from the filtered interactions
protein1_list = filtered_df['protein1'].tolist()  # List of proteins from column 'protein1'
protein2_list = filtered_df['protein2'].tolist()  # List of proteins from column 'protein2'
#
# Combine both lists and get the unique proteins
all_proteins = set(protein1_list + protein2_list)
#
# Convert the set to a list if you need it as a list
all_proteins_list = list(all_proteins)
output_df = pd.DataFrame({
    'Index': range(len(all_proteins_list)),  # Create an index starting from 0
    'GeneSymbol': all_proteins_list
})

# Save the DataFrame to a CSV file
#output_df.to_csv('../CuratedData/FilteredProteinNodes.csv', index=False)
#
# Load the previously created protein list with indices
proteins_df = output_df

# Create a mapping from GeneSymbol to Index
protein_to_index = dict(zip(proteins_df['GeneSymbol'], proteins_df['Index']))

# Initialize lists for storing the edge data
index1_list = []
index2_list = []
protein1_list = []
protein2_list = []
score_list = []
interaction_type_list = []

# Iterate over the filtered interactions
for _, row in filtered_df.iterrows():
    protein1 = row['protein1']
    protein2 = row['protein2']
    score = row['score']

    # Get the index of protein1 and protein2 from the mapping
    if protein1 in protein_to_index and protein2 in protein_to_index:
        index1 = protein_to_index[protein1]
        index2 = protein_to_index[protein2]

        # Append the data to the lists
        index1_list.append(index1)
        index2_list.append(index2)
        protein1_list.append(protein1)
        protein2_list.append(protein2)
        score_list.append(score)
        interaction_type_list.append('Protein_Protein')

# Create a DataFrame for the edges
edges_df = pd.DataFrame({
    'Index1': index1_list,
    'Index2': index2_list,
    'Protein1': protein1_list,
    'Protein2': protein2_list,
    'Score': score_list,
    'InteractionType': interaction_type_list
})
print(edges_df.shape[0])
# Save the edges DataFrame to a new CSV file
#edges_df.to_csv('../CuratedData/FilteredProteinEdges.csv', index=False)
#
#
diseases = ['AAA', 'PAD', 'AMI', 'Stroke', 'HeartFailure']
for disease in diseases:
    if disease not in proteins_df.columns:
        proteins_df[disease] = 0  # Initialize with 0s

# Define a dictionary of disease files and their associated disease names
disease_files = {
    'AAA': '../data/AAA_proteins.csv',
    'PAD': '../data/PAD_proteins.csv',
    'AMI': '../data/AMI_proteins.csv',
    'Stroke': '../data/Stroke_proteins.csv',
    'HeartFailure': '../data/HF_proteins.csv'
}

# Iterate over each disease file
for disease, disease_file in disease_files.items():
    # Load the disease file
    disease_df = pd.read_csv(disease_file)

    # Check if the GeneSymbol column exists in the disease file
    if 'GeneSymbol' not in disease_df.columns:
        print(f"Warning: 'GeneSymbol' column not found in {disease_file}")
        continue

    # Get the list of proteins (GeneSymbols) for this disease
    disease_proteins = disease_df['GeneSymbol'].tolist()

    # Update the node DataFrame: set the disease column to 1 if the GeneSymbol is present
    proteins_df.loc[proteins_df['GeneSymbol'].isin(disease_proteins), disease] = 1

# Save the updated node list with disease associations
output_path = '../../AllProteins/Allprotein_node_list_with_diseases.csv'
proteins_df.to_csv(output_path, index=False)
