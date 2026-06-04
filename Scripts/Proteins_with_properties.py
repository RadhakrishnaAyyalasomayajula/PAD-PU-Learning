import pandas as pd
import numpy as np
import os

# Change the working directory
os.chdir('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/Include_IndirectConnections/Scripts')
df = pd.read_csv('../CuratedData/LargestComponents/filtered_df.csv')
df['NTSEQ Value'] = df['NTSEQ Value'].str.rstrip('///')
#
df_unique = df.drop_duplicates(subset='GeneSymbol', keep='first')
#
df_unique.to_csv('../CuratedData/LargestComponents/filtered_df2.csv', index=False)

property_columns = ['Molecular Weight', 'AASEQ Value', 'AASEQ Length', 'NTSEQ Value', 'NTSEQ Length']
df_cleaned = df_unique.dropna(subset=property_columns)


df_cleaned = df_cleaned[df_cleaned['Molecular Weight'] != 'Not available']
df_cleaned.to_csv('../CuratedData/LargestComponents/cleaned_output.csv', index=False)
########################################################################################################################
########################################################################################################################
########################################################################################################################
import requests
import numpy as np
import pandas as pd
#
os.chdir('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/Include_IndirectConnections/Scripts')
df_cleaned = pd.read_csv('../CuratedData/LargestComponents/cleaned_output.csv')
def fetch_gene_description_from_string(gene_symbol, species=9606, retries=3, delay=5):
    string_api_url = "https://string-db.org/api/json/get_string_ids"
    params = {
        "identifiers": gene_symbol,
        "species": species,
        "limit": 1,
        "echo_query": 1
    }
    headers = {
        "User-Agent": "Python Script"
    }

    for attempt in range(retries):
        try:
            response = requests.get(string_api_url, params=params, headers=headers)
            if response.status_code == 200:
                try:
                    response_json = response.json()
                except ValueError as ve:
                    print(f"Failed to parse JSON response for {gene_symbol}: {ve}")
                    print(f"Response content: {response.text}")
                    continue

                if isinstance(response_json, list) and len(response_json) > 0:
                    result = response_json[0]
                    if 'annotation' in result:
                        return result['annotation']
                    elif 'preferredName' in result:
                        return result['preferredName']
                    else:
                        print(f"No description found for {gene_symbol}")
                        return None
                else:
                    print(f"Unexpected response format for {gene_symbol}: {response_json}")
                    print(f"Response content: {response.text}")
                    if attempt < retries - 1:
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
            else:
                print(f"Failed to fetch data for {gene_symbol}: HTTP {response.status_code}")
                if attempt < retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
        except requests.exceptions.RequestException as e:
            print(f"RequestException fetching description for {gene_symbol}: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
        except Exception as ex:
            print(f"Exception fetching description for {gene_symbol}: {ex}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)

    return None
#
for i in range(df_cleaned.shape[0]):
    gene_symbol = df_cleaned.loc[i, 'GeneSymbol']  # Access GeneSymbol by row number and column name
    if df_cleaned.loc[i, 'Function'] == 'Not available':  # Access 'Function' column by row number
        print(df_cleaned.loc[i, 'Function'])
        description = fetch_gene_description_from_string(gene_symbol)  # Fetch description based on GeneSymbol
        df_cleaned.loc[i, 'Function'] = description  # Update 'Function' column with the new description
#
df_cleaned.to_csv('../CuratedData/LargestComponents/cleaned_output.csv', index=False)
#
import requests
import numpy as np
import pandas as pd
#
os.chdir('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/Include_IndirectConnections/Scripts')

df_cleaned = pd.read_csv('../CuratedData/LargestComponents/cleaned_output.csv')
df_cleaned['Function'] = df_cleaned['Function'].str.replace(r'\s*\(PubMed:.*?\)', '', regex=True)
df_cleaned['Function'] = df_cleaned['Function'].str.replace(r'\[\.\.\.\]$', '', regex=True)

df_cleaned['Function'] = df_cleaned['Function'].str.strip()
df_cleaned.to_csv('../CuratedData/LargestComponents/cleaned_output.csv', index=False)
#
#
import requests
import numpy as np
import pandas as pd
os.chdir('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/Include_IndirectConnections/Scripts')
nodes_df = pd.read_csv('../CuratedData/LargestComponents/Nodes_with_properties.csv')
nodes_df = nodes_df.reset_index(drop=True)
old_to_new_index_map = {old_index: new_index for new_index, old_index in enumerate(nodes_df['NodeIndex'])}
nodes_df['NodeIndex'] = range(len(nodes_df))  # Reassign continuous NodeIndex values
#
nodes_df.to_csv('../CuratedData/LargestComponents/updated_nodes.csv', index=False)
##
##
edges_df = pd.read_csv('../CuratedData/LargestComponents/Edges.csv')
edges_df['Index1'] = edges_df['Index1'].map(old_to_new_index_map)
edges_df['Index2'] = edges_df['Index2'].map(old_to_new_index_map)

# Drop edges where either source or target has been removed (i.e., they map to NaN)
edges_df = edges_df.dropna(subset=['Index1', 'Index2']).reset_index(drop=True)
edges_df.to_csv('../CuratedData/LargestComponents/updated_edges.csv', index=False)
####
####
####
import pandas as pd
import requests

os.chdir('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/Include_IndirectConnections/Scripts')
# Function to fetch protein names from UniProt API
def fetch_uniprot_labels(uniprot_ids):
    base_url = "https://www.uniprot.org/uniprot/"
    labels = []

    for uniprot_id in uniprot_ids:
        response = requests.get(f"{base_url}{uniprot_id}.txt")

        if response.status_code == 200:
            data = response.text
            protein_name = ""

            # Extract the protein name from the data
            for line in data.splitlines():
                if line.startswith("ID"):
                    continue
                if line.startswith("DE") and "RecName" in line:
                    protein_name = line.split("RecName: Full=")[1].split(";")[0].strip()
                    break  # Exit loop once protein name is found

            labels.append({
                'UniProt ID': uniprot_id,
                'Protein Name': protein_name
            })
        else:
            print(f"Error fetching data for {uniprot_id}: {response.status_code}")
            labels.append({
                'UniProt ID': uniprot_id,
                'Protein Name': None
            })

    return pd.DataFrame(labels)


# Read UniProt IDs from the CSV file
input_csv = '../CuratedData/LargestComponents/Nodes_with_properties.csv'  # Replace with your CSV file name
df = pd.read_csv(input_csv)
uniprot_ids = df['UniProt ID'].tolist()  # Extract the UniProt IDs

# Fetch protein names for the UniProt IDs
labels_df = fetch_uniprot_labels(uniprot_ids)

df['Label'] = labels_df['Protein Name']
# Save the updated DataFrame back to a new CSV file
output_csv = '../CuratedData/LargestComponents/Nodes_with_properties1.csv'  # Specify the output file name
df.to_csv(output_csv, index=False)
#
#
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import os
import networkx as nx
os.chdir('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/VasculaidKG/VasculaidKG/CuratedData/Proteins/Include_IndirectConnections/Scripts')
#
G = nx.Graph()
#
nodes = pd.read_csv('../CuratedData/LargestComponents/Nodes_with_properties1.csv')
edges = pd.read_csv('../CuratedData/LargestComponents/Edges_with_nodeproperties.csv')
# Add nodes to the graph
G.add_nodes_from(nodes['NodeIndex'].values)  # Adjust based on the actual node column name
#
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