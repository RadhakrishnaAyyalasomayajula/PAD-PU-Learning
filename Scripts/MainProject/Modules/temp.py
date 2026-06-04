def draw_explainer_subgraph(node_idx, edge_importance, out_file):
    th = 0.5
    important_edges = np.where(edge_importance > th)[0].tolist()

    if len(important_edges) == 0:
        return

    edges = []
    for e_idx in important_edges:
        u = int(edges_df.loc[e_idx, "Index1"])
        v = int(edges_df.loc[e_idx, "Index2"])
        edges.append((u, v))

    subG = G.edge_subgraph(edges).copy()
    if subG.number_of_nodes() == 0:
        return

    plt.figure(figsize=(6,6))
    pos = nx.spring_layout(subG, seed=42)

    # node colors
    node_colors = [
        "red" if int(n) == int(node_idx) else "skyblue"
        for n in subG.nodes()
    ]

    # 🔥 FIX: correct label dict
    label_dict = {
        int(n): nodes_df.loc[int(n), "GeneSymbol"]
        for n in subG.nodes()
    }

    nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_edges(subG, pos, alpha=0.7)
    nx.draw_networkx_labels(subG, pos, labels=label_dict, font_size=8)

    plt.title(f"Explainer Subgraph — {nodes_df.loc[node_idx, 'GeneSymbol']}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()