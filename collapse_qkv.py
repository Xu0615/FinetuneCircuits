import os
import json
import argparse

def collapse_qkv_nodes(source_directory, target_directory):
    """
    Traverse all JSON files in the source directory, merge the Q, K, V nodes of each attention head 
    into the corresponding attention head node, and save the modified files to the target directory.

    :param source_directory: Path to the source directory containing JSON files to process
    :param target_directory: Path to the target directory to save the modified JSON files
    """
    # Check if the source directory exists
    if not os.path.isdir(source_directory):
        print(f"Source directory does not exist: {source_directory}")
        return

    # Create the target directory (if it does not exist)
    os.makedirs(target_directory, exist_ok=True)
    print(f"Target directory created or already exists: {target_directory}\n")

    # Iterate through all files in the source directory
    for filename in os.listdir(source_directory):
        if filename.endswith('.json'):
            source_filepath = os.path.join(source_directory, filename)
            target_filepath = os.path.join(target_directory, filename)
            print(f"Processing file: {source_filepath}")

            # Read the content of the source JSON file
            with open(source_filepath, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError as e:
                    print(f"JSON decoding error in file {filename}: {e}")
                    continue

            nodes = data.get('nodes', {})
            edges = data.get('edges', {})

            # Create a new edges dictionary to store adjusted edges
            new_edges = {}

            for edge, props in edges.items():
                # Split the source and destination of the edge
                if '->' not in edge:
                    print(f"Invalid edge format: {edge}")
                    continue
                src, dst = edge.split('->')

                # Check if the current edge involves Q/K/V nodes
                src_contains_qkv = contains_qkv(src)
                dst_contains_qkv = contains_qkv(dst)

                # Replace <q>, <k>, <v> in the source and destination nodes
                new_src = replace_qkv_with_head(src)
                new_dst = replace_qkv_with_head(dst)

                # Only consider merging into new edges if the original edge's `in_graph` is True
                if props.get('in_graph', False):
                    # Reconstruct the edge name
                    new_edge = f"{new_src}->{new_dst}"

                    # Merge `score` values for the same edge
                    if new_edge in new_edges:
                        new_edges[new_edge]['score'] += props.get('score', 0)
                        # If any original edge's `in_graph` is True, the merged edge is also True
                        new_edges[new_edge]['in_graph'] = True
                    else:
                        # Copy props and ensure `in_graph` is True
                        new_props = props.copy()
                        new_props['in_graph'] = True
                        new_edges[new_edge] = new_props

            # Remove all nodes containing <q>, <k>, <v>
            nodes_before = len(nodes)
            nodes = {node: value for node, value in nodes.items() if not contains_qkv(node)}
            nodes_after = len(nodes)
            removed_nodes = nodes_before - nodes_after
           
            # Update the nodes and edges in the JSON data
            data['nodes'] = nodes
            data['edges'] = new_edges

            # Write the modified data to the target JSON file
            with open(target_filepath, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4)
                print(f"Modified file saved: {target_filepath}\n")

    print("All files have been processed.")

def replace_qkv_with_head(node_name):
    """
    If the node name contains <q>, <k>, or <v>, replace it with the corresponding attention head node name.
    Otherwise, return the original node name.

    :param node_name: Node name
    :return: The replaced node name
    """
    if '<q>' in node_name or '<k>' in node_name or '<v>' in node_name:
        # Find the first occurrence of <q>, <k>, or <v>
        for suffix in ['<q>', '<k>', '<v>']:
            if suffix in node_name:
                return node_name.replace(suffix, '')
    return node_name

def contains_qkv(node_name):
    """
    Check if the node name contains <q>, <k>, or <v>.

    :param node_name: Node name
    :return: True if it contains any of these, otherwise False
    """
    return '<q>' in node_name or '<k>' in node_name or '<v>' in node_name

def is_attention_head(node_name):
    """
    Check if a node is an attention head node, assuming its naming format is 'aX.hY'.

    :param node_name: Node name
    :return: True if it is an attention head node, otherwise False
    """
    # Examples: 'a0.h0', 'a1.h15', etc.
    parts = node_name.split('.')
    if len(parts) != 2:
        return False
    layer, head = parts
    if not layer.startswith('a') or not head.startswith('h'):
        return False
    # Further validate the numeric parts
    try:
        int(layer[1:])
        int(head[1:])
        return True
    except ValueError:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collapse QKV nodes into attention head nodes in JSON files.")
    parser.add_argument("--source_directory", type=str, default="/home/dslabra5/EAP-IG/2_arithmetic_operations_100/graph_results_100/rl_graph_results")
    parser.add_argument("--target_directory", type=str, default="/home/dslabra5/EAP-IG/2_arithmetic_operations_100/graph_results_100/rl_graph_results")
    args = parser.parse_args()

    collapse_qkv_nodes(args.source_directory, args.target_directory)