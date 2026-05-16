import pickle

# Import your project classes (IMPORTANT)
from raptor.tree_structures import Tree, Node


def load_tree(path):
    with open(path, "rb") as f:
        tree = pickle.load(f)

    if not isinstance(tree, Tree):
        raise ValueError("Loaded object is not a Tree")

    return tree


def print_tree_summary(tree):
    print("\n=== TREE SUMMARY ===")
    print(f"Total nodes: {len(tree.all_nodes)}")
    print(f"Leaf nodes: {len(tree.leaf_nodes)}")
    print(f"Root nodes: {len(tree.root_nodes)}")
    print(f"Number of layers: {tree.num_layers}")

    print("\nNodes per layer:")
    for layer, nodes in tree.layer_to_nodes.items():
        print(f"  Layer {layer}: {len(nodes)} nodes")


def print_sample_nodes(tree, num_samples=3):
    print("\n=== SAMPLE LEAF NODES ===")
    leaf_nodes = list(tree.leaf_nodes.values())

    for i, node in enumerate(leaf_nodes[:num_samples]):
        print(f"\n[Leaf Node {i}]")
        print(f"Index: {node.index}")
        print(f"Text preview: {node.text[:200]}...")
        print(f"Children: {node.children}")

    print("\n=== SAMPLE ROOT NODES ===")

    # 🔥 FIX: normalize to list
    root_nodes = tree.root_nodes

    if isinstance(root_nodes, dict):
        root_nodes = list(root_nodes.values())
    elif isinstance(root_nodes, set):
        root_nodes = list(root_nodes)

    for i, node in enumerate(root_nodes[:num_samples]):
        print(f"\n[Root Node {i}]")
        print(f"Index: {node.index}")
        print(f"Text preview: {node.text[:200]}...")
        print(f"Children: {node.children}")


def inspect_node(tree, node_index):
    if node_index not in tree.all_nodes:
        print("Node not found")
        return

    node = tree.all_nodes[node_index]

    print("\n=== NODE DETAILS ===")
    print(f"Index: {node.index}")
    print(f"Text:\n{node.text}")
    print(f"Children: {node.children}")
    print(f"Embedding keys: {list(node.embeddings.keys())}")


def interactive_view(tree):
    print("\n=== INTERACTIVE MODE ===")
    print("Type a node index to inspect, or 'q' to quit")

    while True:
        user_input = input("Node index> ")

        if user_input.lower() == "q":
            break

        try:
            idx = int(user_input)
            inspect_node(tree, idx)
        except:
            print("Invalid input")


if __name__ == "__main__":
    path = "demo_tree.pkl"  # <-- change if needed

    tree = load_tree(path)

    print_tree_summary(tree)
    print_sample_nodes(tree)

    interactive_view(tree)