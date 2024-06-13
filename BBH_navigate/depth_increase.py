import os
from langchain.callbacks import get_openai_callback
import json
import random
import argparse
from copy import deepcopy
from utils import returns_to_start

parser = argparse.ArgumentParser()
parser.add_argument(
    "--step_increase",
    default=2,
    type=int,
)
parser.add_argument(
    "--output_file",
    default="navigate_graph_step_increase_2.json",
    type=str,
)
parser.add_argument(
    "--orginal_file",
    default="navigate_graph.json",
    type=str,
)


def extract_instructions(text):
    # Find the index of the first '?'
    start_idx = text.find("?") + 1
    if start_idx == 0:
        # If '?' is not found, return an empty string
        return "No question mark found in the text."

    # Find the index of 'Options:'
    end_idx = text.find("Options:")
    if end_idx == -1:
        # If 'Options:' is not found, return an empty string
        return "No 'Options:' found in the text."

    # Extract the substring between these indices, stripping any leading/trailing whitespace
    extracted_text = text[start_idx:end_idx].strip()

    return extracted_text


def valid_decomposition(graph, original_node, decomposed_nodes):
    # Simulate movements for the original node and the decomposed nodes
    def simulate_movement(nodes):
        x, y = 0, 0
        for node in nodes:
            steps = node["step_num"]
            if node["direction"] == "forward":
                y += steps
            elif node["direction"] == "backward":
                y -= steps
            elif node["direction"] == "left":
                x -= steps
            elif node["direction"] == "right":
                x += steps
        return (x, y)

    # Compare end points
    original_endpoint = simulate_movement([original_node])
    decomposed_endpoint = simulate_movement(decomposed_nodes)
    return original_endpoint == decomposed_endpoint


def decompose_action(graph, increase_ops=1, max_iterations=1000000):
    # Randomly select a node to decompose
    original_node = random.choice(graph["nodes"])
    original_index = graph["nodes"].index(original_node)

    # Remove the original node from the graph
    graph["nodes"].remove(original_node)

    directions = ["forward", "backward", "left", "right"]
    for _ in range(max_iterations):
        new_steps = [
            random.randint(1, max(10, original_node["step_num"] * 10))
            for _ in range(increase_ops)
        ]
        new_directions = [random.choice(directions) for _ in range(increase_ops)]

        # Create potential new nodes
        new_nodes = [
            {"order": original_node["order"], "step_num": step, "direction": direction}
            for step, direction in zip(new_steps, new_directions)
        ]

        # Validate if the decomposed actions have the same effect
        if valid_decomposition(graph, original_node, new_nodes):
            # Increment the order of subsequent nodes
            for node in graph["nodes"]:
                if node["order"] > original_node["order"]:
                    node["order"] += increase_ops - 1

            # Insert new nodes into the graph at the correct order
            for i, new_node in enumerate(new_nodes):
                new_node["order"] = original_node["order"] + i
                graph["nodes"].append(new_node)

            # Sort all nodes to maintain order
            graph["nodes"].sort(key=lambda x: x["order"])

            return graph, original_node, new_nodes

    # If no valid decomposition is found, restore the original node and adjust orders back
    graph["nodes"].append(original_node)
    for node in graph["nodes"]:
        if node["order"] > original_node["order"]:
            node["order"] -= increase_ops - 1
    graph["nodes"].sort(key=lambda x: x["order"])
    return graph, None, None


args = parser.parse_args()

if os.path.exists(args.output_file):
    output_data = []
    with open(args.output_file, "r") as file:
        for line in file:
            data = json.loads(line)
            output_data.append(data)
else:
    output_data = []
    with open(args.orginal_file, "r") as file:
        for line in file:
            data = json.loads(line)
            output_data.append(data)

with get_openai_callback() as cb:
    for i, data in enumerate(output_data):
        if "Updated Graph" in data.keys():
            print("Skip")
            continue
        print("index:" + str(i))
        graph = deepcopy(data["Graph"])
        modified_graph, replaced_node, new_combinations = decompose_action(
            graph, increase_ops=args.step_increase + 1
        )
        if replaced_node != None and (
            returns_to_start(data["Graph"])
        ) == returns_to_start(modified_graph):
            data["Updated Graph"] = modified_graph
        else:
            continue

with open(args.output_file, "w") as f:
    for item in output_data:
        json_item = json.dumps(item)
        f.write(json_item + "\n")
