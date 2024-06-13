import os
import json
import random
import argparse
from utils import overlap_match
from copy import deepcopy

parser = argparse.ArgumentParser()

parser.add_argument(
    "--increase",
    default=1,
    type=int,
)
parser.add_argument(
    "--output_file",
    default="bbq_500_polarity_increase_1.json",
    type=str,
)
parser.add_argument(
    "--original_file",
    default="bbq_graph.json",
    type=str,
)
parser.add_argument(
    "--negative_attributes_file",
    default="bbq_negative_attributes_filtered.json",
    type=str,
)
parser.add_argument(
    "--postive_attributes_file",
    default="bbq_positve_attributes_filtered.json",
    type=str,
)

args = parser.parse_args()

if os.path.exists(args.output_file):
    output_data = []
    with open(args.output_file, "r") as file:
        for line in file:
            data = json.loads(line)
            output_data.append(data)
else:
    output_data = []
    with open(args.original_file, "r") as file:
        for line in file:
            data = json.loads(line)
            output_data.append(data)

negative_attributes = []
with open(args.negative_attributes_file, "r") as file:
    for line in file:
        data = json.loads(line)
        negative_attributes.append(data)

positive_attributes = []
with open(args.postive_attributes_file, "r") as file:
    for line in file:
        data = json.loads(line)
        positive_attributes.append(data)


def find_node_name_from_idx(graph, non_unknown_idx_name, option_name):

    # Identify person nodes
    person_nodes_content = [
        node["content"] for node in graph["nodes"] if node["type"] == "person"
    ]

    for person_node_name in person_nodes_content:
        if non_unknown_idx_name.lower() in person_node_name.lower().split(" "):
            return person_node_name

    for person_node_name in person_nodes_content:
        if overlap_match(person_node_name, non_unknown_idx_name) or overlap_match(
            person_node_name, option_name
        ):
            return person_node_name

    return None


def retrieve_attributes(
    data,
    attributes_list_all,
    type="Common Attributes",
    num_samples=1,
    person_att_dic=None,
):
    example_id = data["Original"]["example_id"]
    ques_idx = data["Original"]["question_index"]
    category = data["Original"]["category"]
    attributes = None
    if type == "Common Attributes" or type == "Label Attributes":
        for data_2 in attributes_list_all:
            if (
                data_2["Original"]["example_id"] == example_id
                and data_2["Original"]["question_index"] == ques_idx
                and data_2["Original"]["category"] == category
            ):
                attributes = data_2[type]
                break
        sampled_elements = random.sample(attributes, num_samples)
        return sampled_elements
    else:
        sampled_attributes_dict = {}
        for person_node, _ in person_att_dic.items():
            person_idx = data["Person Node Mapping"][person_node]
            for data_2 in attributes_list_all:
                if (
                    data_2["Original"]["example_id"] == example_id
                    and data_2["Original"]["question_index"] == ques_idx
                    and data_2["Original"]["category"] == category
                ):
                    attributes = data_2[type][person_idx]
                    sampled_elements = random.sample(attributes, num_samples)
                    sampled_attributes_dict[person_node] = sampled_elements
                    break

        return sampled_attributes_dict


def add_nodes_by_description(graph, attributes_by_description):
    """
    Add new attribute nodes to a graph based on descriptions of existing person nodes and update edges to connect these new nodes.

    Args:
    graph (dict): The graph containing 'nodes' and 'edges'.
    attributes_by_description (dict): A dictionary where keys are descriptions of person nodes and values are lists of dicts with 'node content' and 'edge content' for the new nodes.

    Returns:
    dict: The updated graph with new nodes and edges.
    """
    # Map existing node contents to their IDs
    node_content_to_id = {
        node["content"]: node["id"]
        for node in graph["nodes"]
        if node["type"] == "person"
    }

    current_max_id = max(
        node["id"] for node in graph["nodes"]
    )  # Find the highest current node ID to ensure new IDs are unique

    # Process each description and its corresponding attributes
    for description, nodes in attributes_by_description.items():
        if description in node_content_to_id:
            person_id = node_content_to_id[description]
            for node in nodes:
                current_max_id += 1
                # Add new node
                graph["nodes"].append(
                    {
                        "id": current_max_id,
                        "content": node["node content"],
                        "type": "attribute",
                    }
                )
                # Add edge from person to this new attribute node
                graph["edges"].append(
                    {
                        "source": person_id,
                        "target": current_max_id,
                        "content": node["edge content"],
                        "type": "to_attribute",
                    }
                )

    return graph


def extract_unique_unrelated_attributes(graph):
    # Extract person nodes
    person_nodes = {
        node["id"]: node["content"]
        for node in graph["nodes"]
        if node["type"] == "person"
    }

    # Intermediate dictionary to hold all unrelated attributes for each person
    person_attributes = {person_nodes[id]: [] for id in person_nodes}

    # Collect all unrelated attributes per person
    for edge in graph["edges"]:
        source = edge["source"]
        target = edge["target"]

        # Check if source is a person and target is an attribute with 'question_relation' == 'unrelated'
        if source in person_nodes and any(
            node["id"] == target
            and node["type"] == "attribute"
            and node["question_relation"] == "unrelated"
            for node in graph["nodes"]
        ):
            # Find the content of the unrelated attribute node
            attribute_content = next(
                node["content"] for node in graph["nodes"] if node["id"] == target
            )
            person_attributes[person_nodes[source]].append(attribute_content)

    # Filter attributes to keep only those that are unique to each person
    unique_person_attributes = {}
    for person, attributes in person_attributes.items():
        # Find attributes that are only associated with this person
        unique_attributes = [
            attr
            for attr in attributes
            if all(
                attr not in person_attributes[other]
                for other in person_attributes
                if other != person
            )
        ]
        unique_person_attributes[person] = unique_attributes

    return unique_person_attributes


for i, data in enumerate(output_data):
    if "Updated Graph" in data.keys():
        print("skip")
        continue
    ori_graph = deepcopy(data["Graph"])
    person_attributes_nodes = extract_unique_unrelated_attributes(ori_graph)
    target = data["Original"]["target_loc"]
    ques_polarity = data["Original"]["question_polarity"]
    node_mapping = data["Person Node Mapping"]
    new_person_attributes = {}
    for person_node, _ in person_attributes_nodes.items():
        if node_mapping[person_node] == "ans" + target:
            if ques_polarity == "nonneg":
                retrieved_attributes_dic = retrieve_attributes(
                    data=data,
                    attributes_list_all=positive_attributes,
                    type="Positive Attributes Filtered",
                    person_att_dic=person_attributes_nodes,
                    num_samples=args.increase,
                )
            elif ques_polarity == "neg":
                retrieved_attributes_dic = retrieve_attributes(
                    data=data,
                    attributes_list_all=negative_attributes,
                    type="Negative Attributes Filtered",
                    person_att_dic=person_attributes_nodes,
                    num_samples=args.increase,
                )
        else:
            if ques_polarity == "nonneg":
                retrieved_attributes_dic = retrieve_attributes(
                    data=data,
                    attributes_list_all=negative_attributes,
                    type="Negative Attributes Filtered",
                    person_att_dic=person_attributes_nodes,
                    num_samples=args.increase,
                )
            elif ques_polarity == "neg":
                retrieved_attributes_dic = retrieve_attributes(
                    data=data,
                    attributes_list_all=positive_attributes,
                    type="Positive Attributes Filtered",
                    person_att_dic=person_attributes_nodes,
                    num_samples=args.increase,
                )

        new_person_attributes[person_node] = retrieved_attributes_dic[person_node]
    print("*************************")
    print(new_person_attributes)
    ori_graph = add_nodes_by_description(ori_graph, new_person_attributes)
    data["Updated_Graph_increase_" + str(args.increase)] = ori_graph

with open(args.output_file, "w") as f:
    for item in output_data:
        json_item = json.dumps(item)
        f.write(json_item + "\n")
