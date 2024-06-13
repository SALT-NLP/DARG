import json
from utils import overlap_match

output_data = []
with open("bbq_graph.json", "r") as file:
    for line in file:
        data = json.loads(line)
        output_data.append(data)


def find_node_name_from_idx(graph, non_unknown_idx_name, option_name):

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


for data in output_data:
    data["Person Node Mapping"] = {}
    non_unknown_idxs = []
    for idx, value in data["Original"]["answer_info"].items():
        if value[1] != "unknown":
            non_unknown_idxs.append(idx)

    for non_unknown_idx in non_unknown_idxs:
        node_name = find_node_name_from_idx(
            data["Graph"],
            data["Original"]["answer_info"][non_unknown_idx][0],
            data["Original"][non_unknown_idx],
        )
        data["Person Node Mapping"][node_name] = non_unknown_idx


with open("bbq_graph.json", "w") as f:
    for item in output_data:
        json_item = json.dumps(item)
        f.write(json_item + "\n")
