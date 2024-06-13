import random
import json
import argparse
import os
from copy import deepcopy
from utils import (
    create_computational_graph,
    compute_complexity,
    compute_graph_values,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--original_file",
    default="gsm8k_graph.json",
    type=str,
)
parser.add_argument(
    "--depth_increase",
    default=1,
    type=int,
)
args = parser.parse_args()


output_data = []
with open(args.original_file, "r") as file:
    for line in file:
        data = json.loads(line)
        output_data.append(data)


def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def decimal_places(value):
    value_str = str(value)
    decimal_point_index = value_str.find(".")
    if decimal_point_index == -1:
        return 0
    else:
        return len(value_str) - decimal_point_index - 1


def sample_decomposed_values(value, operation, sample_range):

    is_value_int = isinstance(value, int) or (
        isinstance(value, float) and value.is_integer()
    )
    flag_valid = False
    if operation == "+":
        if is_value_int:
            value = int(value)
        if isinstance(value, float) and value <= 3:
            part1 = round(random.uniform(0.01, value - 0.01), 2)
        else:
            part1 = random.randint(1, int(max(min(sample_range, value - 1), 3)))
        part2 = value - part1
        flag_valid = True

    elif operation == "-":
        part1 = value + random.randint(1, sample_range)
        part2 = part1 - value
        flag_valid = True

    elif operation == "*":
        if is_value_int and not is_prime(value) and value != 1:
            value = int(value)
            factors = [i for i in range(2, value) if value % i == 0]
            if factors:
                part1 = random.choice(factors) if factors else 1
                part2 = value / part1
                flag_valid = True
            else:
                part1 = -1
                part2 = -1
                flag_valid = False
        else:
            interval = 0.01
            potential_factors = [
                round(f, 2) for f in frange(interval, sample_range - interval, interval)
            ]
            valid_factors = []
            for factor in potential_factors:
                if abs(factor - 1) < 0.01:
                    continue
                result = value / factor
                # Check if result is an int or has at most two decimal places
                if (
                    isinstance(result, int)
                    or (isinstance(result, float) and round(result, 2) == result)
                    or (isinstance(result, float) and result.is_integer())
                ):
                    valid_factors.append(factor)

            if valid_factors:
                part1 = random.choice(valid_factors)
                part2 = value / part1
                if decimal_places(part2) <= 2:
                    flag_valid = True
                else:
                    flag_valid = False
            else:
                # If no valid factor is found, default to a fallback
                part1 = round(random.uniform(1, sample_range), 2)
                part2 = value / part1
                flag_valid = False

    elif operation == "/":
        multiplier = random.randint(2, sample_range)
        part1 = value * multiplier
        part2 = multiplier
        flag_valid = True

    return part1, part2, flag_valid


def frange(start, stop, step):
    """Generate a range of floating point values."""
    while start < stop:
        yield start
        start += step


def increase_graph_depth_with_decomposition(
    graph, depth_increase=1, sample_range=50, max_iterations=1000000, name_index=1
):
    original_numerical_complexity = compute_complexity(graph)["Numerical Complexity"][
        "ave"
    ]
    orginal_depth = compute_complexity(graph)["Graph Depth"]
    operation_map = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
        "sub_by": "-",
        "div_by": "/",
    }
    operation_map_reversed = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
    operations = {operation_map[edge["operation"]] for edge in graph["edges"].values()}

    for iter in range(depth_increase):
        for iteration in range(max_iterations):
            longest_paths = compute_complexity(graph)["Longest Paths"]
            if not longest_paths:
                print("No path to increase depth.")
                return graph

            start_node = longest_paths[0][0]
            selected_operation = random.choice(list(operations))

            # Decompose the node's value
            value = graph["nodes"][start_node]["value"]
            part1, part2, if_valid = sample_decomposed_values(
                value, selected_operation, sample_range
            )

            if not if_valid:
                continue
            if part1 == part2:
                continue
            if iteration < max_iterations / 2:
                if part1 <= 0 or part2 <= 0:
                    continue
            else:
                if part1 < 0 or part2 < 0:
                    continue
            # Apply changes to a copy of the graph for comparison
            graph_copy = deepcopy(graph)
            new_node1_id = str(name_index) + f"_New1_{len(graph_copy['nodes'])+1}"
            new_node2_id = str(name_index) + f"_New2_{len(graph_copy['nodes'])+1}"
            # intermediate_node_id = f"Intermediate_{start_node}"

            # Assuming the graph structure allows direct addition like this; adjust as necessary for your structure
            graph_copy["nodes"][new_node1_id] = {
                "incoming_edges": [],
                "outgoing_edges": [str(len(graph_copy["edges"]) + 1)],
                "type": "initial",
                "value": part1,
            }
            graph_copy["nodes"][new_node2_id] = {
                "incoming_edges": [],
                "outgoing_edges": [str(len(graph_copy["edges"]) + 2)],
                "type": "initial",
                "value": part2,
            }
            # graph_copy['nodes'][intermediate_node_id] = {'type': 'intermediate', 'value': value, 'incoming_edges': [str(len(graph_copy['edges'])+1), str(len(graph_copy['edges'])+2)], 'outgoing_edges': graph_copy['nodes'][start_node]['outgoing_edges']}
            graph_copy["nodes"][start_node]["incoming_edges"] = [
                str(len(graph_copy["edges"]) + 1),
                str(len(graph_copy["edges"]) + 2),
            ]
            graph_copy["nodes"][start_node]["type"] = "intermediate"

            if selected_operation in ["*", "+"]:
                applied_operation_1 = operation_map_reversed[selected_operation]
                applied_operation_2 = operation_map_reversed[selected_operation]
            else:
                applied_operation_1 = operation_map_reversed[selected_operation]
                applied_operation_2 = operation_map_reversed[selected_operation] + "_by"

            graph_copy["edges"][str(len(graph_copy["edges"]) + 1)] = {
                "from": new_node1_id,
                "objective": new_node2_id,
                "operation": applied_operation_1,
                "to": start_node,
            }
            graph_copy["edges"][str(len(graph_copy["edges"]) + 1)] = {
                "from": new_node2_id,
                "objective": new_node1_id,
                "operation": applied_operation_2,
                "to": start_node,
            }

            # Evaluate the numerical complexity change
            new_complexity = compute_complexity(graph)["Numerical Complexity"]["ave"]
            if abs(new_complexity - original_numerical_complexity) <= 1:
                graph = graph_copy  # Apply the successful modification
                break

    updated_depth = compute_complexity(graph)["Graph Depth"]
    if updated_depth - orginal_depth == depth_increase:
        return graph
    else:
        return None


dict_increased_data = {}
for i in range(args.depth_increase):
    dict_increased_data["Data_depth_increase_" + str(i + 1)] = []

for i, data in enumerate(output_data):
    print(i)
    original_mapping = data["Mapping"]
    original_graph = create_computational_graph(original_mapping)
    depth_increased_graph = deepcopy(original_graph)
    list_graphs = []
    for iter in range(args.depth_increase):
        count = 0
        while count < 1000000:
            random.seed(count + iter)
            temp_graph = increase_graph_depth_with_decomposition(
                depth_increased_graph, depth_increase=1, name_index=iter
            )
            count += 1
            if temp_graph is not None:
                depth_increased_graph = temp_graph
                list_graphs.append(depth_increased_graph)
                break

    computed_values = compute_graph_values(depth_increased_graph)
    for node in depth_increased_graph["nodes"].keys():
        if depth_increased_graph["nodes"][node]["type"] == "final":
            label = float(depth_increased_graph["nodes"][node]["value"])
            computed_label = computed_values[node]
            break
    print(
        compute_complexity(depth_increased_graph)["Graph Depth"]
        - compute_complexity(original_graph)["Graph Depth"]
    )
    for j in range(args.depth_increase):
        data_copy = deepcopy(data)
        data_copy["Depth_increase_" + str(j + 1) + "_graph"] = list_graphs[j]
        dict_increased_data["Data_depth_increase_" + str(j + 1)].append(data_copy)

for i in range(args.depth_increase):
    data_list = dict_increased_data["Data_depth_increase_" + str(i + 1)]
    with open("depth_increase_" + str(i + 1) + ".json", "w") as f:
        for item in data_list:
            json_item = json.dumps(item)
            f.write(json_item + "\n")
