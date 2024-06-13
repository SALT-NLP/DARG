import random
import json
import argparse
import os
from copy import deepcopy
from utils import (
    create_computational_graph,
    compute_complexity,
    find_non_longest_paths,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--original_file",
    default="gsm8k_graph.json",
    type=str,
)
parser.add_argument(
    "--width_increase",
    default=4,
    type=int,
)
args = parser.parse_args()


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

    part1 = round(part1, 8)
    part2 = round(part2, 8)
    return part1, part2, flag_valid


def frange(start, stop, step):
    """Generate a range of floating point values."""
    while start < stop:
        yield start
        start += step


def decompose_start_node_and_increase_path_width(
    graph, width_increase=1, sample_range=5000, name_index=1, max_iterations=1000000
):
    """
    Decompose the starting node in one of the non-longest paths of a graph to increase the width of this path,
    limited by the width_increase parameter and ensuring not to exceed the longest path's length, while keeping the numerical complexity nearly unchanged.

    Args:
    - graph: The computational graph.
    - width_increase: The number of additional nodes to add to the non-longest path, within the limit of not exceeding the longest path length.
    - sample_range: The range to sample decomposed values.
    - name_index: Index to name new nodes uniquely.

    Returns:
    - A tuple of the modified graph and the actual width increase.
    """

    # Mapping from your original function
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

    original_numerical_complexity = compute_complexity(graph)["Numerical Complexity"][
        "ave"
    ]
    actual_width_increased = 0

    for iter in range(width_increase):
        non_longest_paths = find_non_longest_paths(graph)
        current_longest_path = compute_complexity(graph)["Longest Paths"]
        if len(non_longest_paths) == 0:
            break  # Break if there are no non-longest paths to decompose
        else:
            list_valid_non_longest_start = []
            for non_longest_path in non_longest_paths:
                non_longest_start = non_longest_path[0]
                if all(
                    non_longest_start not in sublist for sublist in current_longest_path
                ):
                    list_valid_non_longest_start.append(non_longest_start)
            if len(list_valid_non_longest_start) == 0:
                break
        for iteration in range(max_iterations):
            start_node = random.choice(list_valid_non_longest_start)

            # Randomly select an operation based on available operations in the graph
            selected_operation = random.choice(list(operations))

            value = graph["nodes"][start_node]["value"]
            part1, part2, is_valid = sample_decomposed_values(
                value, selected_operation, sample_range
            )
            if part1 == part2:
                continue
            if value > 0:
                if not is_valid or part1 <= 0 or part2 <= 0:
                    continue
            else:
                if not is_valid:
                    continue
            graph_copy = deepcopy(graph)
            new_node1_id = str(name_index) + f"_New1_{len(graph_copy['nodes'])+1}"
            new_node2_id = str(name_index) + f"_New2_{len(graph_copy['nodes'])+1}"

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
            new_complexity = compute_complexity(graph_copy)["Numerical Complexity"][
                "ave"
            ]
            if abs(new_complexity - original_numerical_complexity) <= 1:
                graph = graph_copy  # Apply the successful modification
                actual_width_increased += 1
                break
            else:
                continue  # Stop if the modification significantly changes the numerical complexity

    return graph, actual_width_increased


output_data = []
with open(args.original_file, "r") as file:
    for line in file:
        data = json.loads(line)
        output_data.append(data)

width_increased_data = {}
for i in range(args.width_increase):
    width_increased_data["Data_width_increase_" + str(i + 1)] = []
MAX_ITER = 1000000

for i, data in enumerate(output_data):
    graph = create_computational_graph(data["Mapping"])
    non_longest_paths = find_non_longest_paths(graph)
    complexity = compute_complexity(graph)
    longest_path = complexity["Longest Paths"]
    print("***********************************")
    print("Index: " + str(i))
    print("Non-longest Path: ")
    print(non_longest_paths)
    print("Longest Path: ")
    print(longest_path)
    print("***********************************")
    list_graphs = []
    list_actual_increase = []
    increase = 0
    width_increased_graph = deepcopy(graph)
    for iter in range(args.width_increase):
        count = 0
        while count < MAX_ITER:
            random.seed(count + iter)
            temp_graph, actual_width_increased = (
                decompose_start_node_and_increase_path_width(
                    width_increased_graph,
                    width_increase=1,
                    sample_range=5000,
                    name_index=1,
                )
            )
            count += 1
            if temp_graph is not None:
                width_increased_graph = temp_graph
                list_graphs.append(width_increased_graph)
                increase += actual_width_increased
                list_actual_increase.append(increase)
                break
    for j in range(args.width_increase):
        data_copy = deepcopy(data)
        data_copy["Width_increase_" + str(j + 1) + "_graph"] = list_graphs[j]
        data_copy["Actual Width Increase"] = list_actual_increase[j]
        width_increased_data["Data_width_increase_" + str(j + 1)].append(data_copy)


for i in range(args.width_increase):
    data_list = width_increased_data["Data_width_increase_" + str(i + 1)]
    with open("width_increase_" + str(i + 1) + ".json", "w") as f:
        for item in data_list:
            json_item = json.dumps(item)
            f.write(json_item + "\n")
