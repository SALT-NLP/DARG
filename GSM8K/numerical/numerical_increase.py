import random
from utils import compute_graph_values, compute_complexity, create_computational_graph
import json
import argparse
import os
import random
import itertools
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_file",
    default="numerical.json",
    type=str,
)
parser.add_argument(
    "--original_file",
    default="gsm8k_graph.json",
    type=str,
)
parser.add_argument(
    "--numerical_increase",
    default=1,
    type=float,
)
parser.add_argument(
    "--seed",
    default=42,
    type=float,
)
args = parser.parse_args()

random.seed(a=args.seed)


# Function to adjust initial values based on sampling
def adjust_initial_values_sampled(
    graph, num_complexity_increase, max_iterations=10000000, range_min=2, range_max=100
):

    computed_values = compute_graph_values(graph)
    for node, value in computed_values.items():
        graph["nodes"][node]["value"] = value

    original_complexity = compute_complexity(graph)["Numerical Complexity"]["total"]

    original_value_dic = {}
    for node in graph["nodes"]:
        if graph["nodes"][node]["type"] == "initial":
            original_value_dic[node] = graph["nodes"][node]["value"]

    sampling_range = (range_min, range_max)
    list_cache = []
    for iteration in range(max_iterations):
        if iteration % 200 == 1 and len(list_cache) > 100:
            positive_count = sum(1 for number in list_cache if number > 0)
            percentage_positive = (positive_count / len(list_cache)) * 100
            if percentage_positive > 90:
                sampling_range = (sampling_range[0], int(sampling_range[1] / 2))
                list_cache = []
            elif percentage_positive < 10:
                sampling_range = (sampling_range[0], int(sampling_range[1] * 2))
                list_cache = []

        # Clear all values from the graph's nodes
        for node in graph["nodes"]:
            graph["nodes"][node].pop("value", None)
        # Initialize a dictionary to store the sampled values
        sampled_values = {}
        for node in graph["nodes"]:
            if graph["nodes"][node]["type"] == "initial" and node not in sampled_values:
                is_div_operation = any(
                    edge["operation"] == "div" and edge["from"] == node
                    for edge in graph["edges"].values()
                )
                if is_div_operation:
                    # Find the corresponding edge and node for the division operation
                    div_edge = next(
                        edge
                        for edge in graph["edges"].values()
                        if edge["from"] == node and edge["operation"] == "div"
                    )
                    divisor_node = div_edge["objective"]
                    result_node = div_edge["to"]

                    # Sample values for divisor and result if not already sampled
                    if (
                        graph["nodes"][divisor_node]["type"] == "initial"
                        and 0 < original_value_dic.get(divisor_node) < 1
                    ):
                        divisor_value = sampled_values.get(
                            divisor_node, round(random.uniform(0.01, 0.99), 2)
                        )
                    else:
                        divisor_value = sampled_values.get(
                            divisor_node, random.randint(*sampling_range)
                        )
                    result_value = random.randint(*sampling_range)

                    # Compute the dividend
                    dividend_value = divisor_value * result_value

                    sampled_values[node] = dividend_value
                    sampled_values[divisor_node] = divisor_value
                else:
                    original_value = original_value_dic.get(node)
                    if 0 < original_value < 1:
                        sampled_values[node] = round(random.uniform(0.01, 0.99), 2)
                    else:
                        sampled_values[node] = random.randint(*sampling_range)

        # Update graph with new initial values
        for node, value in sampled_values.items():
            graph["nodes"][node]["value"] = value
        # Recompute values for all nodes
        computed_values = compute_graph_values(graph)
        if computed_values is None:
            continue  # Skip to the next iteration if computation timed out

        for node, value in computed_values.items():
            graph["nodes"][node]["value"] = value
        # Check if all values are integers
        flag = True
        for node in graph["nodes"]:
            node_value = graph["nodes"][node]["value"]
            node_type = graph["nodes"][node]["type"]

            if node_type == "initial":
                # Check if the node value is a non-negative integer or a float with at most two digits after the decimal point
                if not (
                    (
                        (
                            isinstance(node_value, int)
                            or isinstance(value, float)
                            and value.is_integer()
                        )
                        and node_value >= 0
                    )
                    or (
                        isinstance(node_value, float)
                        and node_value >= 0
                        and round(node_value, 2) == node_value
                    )
                ):
                    flag = False
                    break
            else:
                # Check if the node value is a non-negative integer
                if not (
                    (
                        isinstance(node_value, int)
                        or isinstance(value, float)
                        and value.is_integer()
                    )
                    and node_value >= 0
                ):
                    flag = False
                    break

        if flag == False:
            continue
        current_complexity = compute_complexity(graph)["Numerical Complexity"]["total"]
        # Check if the target complexity is achieved within the error margin
        current_target_difference = (
            current_complexity
            - original_complexity
            - num_complexity_increase * len(graph["edges"])
        )
        list_cache.append(current_target_difference)
        if current_target_difference == 0:
            return sampled_values
        elif iteration > 20000 and abs(current_target_difference) <= 1:
            return sampled_values

    return None


output_path = (
    args.output_file[0:-5] + "_increase_" + str(args.numerical_increase) + ".json"
)
if os.path.exists(output_path):
    output_data = []
    with open(output_path, "r") as file:
        for line in file:
            data = json.loads(line)
            output_data.append(data)
else:
    output_data = []
    with open(args.original_file, "r") as file:
        for line in file:
            data = json.loads(line)
            output_data.append(data)

list_edge_idx = []
for i, data in enumerate(reversed(output_data)):
    if len(data["Mapping"].keys()) != 0:
        mapping = data["Mapping"]
    else:
        continue
    if "graph" in data.keys():
        print("skip")
        continue
    if i in list_edge_idx:
        continue
    print("Index: " + str(i))
    graph = create_computational_graph(mapping)
    new_initial_values = adjust_initial_values_sampled(graph, args.numerical_increase)
    if new_initial_values is not None:
        for node in graph["nodes"]:
            if graph["nodes"][node]["type"] != "initial":
                graph["nodes"][node].pop("value", None)
        for node, value in new_initial_values.items():
            graph["nodes"][node]["value"] = value
        print(
            f"New Initial Values for Target Complexity {args.numerical_increase}: {new_initial_values}"
        )
        data["graph"] = graph
        with open(output_path, "w") as f:
            for item in output_data:
                json_item = json.dumps(item)
                f.write(json_item + "\n")

    else:
        print("*********************************************************")
        print(f"Target increased complexity {args.numerical_increase} not reached.")
        print(mapping)
        print("*********************************************************")
