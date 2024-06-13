import os
from langchain.callbacks import get_openai_callback
import json
import argparse
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument(
    "--step_increase",
    default=2,
    type=int,
)
parser.add_argument(
    "--output_file",
    default="navigate_graph_step_increase_2_text.json",
    type=str,
)
parser.add_argument(
    "--original_file",
    default="navigate_graph_step_increase_2.json",
    type=str,
)

args = parser.parse_args()


def graph_to_instructions(graph):
    instructions = [
        "Always face forward."
    ]  # Initial instruction to always face forward

    for node in sorted(graph["nodes"], key=lambda x: x["order"]):
        direction = node["direction"]
        steps = node["step_num"]
        if direction == "forward":
            action = "Take {} steps forward.".format(steps)
        elif direction == "backward":
            action = "Take {} steps backward.".format(steps)
        elif direction == "left":
            action = "Take {} steps left.".format(steps)
        elif direction == "right":
            action = "Take {} steps right.".format(steps)
        instructions.append(action)

    return " ".join(instructions)


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


template = "If you follow these instructions, do you return to the starting point? {instruction} \nOptions:\n- Yes\n- No"
with get_openai_callback() as cb:
    for i, data in enumerate(output_data):
        if "Updated Instruction" in data.keys():
            print("Skip")
            continue
        print("index:" + str(i))
        graph = deepcopy(data["Updated Graph"])
        instruction = graph_to_instructions(graph)
        complete_instruction = template.format(instruction=instruction)
        data["Updated Instruction"] = complete_instruction
        print(complete_instruction)

with open(args.output_file, "w") as f:
    for item in output_data:
        json_item = json.dumps(item)
        f.write(json_item + "\n")

print(all("Updated Instruction" in data.keys() for data in output_data))
