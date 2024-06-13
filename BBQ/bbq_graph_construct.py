import os
from langchain.callbacks import get_openai_callback
import json
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import time
import argparse
from utils import (
    prepare_llm,
    bbq_label_compute,
)
from prompt import bbq_graph_construct

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    default="gpt-4-1106",
    choices=["gpt-4-1106"],
    type=str,
)
parser.add_argument(
    "--temperature",
    default=0.9,
    type=float,
)
parser.add_argument(
    "--max_token",
    default=2048,
    type=float,
)
parser.add_argument(
    "--output_file",
    default="bbq_graph.json",
    type=str,
)
parser.add_argument(
    "--original_file",
    default="bbq_sampled.json",
    type=str,
)


def check_person_attribute_connections(graph):
    # Dictionary to store person nodes and their connection status to attributes
    person_to_attribute_connected = {}

    # Initialize all person nodes in the dictionary with False (no connections found yet)
    for node in graph["nodes"]:
        if node["type"] == "person":
            person_to_attribute_connected[node["id"]] = False

    # Check edges for connections from person nodes to attribute nodes
    for edge in graph["edges"]:
        if edge["source"] in person_to_attribute_connected and edge["target"] in {
            node["id"] for node in graph["nodes"] if node["type"] == "attribute"
        }:
            person_to_attribute_connected[edge["source"]] = True

    # Check if all person nodes have at least one attribute connection
    all_connected = all(person_to_attribute_connected.values())

    return all_connected


def check_direct_person_to_label_connection(graph):
    # Identify person and label nodes
    person_ids = {node["id"] for node in graph["nodes"] if node["type"] == "person"}
    label_ids = {node["id"] for node in graph["nodes"] if node["type"] == "label"}

    # Check if any edge directly connects a person node to a label node
    for edge in graph["edges"]:
        if edge["source"] in person_ids and edge["target"] in label_ids:
            return True  # Found a direct connection from a person to a label

    return False


args = parser.parse_args()
llm = prepare_llm(
    model_name=args.model_name,
    engine=args.model_name,
    max_tokens=args.max_token,
    temperature=args.temperature,
    top_p=0.95,
)

if os.path.exists(args.output_file):
    output_data = []
    with open(args.output_file, "r") as file:
        for line in file:
            data = json.loads(line)
            output_data.append(data)
else:
    output_data = []
    test_data = []
    with open(args.original_file, "r") as file:
        for line in file:
            data = json.loads(line)
            test_data.append(data)
    for i, data in enumerate(test_data):
        output_data.append({"Original": data, "Graph": {}})

template = bbq_graph_construct
response_schemas = [
    ResponseSchema(
        name="Graph", description="the dictionary of the Graph", type="dictionary"
    ),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template=template,
    input_variables=[
        "context_condition",
        "category",
        "answer_info",
        "context",
        "question",
        "label",
    ],
    partial_variables={"format_instructions": format_instructions},
)

with get_openai_callback() as cb:
    for i, data in enumerate(output_data):
        if len(data["Graph"].keys()) != 0:
            print("skip")
            continue
        print("index:" + str(i))
        _input = prompt.format_prompt(
            context_condition=data["Original"]["context_condition"],
            category=data["Original"]["category"],
            answer_info=data["Original"]["answer_info"],
            context=data["Original"]["context"],
            question=data["Original"]["question"],
            label=data["Original"]["label"],
        )
        print(data["Original"]["context"])
        print(data["Original"]["question"])
        print(data["Original"]["answer_info"])
        result = llm.invoke(_input.to_string())
        parsed_output = output_parser.parse(result)["Graph"]
        print(parsed_output)
        answer_info = data["Original"]["answer_info"]
        options = {
            "ans0": data["Original"]["ans0"],
            "ans1": data["Original"]["ans1"],
            "ans2": data["Original"]["ans2"],
        }
        computed_label = bbq_label_compute(parsed_output, answer_info, options)
        print("computed_label" + str(computed_label))
        if computed_label == data["Original"]["label"]:
            print("Success")
            data["Graph"] = parsed_output
            with open(args.output_file, "w") as f:
                for item in output_data:
                    json_item = json.dumps(item)
                    f.write(json_item + "\n")
        else:
            print("Not match")
            print("Computed Label: " + str(computed_label))
            print("True Label: " + str(data["Original"]["label"]))
        time.sleep(3)
