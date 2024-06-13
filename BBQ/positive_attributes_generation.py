import os
from langchain.callbacks import get_openai_callback
import json
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import argparse
from utils import prepare_llm, overlap_match
from prompt import prompt_attributes_positive

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    default="gpt-4-1106",
    choices=["gpt-4-1106"],
    type=str,
)
parser.add_argument(
    "--temperature",
    default=1,
    type=float,
)
parser.add_argument(
    "--max_token",
    default=2048,
    type=float,
)
parser.add_argument(
    "--original_file",
    default="bbq_graph.json",
    type=str,
)
parser.add_argument(
    "--output_file",
    default="bbq_positve_attributes.json",
    type=str,
)

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
    with open(args.original_file, "r") as file:
        for line in file:
            data = json.loads(line)
            non_unknown_idxs = []
            for idx, value in data["Original"]["answer_info"].items():
                if value[1] != "unknown":
                    non_unknown_idxs.append(idx)
            data["Positive Attributes"] = {}
            for non_unknown_idx in non_unknown_idxs:
                data["Positive Attributes"][non_unknown_idx] = []
            output_data.append(data)


template = prompt_attributes_positive
response_schemas = [
    ResponseSchema(name="Attributes", description="", type="dictionary"),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
prompt = PromptTemplate(
    template=template,
    input_variables=[
        "context",
        "graph",
        "question",
        "person",
    ],
)


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


def find_attrbutes_from_node_name(graph, person_node_name):
    person_node_id = [
        node["id"] for node in graph["nodes"] if node["content"] == person_node_name
    ]
    if len(person_node_id) != 1:
        return None

    person_node_id = person_node_id[0]

    attributes_id_list = []
    for edge in graph["edges"]:
        if edge["source"] == person_node_id and edge["type"] == "to_attribute":
            attributes_id = edge["target"]
            attributes_id_list.append(attributes_id)

    return [
        node["content"] for node in graph["nodes"] if node["id"] in attributes_id_list
    ]


with get_openai_callback() as cb:
    for i, data in enumerate(output_data):
        print(data["Positive Attributes"].keys())
        for non_unknown_idx, attributes_values in data["Positive Attributes"].items():
            if len(attributes_values) != 0:
                print("skip")
                continue
            node_name = find_node_name_from_idx(
                data["Graph"],
                data["Original"]["answer_info"][non_unknown_idx][0],
                data["Original"][non_unknown_idx],
            )
            _input = prompt.format_prompt(
                context=data["Original"]["context"],
                graph=str(data["Graph"]),
                question=data["Original"]["question"],
                person=node_name,
            )
            print(_input.to_string())
            result = llm.invoke(_input.to_string())
            print("*************")
            print(result)
            if "'Attributes'" in result:
                result = result.replace("'Attributes'", '"Attributes"')
            parsed_output = output_parser.parse(result)["Attributes"]
            data["Positive Attributes"][non_unknown_idx] = parsed_output
            with open(args.output_file, "w") as f:
                for item in output_data:
                    json_item = json.dumps(item)
                    f.write(json_item + "\n")
