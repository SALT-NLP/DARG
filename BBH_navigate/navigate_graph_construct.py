import os
from langchain.callbacks import get_openai_callback
import json
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import time
import argparse
from prompt import navigate_graph_construct
from utils import prepare_llm, returns_to_start

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
    default="navigate_graph.json",
    type=str,
)
parser.add_argument(
    "--original_file",
    default="navigate.json",
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
        data = json.load(file)
    for data_dic in data["examples"]:
        output_data.append({"Original": data_dic, "Graph": {}})

template = navigate_graph_construct
response_schemas = [
    ResponseSchema(
        name="Graph", description="the dictionary of the Graph", type="dictionary"
    ),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template=template,
    input_variables=["instruction"],
    partial_variables={"format_instructions": format_instructions},
)
with get_openai_callback() as cb:
    for i, data in enumerate(output_data):
        if len(data["Graph"].keys()) != 0:
            print("skip")
            continue
        print("index:" + str(i))
        _input = prompt.format_prompt(
            instruction=extract_instructions(data["Original"]["input"]),
        )
        result = llm.invoke(_input.to_string())
        parsed_output = output_parser.parse(result)["Graph"]
        computed_label = returns_to_start(parsed_output)
        print("computed_label" + str(computed_label))
        if computed_label == data["Original"]["target"]:
            print("Success")
            data["Graph"] = parsed_output
            with open(args.output_file, "w") as f:
                for item in output_data:
                    json_item = json.dumps(item)
                    f.write(json_item + "\n")
        else:
            print("Not match")
            print("Computed Label: " + str(computed_label))
            print("True Label: " + str(data["Original"]["target"]))
        time.sleep(3)
