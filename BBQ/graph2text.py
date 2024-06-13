import os
from langchain.callbacks import get_openai_callback
import json
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import argparse
from utils import prepare_llm
from prompt import prompt_graph2text
from pprint import pprint
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    default="gpt-4-1106",
    choices=["gpt-4-1106"],
    type=str,
)
parser.add_argument(
    "--temperature",
    default=0.7,
    type=float,
)
parser.add_argument(
    "--max_token",
    default=2048,
    type=float,
)
parser.add_argument(
    "--original_file",
    default="bbq_500_polarity_increase_1_new.json",
    type=str,
)
parser.add_argument(
    "--output_file",
    default="bbq_500_polarity_increase_1_text.json",
    type=str,
)
parser.add_argument(
    "--update_graph_key",
    default="Updated_Graph_increase_1",
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
            output_data.append(data)


template = prompt_graph2text
response_schemas = [
    ResponseSchema(name="Sentences", description="", type="dictionary"),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
prompt = PromptTemplate(
    template=template,
    input_variables=["original_graph", "updated_graph", "original_context"],
)


def remove_label_information(graph):
    # Find all label nodes
    for node in graph["nodes"]:
        if "question_relation" in node.keys():
            node.pop("question_relation")

    return graph


with get_openai_callback() as cb:
    for i, data in enumerate((output_data)):
        if "Updated Context" in data.keys():
            print("skip")
            continue
        print("index:" + str(i))
        updated_graph = deepcopy(data[args.update_graph_key])
        print("******************************")
        pprint("Updated Graph" + data[args.update_graph_key])
        _input = prompt.format_prompt(
            original_graph=str(data["Graph"]),
            original_context=data["Original"]["context"],
            updated_graph=remove_label_information(updated_graph),
        )
        result = llm.invoke(_input.to_string())
        parsed_output = output_parser.parse(result)["Sentences"]
        data["Updated Context"] = parsed_output
        print("Updated Context: " + parsed_output)
        print("******************************")
        with open(args.output_file, "w") as f:
            for item in output_data:
                json_item = json.dumps(item)
                f.write(json_item + "\n")
