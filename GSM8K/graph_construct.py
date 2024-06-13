import openai
from langchain.llms import AzureOpenAI
import os
from langchain.callbacks import get_openai_callback
import json
from langchain.prompts import PromptTemplate
import re
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import threading
import time

import argparse
from utils import (
    prepare_llm,
    process_data_and_extract_equations,
    extract_final_answer,
    create_computational_graph,
    compute_graph_values,
)
from prompt import template_graph_construct

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
    default="gsm8k_graph.json",
    type=str,
)
parser.add_argument(
    "--original_file",
    default="test.jsonl",
    type=str,
)


args = parser.parse_args()


def contains_letters(s):
    modified_string = re.sub(r"\d+\s*x\s*\d+", "", s)
    return bool(re.search(r"[a-zA-Z]", modified_string))


def check_equation_content(mapping):
    list_content = [mapping[equation]["content"] for equation in mapping.keys()]
    if any(contains_letters(c) for c in list_content):
        return False
    else:
        return True


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
        output_data.append({"Original": data, "Mapping": {}})

template = template_graph_construct
response_schemas = [
    ResponseSchema(
        name="Mapping", description="the dictionary of the mapping", type="dictionary"
    ),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template=template,
    input_variables=["question", "multiple_equation"],
    partial_variables={"format_instructions": format_instructions},
)

for i, data in enumerate(output_data):
    if len(data["Mapping"].keys()) != 0:
        list_content = [
            data["Mapping"][equation]["content"] for equation in data["Mapping"].keys()
        ]
        if any(contains_letters(c) for c in list_content):
            print(list_content)
            data["Mapping"] = {}

with get_openai_callback() as cb:
    for i, data in enumerate(output_data):
        if len(data["Mapping"].keys()) != 0:
            print("skip")
            continue
        print("index:" + str(i))
        multiple_equation, temp_answer = process_data_and_extract_equations(
            data["Original"]
        )
        temp_data = {"question": data["Original"]["question"], "answer": temp_answer}
        final_answer = float(extract_final_answer(temp_data))
        _input = prompt.format_prompt(
            question=temp_data, multiple_equation=multiple_equation
        )
        print("***********")
        print("Input Prompt" + _input.to_string())
        result = llm.invoke(_input.to_string())
        print("LLM Output" + result)
        print("***********")
        parsed_output = output_parser.parse(result)["Mapping"]

        if not check_equation_content(parsed_output):
            continue
        graph = create_computational_graph(parsed_output)
        for node_name, node_attribute in graph["nodes"].items():
            if node_attribute["type"] != "initial":
                graph["nodes"][node_name].pop("value")
        flag = True
        try:
            values = compute_graph_values(graph)
        except Exception as e:
            flag = False
            data["Mapping"] = {}
            print("Failed")
        if values == None:
            print("time out")
            flag = False
            data["Mapping"] = {}
        if flag:
            for node_name, value in values.items():
                graph["nodes"][node_name]["value"] = value
            for node_name, node_attribute in graph["nodes"].items():
                if node_attribute["type"] == "final":
                    if (float(node_attribute["value"]) - final_answer) < 1e-6:
                        data["Mapping"] = parsed_output
                        print("Succeed")
                        data["Original"]["answer"] = temp_answer
                        break
                    else:
                        print("Not Match")
                        data["Mapping"] = {}
                        break
        with open(args.output_file, "w") as f:
            for item in output_data:
                json_item = json.dumps(item)
                f.write(json_item + "\n")
    print(cb)
