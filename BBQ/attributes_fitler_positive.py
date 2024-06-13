import os
from langchain.callbacks import get_openai_callback
import json
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import argparse
from utils import prepare_llm
from prompt import prompt_attributes_filter

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
    default="bbq_positve_attributes.json",
    type=str,
)
parser.add_argument(
    "--output_file",
    default="bbq_positve_attributes_filtered.json",
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
            data["Positive Attributes Filtered"] = {}
            for non_unknown_idx, _ in data["Positive Attributes"].items():
                data["Positive Attributes Filtered"][non_unknown_idx] = []
            output_data.append(data)


template = prompt_attributes_filter
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
        "attributes",
    ],
)


with get_openai_callback() as cb:
    for i, data in enumerate(output_data):
        print(data["Positive Attributes Filtered"].keys())
        for non_unknown_idx, attributes_values in data[
            "Positive Attributes Filtered"
        ].items():
            if len(attributes_values) != 0:
                print("skip")
                continue
            initial_attributes = data["Positive Attributes"][non_unknown_idx]
            _input = prompt.format_prompt(
                context=data["Original"]["context"],
                graph=str(data["Graph"]),
                question=data["Original"]["question"],
                attributes=initial_attributes,
            )
            result = llm.invoke(_input.to_string())
            print("*************")
            print(result)
            if "'Attributes'" in result:
                result = result.replace("'Attributes'", '"Attributes"')
            parsed_output = output_parser.parse(result)["Attributes"]
            if len(parsed_output) >= 3:
                data["Positive Attributes Filtered"][non_unknown_idx] = parsed_output
            with open(args.output_file, "w") as f:
                for item in output_data:
                    json_item = json.dumps(item)
                    f.write(json_item + "\n")
