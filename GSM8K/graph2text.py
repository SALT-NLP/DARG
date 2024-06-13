import json
import argparse
import os
from langchain.callbacks import get_openai_callback
import json
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import time
from utils import (
    create_computational_graph,
    compute_graph_values,
    prepare_llm,
    computational_graph_to_mapping,
    initialize_code_agent,
)
from prompt import (
    template_graph2text_intermediate,
    template_code,
    improvement_to_use,
    gsm8k_validate_2,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_file",
    default="depth_increase_2_text.json",
    type=str,
)
parser.add_argument(
    "--original_file",
    default="depth_increase_2.json",
    type=str,
)
parser.add_argument(
    "--model_name",
    default="gpt-4-1106",
    choices=["gpt-4-1106"],
    type=str,
)
parser.add_argument(
    "--updated_graph_entry",
    default="graph",
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
    "--max_iter_phase1",
    default=3,
    type=int,
)

parser.add_argument(
    "--max_iter_phase2",
    default=3,
    type=int,
)
parser.add_argument(
    "--max_iter_validate",
    default=3,
    type=int,
)


args = parser.parse_args()


def format_equations_with_types(mapping_dict):
    # Initialize an empty string to store the final output
    output_string = ""

    # Iterate through each equation in the mapping
    for equation_name, equation_details in mapping_dict.items():
        # Extract the content of the equation
        equation_content = equation_details["content"]

        # Build the string for the current equation
        equation_str = f"{equation_name}: {equation_content}\n"

        # Add details about each operator and the result, including their types
        for key in ["operator 1", "operator 2", "result"]:
            name = equation_details[key]["Name"]
            value = equation_details[key]["value"]
            node_type = equation_details[key]["type"]
            equation_str += f"    {name} ({node_type}) = {value}\n"

        # Add this equation's string to the final output string
        output_string += equation_str + "\n"

    return output_string


def exact_match_number_in_words(number, string):
    # Ensure number is treated as float for consistent method access
    string = string.replace("$", "")
    number = float(number)

    # Convert the number to its different string representations
    int_representation = str(int(number))
    float_representation = (
        "{:.1f}".format(number) if not number.is_integer() else str(int(number)) + ".0"
    )
    formatted_representation = f"{int(number):,}"

    # Split the string into words and normalize each word by removing commas for comparison
    words = string.split()
    normalized_words = [word.replace(",", "") for word in words]
    for i, word in enumerate(normalized_words):
        if word[-1] == ".":
            normalized_words[i] = word[0:-1]

    # Check if any word matches the number representations exactly
    matches = any(
        word in [int_representation, float_representation, formatted_representation]
        for word in normalized_words
    )

    return matches


def if_intermediate_value_in(values, graph, candidate_question):
    # We avoid the caces where imtermediate results directly appear in the generated questions.

    list_initial_values = []
    for node, value in values.items():
        if graph["nodes"][node]["type"] == "initial":
            list_initial_values.append(float(value))
    flag_contain_intermediate_value = False
    for node, value in values.items():
        if (
            graph["nodes"][node]["type"] != "initial"
            and float(value) not in list_initial_values
        ):
            flag_contain_intermediate_value = exact_match_number_in_words(
                value, candidate_question
            )
            if flag_contain_intermediate_value:
                return True
    return flag_contain_intermediate_value


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


def validate_question(llm, prompt_template, probelm, code, code_output):
    # Apply a majority vote to further ensure the quality of the generated question

    template_validate = prompt_template
    response_schemas_validate = [
        ResponseSchema(
            name="If_match",
            description="Yes or No. If the math problem is valide, the code's logic align with the solving process.",
            type="str",
        ),
    ]
    output_parser_validate = StructuredOutputParser.from_response_schemas(
        response_schemas_validate
    )
    format_instructions_validate = output_parser_validate.get_format_instructions()
    prompt_validate = PromptTemplate(
        template=template_validate,
        input_variables=["new_probelm", "code", "code_output"],
        partial_variables={"format_instructions": format_instructions_validate},
    )

    _input_validate = prompt_validate.format_prompt(
        new_probelm=probelm, code=code, code_output=code_output
    )
    list_vote = []
    print("\033[91m" + "Begin Validating" + "\033[0m")
    for _ in range(args.max_iter_validate):
        validate_result = llm.invoke(_input_validate.to_string())
        print(validate_result)
        if_match_index = validate_result.find("If_match")
        if if_match_index != -1:
            yes_index = validate_result.find("Yes", if_match_index)
            if yes_index != -1:
                list_vote.append(1)
            else:
                list_vote.append(0)
    if sum(list_vote) > int(args.max_iter_validate / 2):
        print("\033[91m" + "Pass the valdation" + "\033[0m")
        return True
    else:
        print("\033[91m" + "Fail to pass the valdation" + "\033[0m")
        return False


llm = prepare_llm(
    model_name=args.model_name,
    engine=args.model_name,
    max_tokens=args.max_token,
    temperature=args.temperature,
    top_p=0.95,
)
llm_3 = prepare_llm(
    model_name=args.model_name,
    engine=args.model_name,
    max_tokens=args.max_token,
    temperature=args.temperature,
    top_p=0.95,
)
llm_3 = prepare_llm(
    model_name=args.model_name,
    engine=args.model_name,
    max_tokens=args.max_token,
    temperature=args.temperature,
    top_p=0.95,
)

template = template_graph2text_intermediate
response_schemas = [
    ResponseSchema(
        name="math_problem",
        description="the math problem that aligns with the above equations, please try to make it concise while not introducing any ambiguity",
        type="str",
    ),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate(
    template=template,
    input_variables=["original_mapping", "original_problem", "new_mapping"],
    partial_variables={"format_instructions": format_instructions},
)

prompt_improve = PromptTemplate(
    template=improvement_to_use,
    input_variables=[
        "original_problem",
        "code",
        "code_output",
        "equations",
    ],
    partial_variables={"format_instructions": format_instructions},
)

with get_openai_callback() as cb:
    for i, data in enumerate((output_data[0:200])):
        print(i)
        if args.updated_graph_entry not in data.keys():
            continue
        if "Updated Question" in data.keys():
            print("Skip")
            continue
        original_graph = create_computational_graph(data["Mapping"])
        original_mapping = data["Mapping"]
        updated_graph = data[args.updated_graph_entry]
        values = compute_graph_values(updated_graph)
        for node, value in values.items():
            if updated_graph["nodes"][node]["type"] == "final":
                label_value = value
                break
        for node, value in values.items():
            updated_graph["nodes"][node]["value"] = values[node]

        updated_mapping = computational_graph_to_mapping(updated_graph)
        _input = prompt.format_prompt(
            original_problem=data["Original"]["question"],
            original_mapping=format_equations_with_types(original_mapping),
            new_mapping=format_equations_with_types(updated_mapping),
        )
        print(_input.to_string())
        for _ in range(args.max_iter_phase1):
            result = llm.invoke(_input.to_string())
            time.sleep(1)
            candidate_question = output_parser.parse(result)["math_problem"]
            if if_intermediate_value_in(values, updated_graph, candidate_question):
                continue
            if_good_problem = False
            for j in range(args.max_iter_phase2):
                if if_intermediate_value_in(values, updated_graph, candidate_question):
                    continue
                agent_executor = initialize_code_agent(
                    llm=llm, prompt_template=template_code
                )
                agent_iter = agent_executor.iter({"question": candidate_question})
                prediction = None
                for step in agent_iter:
                    if output := step.get("intermediate_step"):
                        action, observation = output[0]
                        if (
                            str(label_value) in observation
                            or str(int(label_value)) in observation
                            or str(float(label_value)) in observation
                        ):
                            if_match = validate_question(
                                llm,
                                gsm8k_validate_2,
                                candidate_question,
                                code=action.tool_input,
                                code_output=observation,
                            )
                            if if_match and not if_intermediate_value_in(
                                values, updated_graph, candidate_question
                            ):
                                prediction = observation
                            else:
                                prediction = None
                        print(observation)
                        break
                time.sleep(1)
                if prediction != None:
                    if_good_problem = True
                    break
                prompt_improve = PromptTemplate(
                    template=improvement_to_use,
                    input_variables=[
                        "original_problem",
                        "code",
                        "code_output",
                        "equations",
                    ],
                    partial_variables={"format_instructions": format_instructions},
                )
                _input_improve = prompt_improve.format_prompt(
                    original_problem=candidate_question,
                    code=action.tool_input,
                    code_output=observation,
                    equations=format_equations_with_types(updated_mapping),
                )
                result_improve = llm.invoke(_input_improve.to_string())
                candidate_question = output_parser.parse(result_improve)["math_problem"]
                if if_intermediate_value_in(values, updated_graph, candidate_question):
                    continue
            if if_good_problem:
                break
        if if_good_problem == True and not if_intermediate_value_in(
            values, updated_graph, candidate_question
        ):
            data["Updated Question"] = candidate_question
            with open(args.output_file, "w") as f:
                for item in output_data:
                    json_item = json.dumps(item)
                    f.write(json_item + "\n")

print(cb)
