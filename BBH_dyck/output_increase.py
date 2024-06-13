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
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--increase_output",
    default=0.25,
    type=float,
)
parser.add_argument(
    "--output_file",
    default="dyck_languages_increase_output_0.25.json",
    type=str,
)
parser.add_argument(
    "--orginal_file",
    default="dyck_languages.json",
    type=str,
)

args = parser.parse_args()


def extract_input(text):
    # Find the position of "Input:"
    pos = text.find("Input:")
    if pos == -1:
        return "Input: not found"

    # Extract everything after "Input:"
    extracted_text = text[pos + len("Input:") :].strip()

    # Remove all spaces
    no_spaces = extracted_text.replace(" ", "")

    return no_spaces


def generate_valid_bracket_sequence(n):
    if n % 2 != 0:
        return "Error: n must be even to form valid pairs."

    bracket_map = {"(": ")", "[": "]", "{": "}", "<": ">"}
    stack = []
    sequence = []

    for _ in range(n // 2):
        open_bracket = random.choice(list(bracket_map.keys()))
        sequence.append(open_bracket)
        stack.append(bracket_map[open_bracket])

    random.shuffle(stack)  # Shuffle to introduce randomness in closing order
    sequence.extend(stack)  # Close all opened brackets

    return "".join(sequence)


def generate_bracket_sequence_with_cutoff(n, m):
    if n % 2 != 0:
        return "Error: n must be even to form valid pairs."

    valid = False
    while not valid:
        bracket_map = {"(": ")", "[": "]", "{": "}", "<": ">"}
        stack = []
        sequence = []
        open_brackets = list(bracket_map.keys())

        # Generate the initial part ensuring it has unmatched open brackets at the end
        while len(sequence) < n - m:
            if not stack or random.choice([True, False]):
                open_bracket = random.choice(open_brackets)
                stack.append(open_bracket)
                sequence.append(open_bracket)
            else:
                sequence.append(bracket_map[stack.pop()])

        # Add extra unmatched open brackets if needed
        extra_opens_needed = (m + 1) // 2  # Calculate needed opens for an odd m
        while len(stack) < extra_opens_needed:
            open_bracket = random.choice(open_brackets)
            stack.append(open_bracket)
            sequence.append(open_bracket)

        ending = [bracket_map[stack.pop()] for _ in range(len(stack))]

        full_sequence = sequence + ending
        cutoff_sequence = sequence
        exact_ending = ending

        # Check if the lengths are correct
        if (
            len(full_sequence) == n
            and len(cutoff_sequence) == n - m
            and len(exact_ending) == m
        ):
            valid = True

    return "".join(full_sequence), "".join(cutoff_sequence), "".join(exact_ending)


def count_br(str):
    list_of_br = ["(", ")", "[", "]", "{", "}", "<", ">"]
    count = 0
    for char in str:
        if char in list_of_br:
            count += 1
    return count


if os.path.exists(args.output_file):
    output_data = []
    with open(args.output_file, "r") as file:
        for line in file:
            data = json.loads(line)
            output_data.append(data)
else:
    output_data = []
    with open(args.orginal_file, "r") as file:
        data = json.load(file)
    for data_dic in data["examples"]:
        output_data.append(data_dic)


for i, data in enumerate(output_data):
    input_br = extract_input(data["input"])
    output_br = data["target"].replace(" ", "")
    input_count = count_br(input_br)
    output_count = count_br(output_br)
    if input_count == output_count:
        full, input_br, output_br = generate_bracket_sequence_with_cutoff(
            input_count + output_count, output_count
        )
        data["Actual Output Increase"] = 0
    else:
        increase = max(int((input_count - output_count) * args.increase_output), 1)
        if increase % 2 != 0:
            increase += 1
        if output_count + increase > input_count:
            full, input_br, output_br = generate_bracket_sequence_with_cutoff(
                input_count + input_count, input_count
            )
            data["Actual Output Increase"] = input_count - output_count
        else:
            full, input_br, output_br = generate_bracket_sequence_with_cutoff(
                input_count + output_count + increase, output_count + increase
            )
            data["Actual Output Increase"] = increase
    data["Updated Input"] = input_br
    data["Updated Target"] = output_br


with open(args.output_file, "w") as f:
    for item in output_data:
        json_item = json.dumps(item)
        f.write(json_item + "\n")
