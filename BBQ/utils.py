from langchain.llms import AzureOpenAI
import openai
import os
from copy import deepcopy

openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_KEY")


def prepare_llm(
    model_name: str, engine: str, temperature: float, max_tokens: int, top_p: float
):
    """
    Prepares an instance of the AzureOpenAI LLM.

    Args:
        model_name (str): The name of the model to use (e.g., gpt-4-1106).
        engine (str): The name of the engine to use (usually the same as model_name).
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum number of tokens to generate.
        top_p (float): Top-p (nucleus) sampling.

    Returns:
        LLM: An instance of the AzureOpenAI LLM configured with the specified parameters.
    """
    # The function to return a langchain LLM instance.
    # You can replace it with any other Langchain LLMs which support llm.invoke(_input.to_string())
    # to generate an output given a prompt as the input to the LLM.

    # You may also not use Langchain LLM. If so, you should modify the places for inference accordingly.

    try:
        llm = AzureOpenAI(
            model_name=model_name,
            engine=engine,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return llm
    except Exception as e:
        print(f"Error preparing the LLM: {e}")
        raise


def overlap_match(str1, str2, threshold=0.8):
    # Convert both strings to lower case
    str1, str2 = str1.lower().replace(" ", ""), str2.lower().replace(" ", "")

    # Helper function to find longest common substring
    def longest_common_substring(s1, s2):
        m = len(s1)
        n = len(s2)
        # Create a 2D array to store lengths of longest common suffixes
        lcs = [[0] * (n + 1) for _ in range(m + 1)]
        longest = 0  # To store length of the longest common substring

        # Building the LCS array
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    lcs[i][j] = lcs[i - 1][j - 1] + 1
                    longest = max(longest, lcs[i][j])
                else:
                    lcs[i][j] = 0
        return longest

    # Find the longest common substring length
    lcs_length = longest_common_substring(str1, str2)

    # Calculate the minimum length needed to be considered a match
    min_length_needed = int(min(len(str1), len(str2)) * threshold)

    # Determine if the overlap is sufficient
    return lcs_length >= min_length_needed


def bbq_label_compute(graph, answer_info, options):
    # Extract nodes and edges from the graph
    nodes = {node["id"]: node for node in graph["nodes"]}
    edges = graph["edges"]

    # Identify the two person nodes
    person_nodes = [node["id"] for node in graph["nodes"] if node["type"] == "person"]
    person_nodes_id_content_map = {}
    for node in graph["nodes"]:
        if node["type"] == "person":
            person_nodes_id_content_map[node["id"]] = node["content"]

    # There should be exactly two person nodes, otherwise return None
    if len(person_nodes) != 2:
        return None

    # Initialize a dictionary to keep track of person-to-attribute relationships
    person_to_attributes = {person_node: [] for person_node in person_nodes}

    for edge in edges:
        if edge["type"] == "to_attribute":
            person_id = edge["source"]
            attribute_id = edge["target"]
            attribute_info = nodes[attribute_id]
            person_to_attributes[person_id].append(attribute_info)

    # Determine potential answers based on attributes' question_relation
    # Possible answers will store if person ID is answer True or False
    possible_answers = {}

    for person_id, attributes in person_to_attributes.items():
        for attribute in attributes:
            relation = attribute["question_relation"]
            if relation == "positive":
                if (
                    person_id in possible_answers
                    and possible_answers[person_id] == False
                ):
                    return None  # Contradiction found
                possible_answers[person_id] = True
            elif relation == "negative":
                # Find the other person ID
                other_person_id = (
                    person_nodes[1] if person_id == person_nodes[0] else person_nodes[0]
                )
                if (
                    other_person_id in possible_answers
                    and possible_answers[other_person_id] == False
                ):
                    return None  # Contradiction found
                possible_answers[other_person_id] = (
                    True  # Set the other person as the answer
                )

    person_label_connection = {}
    for person_id, _ in person_to_attributes.items():
        if person_id in possible_answers.keys():
            person_label_connection[person_nodes_id_content_map[person_id].lower()] = (
                possible_answers[person_id]
            )
        else:
            person_label_connection[person_nodes_id_content_map[person_id].lower()] = (
                False
            )

    non_unknown_idxs = []
    unknown_idx = None
    list_person_node_names = list(person_label_connection.keys())
    list_person_names_options = list(options.values())
    list_person_names_answer_info = []

    for ans_index, answer_content in answer_info.items():
        if answer_content[1] == "unknown":
            unknown_idx = ans_index
        else:
            non_unknown_idxs.append(ans_index)
            answer_name = answer_content[0].lower()
            list_person_names_answer_info.append(answer_name)

    flag_person_name_match = False
    person_label_connection_extended = deepcopy(person_label_connection)

    count_num_inside = 0
    for person_name in list_person_names_answer_info:
        for person_node_name in list_person_node_names:
            if (
                person_name.lower() in person_node_name.lower().split(" ")
                or person_name.lower() == person_node_name.lower()
            ):
                person_label_connection_extended[person_name] = person_label_connection[
                    person_node_name
                ]
                count_num_inside += 1
                break
    if count_num_inside == len(list_person_names_answer_info):
        flag_person_name_match = True
    else:
        for person_node_name in list_person_node_names:
            for person_name in (
                list_person_names_answer_info + list_person_names_options
            ):
                # man and women could be overlapped
                if overlap_match(person_node_name, person_name):
                    print(person_node_name, person_name)
                    flag_person_name_match = True
                    person_label_connection_extended[person_name] = (
                        person_label_connection[person_node_name]
                    )
    if not flag_person_name_match:
        return None
    true_count = sum(person_label_connection.values())
    if true_count == 1:
        for non_unknown_idx in non_unknown_idxs:
            if (
                person_label_connection_extended[
                    answer_info[non_unknown_idx][0].lower()
                ]
                == True
            ):
                return int(non_unknown_idx.replace("ans", ""))
    elif true_count == 0:
        return int(unknown_idx.replace("ans", ""))
    else:
        return "mul_ans"
