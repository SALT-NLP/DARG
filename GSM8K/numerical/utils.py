import openai
import os
import re
import threading
import time
import decimal
from typing import List, Union
from langchain.llms import AzureOpenAI
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from langchain.agents import (
    Tool,
    AgentExecutor,
    LLMSingleActionAgent,
    AgentOutputParser,
)
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain_experimental.tools import PythonREPLTool

decimal.getcontext().prec = 50

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


def process_data_and_extract_equations(data):
    """
    Process the data by converting percentages to decimals in the answer and then
    extract equations with multiple operators from the answer,
    excluding those that contain fractions found in the question.
    """
    question = data["question"]
    answer = data["answer"]
    # Convert percentages in the answer to decimals
    percentages = re.findall(r"(\d+(?:\.\d+)?)%", question)

    bracket_contents = re.findall(r"<<([^<>]*)>>|\$(.*?)\$", answer)

    for perc in percentages:
        # Calculate the decimal equivalent
        decimal_equivalent = float(perc) / 100

        for i, content in enumerate(bracket_contents):
            if content:  # Check if content is not None
                content = (
                    content[0] if content[0] else content[1]
                )  # Select the non-empty match
                # Replace the specific fraction with its decimal equivalent in the content
                if f"{perc}/100" in content:
                    updated_content = content.replace(
                        f"{perc}/100", str(decimal_equivalent)
                    )
                    bracket_contents[i] = (updated_content, updated_content)

    # Reconstruct the answer with updated content
    def replace_content(match):
        content = bracket_contents.pop(0)
        return (
            f"<<{content[0]}>>"
            if match.group(0).startswith("<<")
            else f"${content[1]}$"
        )

    answer = re.sub(r"<<[^<>]*>>|\$(.*?)\$", replace_content, answer)
    # Find fractions in the question
    fractions = re.findall(r"\b\d+/\d+\b", question)

    # Regular expression to find equations inside <<>>
    equations = re.findall(r"<<([^<>]+)>>", answer)

    # Filter equations with multiple operators
    mult_operator_equations = []
    for eq in equations:
        operators = re.findall(r"[\+\-\*\/]", eq)
        for frac in fractions:
            if frac in eq and "/" in operators:
                operators.remove("/")
        if len(operators) > 1:
            mult_operator_equations.append(eq)

    return mult_operator_equations, answer


def extract_final_answer(input_dict):

    # Extracting the final answer after "####"
    final_answer_match = re.search(r"####\s*(.*)", input_dict["answer"])
    final_answer_text = (
        final_answer_match.group(1) if final_answer_match else "No final answer found"
    )

    return final_answer_text


def create_computational_graph(mapping):
    nodes = {}
    edges = {}
    edge_count = 1

    def update_node_info(node_name, node_type, node_value=None):
        if node_name in nodes:
            nodes[node_name]["type"] = node_type
        else:
            nodes[node_name] = {
                "type": node_type,
                "value": node_value,
                "incoming_edges": [],
                "outgoing_edges": [],
            }

    def add_edge(from_node, to_node, operation, objective):
        nonlocal edge_count
        edge_key = str(edge_count)
        edges[edge_key] = {
            "from": from_node,
            "to": to_node,
            "operation": operation,
            "objective": objective,
        }
        nodes[from_node]["outgoing_edges"].append(edge_key)
        nodes[to_node]["incoming_edges"].append(edge_key)
        edge_count += 1

    for equation in mapping.values():
        # Update node information
        operator1 = equation["operator 1"]["Name"]
        operator2 = equation["operator 2"]["Name"]
        result = equation["result"]["Name"]

        update_node_info(
            operator1,
            equation["operator 1"]["type"],
            equation["operator 1"].get("value"),
        )
        update_node_info(
            operator2,
            equation["operator 2"]["type"],
            equation["operator 2"].get("value"),
        )
        update_node_info(
            result, equation["result"]["type"], equation["result"].get("value")
        )

        # Determine the operation
        content = equation["content"]

        operation = (
            "mul"
            if ("*" in content or "x" in content)
            else "div" if "/" in content else "add" if "+" in content else "sub"
        )

        # Add edges
        add_edge(operator1, result, operation, operator2)
        if operation in ["mul", "add"]:
            add_edge(operator2, result, operation, operator1)
        else:  # For non-commutative operations
            reciprocal_operation = "div_by" if operation == "div" else "sub_by"
            add_edge(operator2, result, reciprocal_operation, operator1)

    return {"nodes": nodes, "edges": edges}


def compute_graph_values(graph):
    # Function to perform the operation based on the type
    start_time = time.time()

    def perform_operation(op, a, b, decimal_places=10):
        # Return None if any operand is None
        if a is None or b is None:
            return None

        # Perform the operation based on the type specified
        if op == "mul":
            result = a * b
        elif op == "div":
            if b == 0:
                return None
            else:
                result = a / b
        elif op == "div_by":
            if a == 0:
                return None
            else:
                result = b / a
        elif op == "add":
            result = a + b
        elif op == "sub":
            result = a - b
        elif op == "sub_by" or op == "sub_from":
            result = b - a
        else:
            raise ValueError(f"Unknown operation: {op}")

        nearest_int = round(result)
        if abs(result - nearest_int) < 1e-4:
            result = nearest_int

        # Round the result to the specified number of decimal places
        return round(result, decimal_places)

    # Storing the values of each node
    values = {}

    # First, assign values to initial nodes
    for node, attrs in graph["nodes"].items():
        if attrs["type"] == "initial":
            values[node] = attrs["value"]

    # Compute values for intermediate and final nodes
    # Keep iterating until all nodes have values
    while len(values) < len(graph["nodes"]):
        if time.time() - start_time > 30:
            values = None
            return values
        for edge_id, edge_attrs in graph["edges"].items():
            if time.time() - start_time > 30:
                values = None
                return values
            from_node = edge_attrs["from"]
            to_node = edge_attrs["to"]
            operation = edge_attrs["operation"]
            objective_node = edge_attrs["objective"]

            # Check if the nodes involved in the operation have values
            if (
                from_node in values
                and objective_node in values
                and to_node not in values
            ):
                values[to_node] = perform_operation(
                    operation, values[from_node], values[objective_node]
                )

    return values


def compute_graph_values_with_timeout(graph, timeout=30):
    result = {"values": None, "error": None}

    def target():
        try:
            result["values"] = compute_graph_values(graph)
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        thread.join()  # Ensure the thread finishes execution
        raise TimeoutError("Function exceeded the maximum allowed time")

    if result["error"]:
        raise result["error"]

    return result["values"]


def computational_graph_to_mapping(graph):
    nodes = graph["nodes"]
    edges = graph["edges"]
    mapping = {}
    processed_results = set()

    # Helper to determine if a node is initial (no incoming edges)
    def is_initial(node_name):
        return len(nodes[node_name]["incoming_edges"]) == 0

    # Helper to reconstruct equation content
    def make_equation_content(op1, op2, result, operation):
        operations = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "div": "/",
            "sub_by": "by -",
            "div_by": "by /",
        }
        symbol = operations.get(operation, "?")
        if operation in ["sub_by", "div_by"]:
            return f"{op2} {symbol} {op1} = {result}"  # Adjust the order for these operations
        else:
            return f"{op1} {symbol} {op2} = {result}"

    # Ensure equations are created in a logical order
    dependency_levels = {node: 0 for node in nodes}

    def update_dependency_levels(node, level):
        if dependency_levels[node] < level:
            dependency_levels[node] = level
        for edge_id in nodes[node]["outgoing_edges"]:
            next_node = edges[edge_id]["to"]
            update_dependency_levels(next_node, level + 1)

    # Initialize dependency levels
    for node in nodes:
        if is_initial(node):
            update_dependency_levels(node, 0)

    # Sort nodes by their dependency level, then alphabetically
    sorted_nodes = sorted(nodes.keys(), key=lambda x: (dependency_levels[x], x))
    equation_number = 1
    for node in sorted_nodes:
        # Skip initial nodes, as they don't form equations by themselves
        if is_initial(node):
            continue

        if node in processed_results:
            continue  # Avoid processing a result node more than once

        incoming_edges = nodes[node]["incoming_edges"]
        for edge_id in incoming_edges:
            edge = edges[edge_id]
            operation = edge["operation"]
            from_node = edge["from"]
            objective = edge["objective"]

            if operation in ["add", "mul"] and from_node > objective:
                continue  # Avoid duplication for commutative operations

            # Extract values or node names for equation content
            from_value = nodes[from_node].get("value", from_node)
            objective_value = nodes[objective].get("value", objective)
            result_value = nodes[node].get("value", node)

            # Construct equation content
            equation_content = make_equation_content(
                from_value, objective_value, result_value, operation
            )

            # Populate the mapping
            mapping[f"Equation{equation_number}"] = {
                "content": equation_content,
                "operator 1": {
                    "Name": from_node,
                    "type": nodes[from_node]["type"],
                    "value": nodes[from_node].get("value"),
                },
                "operator 2": {
                    "Name": objective,
                    "type": nodes[objective]["type"],
                    "value": nodes[objective].get("value"),
                },
                "result": {
                    "Name": node,
                    "type": nodes[node]["type"],
                    "value": nodes[node].get("value"),
                },
            }
            processed_results.add(node)  # Mark this result node as processed
            equation_number += 1
            break  # Move to the next result node after processing
    return mapping


def compute_complexity(graph):
    nodes = graph["nodes"]
    edges = graph["edges"]

    # Complexity metrics
    num_nodes = len(nodes)
    num_edges = len(edges)
    operation_diversity = len(
        set(
            (
                "div"
                if edge["operation"] in ["div", "div_by"]
                else (
                    "sub"
                    if edge["operation"] in ["sub", "sub_by", "sub_from"]
                    else edge["operation"]
                )
            )
            for edge in edges.values()
        )
    )

    # Helper function for graph depth and longest paths calculation
    def depth_and_paths_calculation(
        node, nodes, current_path, longest_paths, max_depth
    ):
        current_path.append(node)

        if "outgoing_edges" not in nodes[node] or not nodes[node]["outgoing_edges"]:
            if len(current_path) > max_depth:
                max_depth = len(current_path)
                longest_paths.clear()
                longest_paths.append(current_path[:])
            elif len(current_path) == max_depth:
                longest_paths.append(current_path[:])
        else:
            for edge in nodes[node]["outgoing_edges"]:
                next_node = edges[edge]["to"]
                max_depth = depth_and_paths_calculation(
                    next_node, nodes, current_path[:], longest_paths, max_depth
                )

        return max_depth

    # Initialize variables for longest paths
    longest_paths = []
    max_depth = 0

    # Calculate longest paths
    for node in nodes:
        if "incoming_edges" in nodes[node] and not nodes[node]["incoming_edges"]:
            max_depth = depth_and_paths_calculation(
                node, nodes, [], longest_paths, max_depth
            )

    # Graph depth is the number of edges in the longest path
    graph_depth = max_depth - 1

    # Final Complexity Score calculation
    def count_digits(value):
        # Convert the value to string to count digits before and after the decimal point
        # if it's a float, split by '.', otherwise consider it an integer
        str_value = str(value)
        if "." in str_value:
            return len(str_value.replace(".", ""))
        else:
            return len(str_value)

    def calculate_edge_complexity(edge):
        operation = edge["operation"]
        from_node_value = nodes[edge["from"]]["value"]
        objective_node_value = nodes[edge["objective"]]["value"]

        from_digits = count_digits(from_node_value)
        objective_digits = count_digits(objective_node_value)

        if operation in ["add", "sub", "sub_by"]:
            return min(from_digits, objective_digits)
        elif operation in ["mul", "div"]:
            return from_digits * objective_digits
        else:
            return 0  # Default case if operation is unknown

    total_complexity = sum(calculate_edge_complexity(edge) for edge in edges.values())
    # Normalize numerical complexity by the number of edges
    numerical_complexity = {
        "total": total_complexity,
        "ave": total_complexity / len(edges) if edges else 0,
    }

    return {
        "Number of Nodes": num_nodes,
        "Number of Edges": num_edges,
        "Operation Diversity": operation_diversity,
        "Graph Depth": graph_depth,
        "Numerical Complexity": numerical_complexity,
        "Longest Paths": longest_paths,
    }


def find_non_longest_paths(graph):
    def depth_first_search(node_id, current_path, all_paths):
        """
        Performs a depth-first search to explore all paths.
        """
        current_path.append(node_id)
        outgoing_edges = graph["nodes"][node_id]["outgoing_edges"]
        if not outgoing_edges:  # If there are no outgoing edges, it's a final node
            all_paths.append(current_path[:])
        else:
            for edge_id in outgoing_edges:
                next_node_id = graph["edges"][edge_id]["to"]
                depth_first_search(next_node_id, current_path[:], all_paths)

    # Find all initial nodes
    initial_nodes = [
        node_id
        for node_id, node_data in graph["nodes"].items()
        if not node_data["incoming_edges"]
    ]

    # Collect all paths
    all_paths = []
    for initial_node in initial_nodes:
        depth_first_search(initial_node, [], all_paths)

    # Identify the longest path(s)
    max_length = max(len(path) for path in all_paths)
    longest_paths = [path for path in all_paths if len(path) == max_length]

    # Filter out the longest path(s), leaving only the non-longest
    non_longest_paths = [path for path in all_paths if path not in longest_paths]

    return non_longest_paths



def initialize_code_agent(llm, prompt_template):
    python_tool = PythonREPLTool()
    tools = [
        Tool(
            name="python_repl",
            func=python_tool.run,
            description="useful for running generated python code",
        )
    ]
    # Set up the base template
    template_tool = prompt_template

    # Set up a prompt template
    class CustomPromptTemplate(StringPromptTemplate):
        # The template to use
        template: str
        # The list of tools available
        tools: List[Tool]

        def format(self, **kwargs) -> str:
            # Get the intermediate steps (AgentAction, Observation tuples)
            # Format them in a particular way
            intermediate_steps = kwargs.pop("intermediate_steps")
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            # Set the agent_scratchpad variable to that value
            kwargs["agent_scratchpad"] = thoughts
            # Create a tools variable from the list of tools provided
            kwargs["tools"] = "\n".join(
                [f"{tool.name}: {tool.description}" for tool in self.tools]
            )
            # Create a list of tool names for the tools provided
            kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
            return self.template.format(**kwargs)

    prompt = CustomPromptTemplate(
        template=template_tool,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["question", "intermediate_steps"],
    )

    class CustomOutputParser(AgentOutputParser):

        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
            # Check if agent should finish
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    # Return values is generally always a dictionary with a single `output` key
                    # It is not recommended to try anything else at the moment :)
                    return_values={
                        "output": llm_output.split("Final Answer:")[-1]
                        .strip()
                        .split("\n")[0]
                        .replace("<|im_end|>", "")
                    },
                    log=llm_output,
                )
            # Parse out the action and action input
            # Need to be fixed to adapted to other tools
            regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            time.sleep(15)
            if not match:
                raise OutputParserException(
                    f"Could not parse LLM output: `{llm_output}`"
                )
            action = match.group(1).strip()
            action_input = match.group(2)
            # Return the action and action input
            return AgentAction(
                tool=action,
                tool_input=action_input.strip(" ").strip('"'),
                log=llm_output,
            )

    output_parser_1 = CustomOutputParser()
    # LLM chain consisting of the LLM and a prompt

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser_1,
        stop=["\nObservation:", "\Question:"],
        allowed_tools=tool_names,
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, max_iterations=2
    )

    return agent_executor
