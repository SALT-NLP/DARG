from langchain.llms import AzureOpenAI
import openai
import os

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


def returns_to_start(graph):
    # Initialize coordinates
    x, y = 0, 0

    # Process each node (action)
    for node in graph["nodes"]:
        direction = node["direction"]
        steps = node["step_num"]

        if direction == "forward":
            y += steps  # Move up on the y-axis
        elif direction == "backward":
            y -= steps  # Move down on the y-axis
        elif direction == "left":
            x -= steps  # Move left on the x-axis
        elif direction == "right":
            x += steps  # Move right on the x-axis

    # Check if the final coordinates are the same as the initial (0, 0)
    if x == 0 and y == 0:
        return "Yes"
    else:
        return "No"
