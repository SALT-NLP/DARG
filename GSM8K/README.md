# GSM8K Reasoning Graph Construction and Graph-to-Text Decoding

This repository provides the tools for constructing reasoning/computational graphs from the GSM8K dataset and converting these graphs back to text using a code-augmented Large Language Model (LLM) agent.

## 1. Graph Construction

To construct reasoning graphs from the GSM8K dataset, run the following command:

```bash
python graph_construct.py --original_file <path_to_original_file> --output_file <name_of_output_file>
```

### Parameters:
- `--original_file`: Path to the original dataset file (default: `test.jsonl`).
- `--output_file`: Name of the output file containing the reasoning graph for each data point (default: `gsm8k_graph.json`).

This script leverages GPT-4 to convert the GSM8K test data (sample data in `test.jsonl`, full dataset available [here](https://github.com/openai/grade-school-math)) into a structure referred to as "mapping". The mapping represents the relationship between numbers in the equations and nodes in the reasoning graph. Essentially, the mapping and reasoning graph are interchangeable.

Upon execution, the script will generate a file, `gsm8k_graph.json`, containing reasoning graphs for each data point, structured as follows:

- **Original**: The original data point.
- **Mapping**: The generated mapping (reasoning graph) corresponding to the original data. The components of the graph are detailed in `prompt.py`, and the graph values can be computed using the `compute_graph_values` function.

## 2. Graph-to-Text Decoding

To convert the reasoning graphs back to natural language questions, run:

```bash
python graph2text.py --original_file <path_to_input_file> --output_file <name_of_output_file> --max_iter_phase1 <iterations_phase1> --max_iter_phase2 <iterations_phase2> --max_iter_validate <iterations_validate>
```

### Parameters:
- `--original_file`: Path to the file with modified reasoning graphs (post-interpolation, found in `./depth`, `./numerical`, `./width`).
- `--output_file`: Name of the output file containing newly generated questions in the "Updated Question" entry.
- `--max_iter_phase1`: Maximum iterations for the initial question generation phase.
- `--max_iter_phase2`: Maximum iterations for the interaction phase between the code-augmented agent and the question generator.
- `--max_iter_validate`: Maximum iterations for majority voting to validate the newly generated questions.

Increasing the values for the last three parameters generally results in higher quality questions, but also increases the frequency of LLM calls, leading to higher costs and more processing time.

## 3. Files

- **graph_construct.py**: Script for converting GSM8K test data to reasoning graphs.
- **prompt.py**: Contains the prompt used for constructing mappings (reasoning graphs).
- **utils.py**: Utility functions used throughout the process.
- **graph2text.py**: Script for converting reasoning graphs back to natural language questions.
