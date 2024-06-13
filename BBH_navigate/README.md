# BBH Navigate Dataset Reasoning Graph Construction and Depth Increase

This directory contains the code for constructing reasoning graphs and increasing the depth of the reasoning graphs for BBH navigate dataset. Please note that this task only need a rule-based function to convert the graph back to the original format wihtout using LLMs.

## Usage

### Graph Construction

To construct the reasoning graph (linear graph with each node represent a single action), please execute the following script

```bash
python navigate_graph_construct.py       
```

## Parameters

- `--original_file`: The file path to the original BBH Navigate dataset.
- `--output_file`: The file path for the output file which contains the reasoning graph for each data point

### Graph Depth Increase

To increase the depth of the reasoning graphs (the number of steps), please execute the following script

```bash
python depth_increase.py
```
## Parameters

- `--original_file`: The file obtained from the previous step with reasoning graph
- `--output_file`: The file path for the output file which contains the updated reasoning graph with increasing depth
- `--step_increase`: The specified increase value for the reasoning graph's depth

### Graph-to-Text Decoding

To convert the reasoning graph (linear graph with each node represent a single action) back to the original data format, please execute the following script

```bash
python graph2text.py    
```

## Parameters

- `--original_file`: The file obtained from the previous step with reasoning graph with increased depth
- `--output_file`: The file path for the output file which contains the new data point in the **Updated Instruction** entry 