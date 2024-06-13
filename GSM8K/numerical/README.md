# Reasoning Graph Numerical Complexity Increase

This directory contains the code to increase the numerical complexity of the reasoning graphs generated from the previous step. Specifically, this process modifes the original reasoning graph's nodes' values by sampling new values to achieve the desired numerical complexity increase. The numerical complexity  is defined as the number of unit additions in the calculations.

## Usage

```bash
python numerical_increase.py --original_file <path_to_original_graph> --numerical_increase <desired_numerical_increase>
```

- **--original_file**: The original reasoning graph file constructed in the previous step. Please copy this file into the current directory.
- **--numerical_increase**: The desired numeircal complexity increase.

After running this script, it will generate a file corresponding to the reasoning graph with specified numerical complexity increase. 

## Output Files

For each file, each entry (data point) will have the same keys as the original data, plus an additional key:

- **graph**: The reasoning graph with the specified numerical increase.
