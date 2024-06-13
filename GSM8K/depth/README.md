# Reasoning Graph Depth Increase

This directory contains the code to increase the depth of the reasoning graphs generated from the previous step. Specifically, this process identifies the longest path in the original reasoning graph and splits their starting points to increase the Depth of the graph without increasing its numerical complexity. 

## Usage

```bash
python depth_increase.py --original_file <path_to_original_graph> --depth_increase <desired_Depth_increase>
```

- **--original_file**: The original reasoning graph file constructed in the previous step. Please copy this file into the current directory.
- **--depth_increase**: The desired depth increase.

After running this script, it will generate multiple files corresponding to the reasoning graphs with Depth increases of 1, 2, ..., up to the specified depth increase. The generated files will be named `depth_increase_<i+1>.json`, where `i` ranges from 0 to `depth_increase-1`.

## Output Files

For each file, each entry (data point) will have the same keys as the original data, plus two additional keys:

- **Depth_increase_i_graph**: The reasoning graph with the specified Depth increase.
