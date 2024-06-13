# Reasoning Graph Width Increase

This directory contains the code to increase the width of the reasoning graphs generated from the previous step. Specifically, this process identifies the non-longest paths in the original reasoning graph and splits their starting points to increase the width of the graph without increasing its depth (the number of nodes in the longest path) or numerical complexity. In cases where the desired width increase is not feasible (e.g., there are not enough non-longest paths), the script will increase the width to the maximum possible extent.

## Usage

```bash
python width_increase.py --original_file <path_to_original_graph> --width_increase <desired_width_increase>
```

- **--original_file**: The original reasoning graph file constructed in the previous step. Please copy this file into the current directory.
- **--width_increase**: The desired width increase.

After running this script, it will generate multiple files corresponding to the reasoning graphs with width increases of 1, 2, ..., up to the specified width increase. The generated files will be named `width_increase_<i+1>.json`, where `i` ranges from 0 to `width_increase-1`.

## Output Files

For each file, each entry (data point) will have the same keys as the original data, plus two additional keys:

- **Width_increase_i_graph**: The reasoning graph with the specified width increase.
- **Actual Width Increase**: Indicates the actual width increase achieved for the reasoning graph.
