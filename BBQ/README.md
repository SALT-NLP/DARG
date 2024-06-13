# Bias Benchmark for QA (BBQ) Dataset: Graph Construction, Interpolation, and Graph-to-Text Decoding

This directory contains the code for constructing reasoning graphs, modifying graphs, and decoding graphs to text for the BBQ Dataset.

## Usage

### Reasoning Graph Construction

To construct the reasoning graph for the BBQ dataset, use the `bbq_graph_construct.py` script with the following options:

```bash
python bbq_graph_construct.py
  --original_file <path>         
  --output_file <path>           
```

The graph structure and components are described in Section 3.2 of the paper, with an example graph shown in Figure 18.

#### Parameters

- `--original_file`: The file path to the original BBQ dataset. We sample 660 data points and provide it in `bbq_sampled.json`.
- `--output_file`: The file path for the output file containing the reasoning graph.

### Post-Processing

After obtaining the reasoning graph, run the following script to generate an additional entry (`Person Node Mapping`) for each data point. This entry represents the mapping between the person node's name and the options, which will be used in the following stages:

```bash
python node_option_mapping.py     
```

### Attribute Generation

To generate new positive and negative attributes for augmenting reasoning graphs, use the following scripts:

```bash
python positive_attributes_generation.py  # Generate positive attributes
```

```bash
python negative_attributes_generation.py  # Generate negative attributes
```

These scripts will generate 10 positive and 10 negative attributes for each person node in each data point. To ensure these attributes are unrelated to the original question's answer, use the following scripts for filtering:

```bash
python attributes_filter_positive.py  # Filter the positive attributes
```

```bash
python attributes_filter_negative.py  # Filter the negative attributes
```

### Graph Modification

To modify the reasoning graph for the BBQ dataset as described in Section 3.2, use the `polarity_increase.py` script with the following options: 

```bash
python polarity_increase.py
  --original_file <path>         
  --output_file <path>   
  --negative_attributes_file <path>      
  --positive_attributes_file <path>     
  --increase <value>
```

#### Parameters

- `--original_file`: The file path to the BBQ dataset with the reasoning graph.
- `--output_file`: The file path for the output file containing the updated reasoning graph.
- `--negative_attributes_file`: The file path containing the filtered negative attributes generated in the previous stages.
- `--positive_attributes_file`: The file path containing the filtered positive attributes generated in the previous stages.
- `--increase`: The number of pairs of attributes to add to the person nodes in the reasoning graph.

### Graph-to-Text Decoding

To convert the reasoning graph for the BBQ dataset back to the original context format, use the `graph2text.py` script with the following options:

```bash
python graph2text.py
  --original_file <path>         
  --output_file <path>   
  --update_graph_key <value>
```

#### Parameters

- `--original_file`: The file path to the BBQ dataset with the modified reasoning graph.
- `--output_file`: The file path for the output file containing the updated context.
- `--update_graph_key`: The entry name of the updated graph in each data point.