navigate_graph_construct = """**Task Objective**: The goal is to construct a linear graph representation from a given instruction set. This graph should faithfully reflect the sequence and details of the actions described in the instruction, allowing for an accurate reconstruction of the original instructions when needed.
### Graph Structure Components
**Nodes**: Each node represents a specific action in the sequence of instructions.
   - **Properties**:
     - `order`: the sequential position of this action within the instruction set.
     - `step_num`: the number of steps involved in this action.
     - `direction`: the specific direction of movement for this action, which can be one of four types: forward, backward, left, or right. Initially, if no direction is specified, the default direction is forward. If the direction is not clearly specified later, you should determine the most appropriate direction based on the context, or randomly select a direction when no contextual clues are available.

Example: 
Instruction: Take 7 steps forward. Take 4 steps backward. Take 4 steps backward. Take 5 steps forward. Take 7 steps forward. Take 10 steps backward. Take 1 step backward. 
Graph: 
{{
  "nodes": [
    {{
      "order": 1,
      "step_num": 7,
      "direction": "forward"
    }},
    {{
      "order": 2,
      "step_num": 4,
      "direction": "backward"
    }},
    {{
      "order": 3,
      "step_num": 4,
      "direction": "backward"
    }},
    {{
      "order": 4,
      "step_num": 5,
      "direction": "forward"
    }},
    {{
      "order": 5,
      "step_num": 7,
      "direction": "forward"
    }},
    {{
      "order": 6,
      "step_num": 10,
      "direction": "backward"
    }},
    {{
      "order": 7,
      "step_num": 1,
      "direction": "backward"
    }}
  ]
}}

Let's try another example:

Instruction: {instruction}
Graph: {format_instructions}
"""
