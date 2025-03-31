"""
Output formatting utility for search results
"""

def format_output(filename, method, goal, nodes_created, path):
    # Convert path to string format
    path_str = " ".join(map(str, path))
    
    # Format the output according to requirements
    output = f"{filename} {method}\n{goal} {nodes_created}\n{path_str}"
    
    return output