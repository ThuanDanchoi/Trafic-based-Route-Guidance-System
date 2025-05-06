# Trafic-based-Route-Guidance-System

## Introduction
This project implements tree-based search algorithms to solve the Route Finding problem. The objective is to find optimal paths (lowest cost) from an origin node to one or more destination nodes on a 2D graph.

The implementation includes both informed and uninformed search methods to compare their effectiveness in solving route finding problems.

## Search Algorithms
This project implements 6 different search algorithms:

### Uninformed Search Methods
- **DFS** (Depth-First Search): Selects one option, tries it, goes back when there are no more options
- **BFS** (Breadth-First Search): Expands all options one level at a time
- **CUS1** (IDDFS - Iterative Deepening Depth-First Search): Starts searching with depth 1, then gradually increases the depth with each iteration. It finds the same path as BFS but uses less memory as it's DFS with a depth limit.

### Informed Search Methods
- **GBFS** (Greedy Best-First Search): Uses only the cost to reach the goal from the current node to evaluate the node
- **AS** (A* - A Star): Uses both the cost to reach the goal from the current node and the cost to reach this node to evaluate the node
- **CUS2** (Weighted A*): Similar to A* but places a larger weight on the heuristic component. This makes it biased toward exploring nodes with smaller heuristic values (closer to the goal), potentially increasing search speed in some cases

## Installation

### System Requirements
- Python 3.x
- Windows 10

### How to Install
1. Clone this repository:
```
git clone https://github.com/yourusername/route-finding.git](https://github.com/ThuanDanchoi/Trafic-based-Route-Guidance-System.git)
```

2. Navigate to the project directory:
```
cd part_a
```

3. No additional packages are required for the base functionality.

## Usage
The program runs from the Command Line Interface with the format:

```
python search.py <filename> <method>
```

Where:
- `<filename>`: Name of the file containing the graph information
- `<method>`: Search method (DFS, BFS, GBFS, AS, CUS1, CUS2)

### Output Format
When a goal can be reached, the standard output will have the format:
```
filename method
goal number_of_nodes
path
```

## Features
- Support for 6 different search algorithms
- Reading graph information from text files
- Finding optimal paths from the origin node to one of the destination nodes
- Displaying results including the goal node reached, number of nodes created, and the path

## Notes
- When all other factors are equal, nodes will be expanded in ascending order (from smaller to larger)
- The objective is to reach one of the destination nodes

## Development Team
- Duc Thuan Tran (104330455)
- Vu Anh Le (104653505)
- Harrish (103333333)

## License
This project is developed for educational purposes as part of the COS30019 - Introduction to AI course.
