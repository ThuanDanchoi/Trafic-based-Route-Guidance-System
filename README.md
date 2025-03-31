# Trafic-based-Route-Guidance-System

## Introduction
This project is Part A of Assignment 2 for the COS30019 - Introduction to AI course. The objective is to implement tree-based search algorithms to solve the Route Finding problem.

In this problem, we need to find optimal paths (lowest cost) from an origin node to one or more destination nodes on a 2D graph. The project involves implementing both informed and uninformed search methods.

## Search Algorithms
This project implements 6 different search algorithms:

### Uninformed Search Methods
- **DFS** (Depth-First Search): Selects one option, tries it, goes back when there are no more options
- **BFS** (Breadth-First Search): Expands all options one level at a time
- **CUS1** (Custom Strategy 1): A custom uninformed method to find a path to reach the goal

### Informed Search Methods
- **GBFS** (Greedy Best-First Search): Uses only the cost to reach the goal from the current node to evaluate the node
- **AS** (A* - A Star): Uses both the cost to reach the goal from the current node and the cost to reach this node to evaluate the node
- **CUS2** (Custom Strategy 2): A custom informed method to find the shortest path (with least moves) to reach the goal

## Installation

### System Requirements
- Python 3.x
- Windows 10

### How to Install
1. Clone this repository:
```
git clone https://github.com/yourusername/route-finding.git
```

2. Navigate to the project directory:
```
cd route-finding
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
- Team Member 1 (ID)
- Team Member 2 (ID)
- Team Member 3 (ID)

## License
This project is developed for educational purposes as part of the COS30019 - Introduction to AI course.
