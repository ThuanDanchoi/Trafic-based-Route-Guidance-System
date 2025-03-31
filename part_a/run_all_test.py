"""
Search program for Traffic-based Route Guidance System - Part A
Usage: python search.py <filename> <method>
"""

import os
import subprocess

test_directory = "test_cases"

test_files = ["test1.txt", "test2.txt", "test3.txt", "test4.txt", "test5.txt", 
              "test6.txt", "test7.txt", "test8.txt", "test9.txt", "test10.txt"]

algorithms = ["DFS", "BFS", "GBFS", "AS", "CUS1", "CUS2"]

for test_file in test_files:
    file_path = os.path.join(test_directory, test_file)
    for algorithm in algorithms:
        print(f"\nRunning {algorithm} on {test_file}")
        subprocess.run(["python", "search.py", file_path, algorithm])