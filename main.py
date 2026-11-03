import sys
from pathlib import Path

print("Welcome to SortWise AI!")
print("Project setup test: SUCCESS")

print("Python version:", sys.version)
print("Python executable:", sys.executable)

project_folder = Path.cwd()
print("Current project folder:", project_folder)
