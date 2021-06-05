import os
import sys

sys.path.insert(1, os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

LIB_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")

print(f"Changed current working directory to {LIB_DIR}")
os.chdir(LIB_DIR)
