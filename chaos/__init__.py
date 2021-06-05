import os
import sys

LIB_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")

# Inject local libraries (e.g. git submodules)
local_libs = ('grapresso',)
for lib in local_libs:
    path = os.path.join(LIB_DIR, lib)
    if path not in sys.path:
        sys.path.append(path)
