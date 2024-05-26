import os

def find_py_files_recursively(exclude_dirs=[]):
    """Recursively find all .py files and return paths relative to the current working directory."""
    py_files = []
    root_directory = os.getcwd()
    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(root_directory):
        # Remove any unwanted directories from dirnames
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for filename in filenames:
            if filename.endswith('.py'):
                # Create a relative path from the root directory
                rel_path = os.path.relpath(os.path.join(dirpath, filename), root_directory)
                py_files.append(rel_path)
    return py_files




