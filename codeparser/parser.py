import ast

def parse_code(file_path):
    try:
        with open(file_path, "r") as source:
            # Read and parse the source code to an AST
            tree = ast.parse(source.read(), filename=file_path)

        # Initialize lists to store functions and classes
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        # Initialize list to store code chunks
        chunks = []

        # Extract details from functions
        for function in functions:
            start_line = function.lineno
            end_line = function.end_lineno
            # Generate readable source code from the AST of the function
            function_code = ast.unparse(function)
            # Append the filename along with other details
            chunks.append((file_path, start_line, end_line, function_code))

        # Extract details from classes
        for cls in classes:
            start_line = cls.lineno
            end_line = cls.end_lineno
            # Generate readable source code from the AST of the class
            class_code = ast.unparse(cls)
            # Append the filename along with other details
            chunks.append((file_path, start_line, end_line, class_code))

        return chunks

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {str(e)}")
        return []
