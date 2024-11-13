import os
import ast
from pathlib import Path
from collections import defaultdict

# Function to get all imports from a given Python file
def get_imports_from_file(filepath):
    with open(filepath, "r") as file:
        tree = ast.parse(file.read(), filename=str(filepath))
    imports = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ""
                imports.append(module)

    return imports

# Traverse all Python files and create an import map
def create_import_map(base_directory):
    import_map = defaultdict(set)
    base_path = Path(base_directory)
    
    for filepath in base_path.rglob("*.py"):
        relative_filepath = filepath.relative_to(base_path)
        module_name = str(relative_filepath).replace("/", ".").replace(".py", "")
        
        try:
            imports = get_imports_from_file(filepath)
            for imported_module in imports:
                import_map[module_name].add(imported_module)
        except (SyntaxError, UnicodeDecodeError):
            # Skip files that cannot be parsed
            pass

    return import_map

# Perform a depth-first search to detect cycles
def find_cycles(import_map):
    def dfs(module, visited, stack):
        if module not in import_map:
            return False
        if module in stack:
            return stack[stack.index(module):]  # Circular path found
        if module in visited:
            return False
        visited.add(module)
        stack.append(module)

        for neighbor in import_map[module]:
            if dfs(neighbor, visited, stack):
                return True

        stack.pop()
        return False

    visited = set()
    for module in import_map:
        stack = []
        if dfs(module, visited, stack):
            print(f"Cycle detected: {' -> '.join(stack)}")
            return

    print("No circular imports found.")

# Main function to execute the circular dependency check
def main():
    base_directory = "."  # Current directory
    import_map = create_import_map(base_directory)
    find_cycles(import_map)

if __name__ == "__main__":
    main()
