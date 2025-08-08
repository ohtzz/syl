import ast
import textwrap
import tree_sitter_python as tspython

from .parser import CodeParser, FunctionEmbedding

from loguru import logger as log
from typing import List
from tree_sitter import Language, Parser
from radon.complexity import cc_visit
from radon.metrics import mi_visit


class PythonCodeProcessor(CodeParser):
    """Processes Python code to extract functions and generate embeddings"""

    def __init__(self, model_name: str, vector_size: int = 1024, embed_model_instance=None):
        super().__init__(model_name=model_name, vector_size=vector_size, embed_model_instance=embed_model_instance)

        try:
            PY_LANGUAGE = Language(tspython.language())
            self.parser = Parser(PY_LANGUAGE)
        except Exception as e:
            log.error(f'Failed to setup Python tree-sitter: {e}')
            raise SystemExit(1)

    def extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f'{module}.{alias.name}' if module else alias.name)
        return imports

    def extract_function_calls(self, node: ast.AST) -> List[str]:
        """Extract function calls from AST node"""
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
        return list(set(calls))  # Remove duplicates

    def calculate_complexity(self, code: str) -> tuple[int, float]:
        """Calculate cyclomatic complexity and maintainability index"""
        try:
            # Try and make sense of some indentation
            lines = code.strip().split('\n')
            if not lines or not lines[0].strip():
                return 0, 0.0

            # TODO this doesn't handle all indentation cases \
            # TODO which is a problem.. but won't be an issue for JS/Go/whatever
            clean_code = textwrap.dedent(code)

            clean_lines = []
            for line in clean_code.split('\n'):
                # Skip empty lines and comments-only lines for complexity
                if line.strip() and not line.strip().startswith('#'):
                    clean_lines.append(line)

            if not clean_lines:
                return 0, 0.0

            # Reconstruct
            final_code = '\n'.join(clean_lines)

            # Validate the code is syntactically correct
            try:
                compile(final_code, '<string>', 'exec')
            except SyntaxError as se:
                log.debug(f'Syntax error in extracted code: {se}')

                # Handle cases where we need to dedent more
                # because Python is [REDACTED]..
                if 'unexpected indent' in str(se):
                    final_code = '\n'.join(line.lstrip() for line in final_code.split('\n'))
                    try:
                        compile(final_code, '<string>', 'exec')
                    except SyntaxError:
                        log.debug('Could not fix syntax error, skipping complexity calculation')
                        return 0, 0.0
                else:
                    return 0, 0.0

            # Cyclomatic complexity
            complexity_results = cc_visit(final_code)
            total_complexity = sum(item.complexity for item in complexity_results)

            # Maintainability index
            mi_results = mi_visit(final_code, multi=True)
            mi_score = mi_results if isinstance(mi_results, (int, float)) else 0

            return total_complexity, mi_score

        except Exception as e:
            log.error(f'Failed to calculate complexity: {e}')
            return 0, 0.0

    def extract_function_metadata(self, file_path: str, code: str) -> List[FunctionEmbedding]:
        """Extract metadata for all functions in a file"""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            log.warning(f'Syntax error in {file_path}: {e}')
            return []

        functions = []
        imports = self.extract_imports(tree)

        class FunctionVisitor(ast.NodeVisitor):
            def __init__(self, processor, file_path, code, imports, functions):
                self.processor = processor
                self.file_path = file_path
                self.code = code
                self.imports = imports
                self.functions = functions
                self.current_class = None

            def visit_ClassDef(self, node):
                old_class = self.current_class
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = old_class

            def visit_FunctionDef(self, node):
                self.process_function(node)
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node):
                self.process_function(node, is_async=True)
                self.generic_visit(node)

            def process_function(self, node, is_async=False):
                # Extract function code
                func_lines = self.code.split('\n')[node.lineno - 1 : node.end_lineno]
                func_code = '\n'.join(func_lines)

                # Get function calls
                calls = self.processor.extract_function_calls(node)

                # Get parameters
                parameters = [arg.arg for arg in node.args.args]

                # Get return annotation
                returns = None
                if node.returns:
                    if isinstance(node.returns, ast.Name):
                        returns = node.returns.id
                    elif isinstance(node.returns, ast.Constant):
                        returns = str(node.returns.value)

                # Get docstring
                docstring = None
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    docstring = node.body[0].value.value

                # Calculate function-specific complexity (skip if causing issues)
                try:
                    func_complexity, func_mi = self.processor.calculate_complexity(func_code)
                except Exception as e:
                    # Skip complexity calculation if it fails, just use 0
                    log.warning(f'Failed to calculate complexity for function {self.file_path}: {str(e)}')
                    func_complexity, func_mi = 0, 0.0

                metadata = FunctionEmbedding(
                    id='',
                    embedding=[],
                    code='',
                    function_name=node.name,
                    file_path=self.file_path,
                    line_start=node.lineno,
                    line_end=node.end_lineno,
                    calls=calls,
                    called_by=[],  # Will be populated later
                    complexity=func_complexity,
                    maintainability_index=func_mi,
                    parameters=parameters,
                    returns=returns,
                    docstring=docstring,
                    is_async=is_async,
                    is_method=self.current_class is not None,
                    class_name=self.current_class,
                    imports=self.imports,  # TODO we're not really using these and we _shouldn't_ need them if we end up putting them in the files table..
                )

                self.functions.append(metadata)

        visitor = FunctionVisitor(self, file_path, code, imports, functions)
        visitor.visit(tree)

        return functions

    def process_file(self, file_path: str) -> List[FunctionEmbedding]:
        """Process a single Python file"""
        log.info(f'Processing: {file_path}')

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            log.error(f'Failed to read {file_path}: {e}')
            return []

        if not code.strip():
            return []

        # Extract function metadata
        functions_metadata = self.extract_function_metadata(file_path, code)

        if not functions_metadata:
            return []

        # Generate embeddings for each function
        function_embeddings = []

        for func_meta in functions_metadata:
            try:
                # Extract function code
                lines = code.split('\n')
                func_code = '\n'.join(lines[func_meta.line_start - 1 : func_meta.line_end])

                # Create embedding input (function signature + docstring + code)
                embedding_text = f'def {func_meta.function_name}({", ".join(func_meta.parameters or [])}):\n'
                if func_meta.docstring:
                    embedding_text += f'    """{func_meta.docstring}"""\n'
                embedding_text += func_code

                # Generate embedding
                embedding = self.embed_model.encode(embedding_text).tolist()

                # Validate embedding dimension
                if len(embedding) != self.vector_size:
                    log.error(
                        f"Generated embedding dimension ({len(embedding)}) doesn't match expected size ({self.vector_size})"
                    )
                    continue

                # Create function embedding object
                func_id = self.generate_function_id(func_meta)
                func_embedding = FunctionEmbedding(
                    id=func_id,
                    embedding=embedding,
                    code=func_code,
                    function_name=func_meta.function_name,
                    file_path=func_meta.file_path,
                    line_start=func_meta.line_start,
                    line_end=func_meta.line_end,
                    calls=func_meta.calls,
                    called_by=func_meta.called_by,
                    complexity=func_meta.complexity,
                    maintainability_index=func_meta.maintainability_index,
                    parameters=func_meta.parameters,
                    returns=func_meta.returns,
                    docstring=func_meta.docstring,
                    is_async=func_meta.is_async,
                    is_method=func_meta.is_method,
                    class_name=func_meta.class_name,
                    imports=func_meta.imports,
                )

                function_embeddings.append(func_embedding)

            except Exception as e:
                log.error(f'Failed to process function {func_meta.function_name} in {file_path}: {e}')
                continue

        return function_embeddings
