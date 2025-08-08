import re
import math

from loguru import logger as log
from typing import List, Optional

from .parser import CodeParser, FunctionEmbedding

import tree_sitter_javascript as tsjs
import tree_sitter_typescript as tsts
from tree_sitter import Language, Parser, Node


class JavaScriptCodeProcessor(CodeParser):
    """Processes JavaScript/TypeScript code to extract functions and generate embeddings"""

    def __init__(self, model_name: str, vector_size: int = 1024, embed_model_instance=None):
        super().__init__(model_name=model_name, vector_size=vector_size, embed_model_instance=embed_model_instance)

        self.js_parser = None
        self.ts_parser = None

        try:
            JS_LANGUAGE = Language(tsjs.language())
            self.js_parser = Parser(JS_LANGUAGE)

            TS_LANGUAGE = Language(tsts.language_typescript())
            self.ts_parser = Parser(TS_LANGUAGE)

            log.info('Tree-sitter JavaScript/TypeScript parsers initialized')
        except Exception as e:
            log.warning(f'Failed to setup JS/TS tree-sitter parsers: {e}')
            raise SystemExit(1)

    def extract_imports(self, tree: 'Node', is_typescript: bool = False) -> List[str]:
        """Extract import statements from tree-sitter AST"""
        imports = []

        def visit_node(node: 'Node'):
            if node.type == 'import_statement':
                # Handle various import patterns
                import_text = node.text.decode('utf-8')

                # Extract module name from import
                import_from_match = re.search(r'from\s+[\'"]([^\'"]+)[\'"]', import_text)
                if import_from_match:
                    imports.append(import_from_match.group(1))
                else:
                    # Handle direct imports like import 'module'
                    direct_import_match = re.search(r'import\s+[\'"]([^\'"]+)[\'"]', import_text)
                    if direct_import_match:
                        imports.append(direct_import_match.group(1))

            elif node.type == 'call_expression':
                # Handle require() calls
                if node.children and node.children[0].type == 'identifier':
                    func_name = node.children[0].text.decode('utf-8')
                    if func_name == 'require' and len(node.children) > 1:
                        args = node.children[1]  # arguments
                        if args.children and args.children[0].type == 'string':
                            module_name = args.children[0].text.decode('utf-8').strip('\'"')
                            imports.append(module_name)

            for child in node.children:
                visit_node(child)

        visit_node(tree.root_node)
        return list(set(imports))  # Remove duplicates

    def extract_function_calls(self, node: 'Node') -> List[str]:
        """Extract function calls from tree-sitter AST node"""
        calls = []

        def visit_node(n: 'Node'):
            if n.type == 'call_expression':
                if n.children and n.children[0].type == 'identifier':
                    # Simple function call
                    calls.append(n.children[0].text.decode('utf-8'))
                elif n.children and n.children[0].type == 'member_expression':
                    # Method call like obj.method()
                    member_expr = n.children[0]
                    if member_expr.children and len(member_expr.children) >= 3:
                        # Get the method name (property)
                        prop_node = member_expr.children[2]  # object.property
                        if prop_node.type == 'property_identifier':
                            calls.append(prop_node.text.decode('utf-8'))

            for child in n.children:
                visit_node(child)

        visit_node(node)
        return list(set(calls))  # Remove duplicates

    def calculate_complexity(self, code: str) -> tuple[int, float]:
        """Calculate rough cyclomatic complexity for JavaScript/TypeScript code"""
        try:
            complexity = 1  # Base complexity

            # Control flow keywords
            control_keywords = ['if', 'while', 'for', 'switch', 'catch', 'try', 'do']
            for keyword in control_keywords:
                # Use word boundaries to avoid matching within other words
                pattern = r'\b' + keyword + r'\b'
                complexity += len(re.findall(pattern, code))

            # Count 'else if' separately
            complexity += len(re.findall(r'\belse\s+if\b', code))

            # Logical operators
            complexity += len(re.findall(r'&&', code))
            complexity += len(re.findall(r'\|\|', code))

            # Ternary operators
            complexity += len(re.findall(r'\?[^?:]*:', code))

            # Close-ish: MI = 171 - 5.2*CC - 0.23*LOC - 16.2*ln(LOC)
            lines_of_code = len([line for line in code.split('\n') if line.strip()])

            # Calc and normalize to match radon 0-100
            if lines_of_code > 0:
                log_loc = math.log(lines_of_code)
                raw_mi = 171 - 5.2 * complexity - 0.23 * lines_of_code - 16.2 * log_loc
                maintainability = max(0, min(100, raw_mi * 100 / 171))
            else:
                maintainability = 0.0

            return complexity, maintainability

        except Exception as e:
            log.error(f'Failed to calculate complexity: {e}')
            return 0, 0.0

    def extract_function_metadata(
        self, file_path: str, code: str, is_typescript: bool = False
    ) -> List[FunctionEmbedding]:
        """Extract metadata for all functions in a JavaScript/TypeScript file"""
        try:
            parser = self.ts_parser if is_typescript else self.js_parser
            if not parser:
                log.warning(f'Parser not available for {"TypeScript" if is_typescript else "JavaScript"}')
                return []

            tree = parser.parse(bytes(code, 'utf-8'))
        except Exception as e:
            log.warning(f'Parse error in {file_path}: {e}')
            return []

        functions = []
        imports = self.extract_imports(tree, is_typescript)

        def visit_node(node: 'Node', current_class: Optional[str] = None):
            if node.type in ['function_declaration', 'function_expression', 'arrow_function']:
                func_meta = self.process_function_node(node, file_path, code, imports, current_class)
                if func_meta:
                    functions.append(func_meta)

            elif node.type == 'method_definition':
                # Class methods
                func_meta = self.process_method_node(node, file_path, code, imports, current_class)
                if func_meta:
                    functions.append(func_meta)

            elif node.type == 'class_declaration':
                # Extract class name and visit children
                class_name = None
                for child in node.children:
                    if child.type == 'identifier':
                        class_name = child.text.decode('utf-8')
                        break

                for child in node.children:
                    visit_node(child, class_name)
                return  # Don't visit children again

            # Visit all children
            for child in node.children:
                visit_node(child, current_class)

        visit_node(tree.root_node)
        return functions

    def process_function_node(
        self, node: 'Node', file_path: str, code: str, imports: List[str], current_class: Optional[str] = None
    ) -> Optional[FunctionEmbedding]:
        """Process a function node and extract metadata"""
        try:
            # Get function name
            function_name = 'anonymous'
            is_async = False

            for child in node.children:
                if child.type == 'identifier':
                    function_name = child.text.decode('utf-8')
                    break
                elif child.type == 'async':
                    is_async = True

            # Check for async keyword in parent nodes
            parent = node.parent
            while parent:
                if 'async' in parent.text.decode('utf-8'):
                    is_async = True
                    break
                parent = parent.parent

            # Get line numbers
            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1

            # Extract function code
            lines = code.split('\n')
            func_code = '\n'.join(lines[line_start - 1 : line_end])

            # Get function calls
            calls = self.extract_function_calls(node)

            # Get parameters
            parameters = []
            for child in node.children:
                if child.type == 'formal_parameters':
                    for param_child in child.children:
                        if param_child.type == 'identifier':
                            parameters.append(param_child.text.decode('utf-8'))
                        elif param_child.type == 'assignment_pattern':
                            # Handle default parameters
                            for assignment_child in param_child.children:
                                if assignment_child.type == 'identifier':
                                    parameters.append(assignment_child.text.decode('utf-8'))
                                    break

            # Get return type (TypeScript)
            returns = None
            if node.type == 'function_declaration':
                func_text = node.text.decode('utf-8')
                return_type_match = re.search(r':\s*([^{]+?)\s*{', func_text)
                if return_type_match:
                    returns = return_type_match.group(1).strip()

            # Extract JSDoc comment
            docstring = self.extract_jsdoc(node, code)

            # Calculate complexity
            complexity, maintainability = self.calculate_complexity(func_code)

            metadata = FunctionEmbedding(
                id='',
                embedding=[],
                code='',
                function_name=function_name,
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                calls=calls,
                called_by=[],  # Will be populated later
                complexity=complexity,
                maintainability_index=maintainability,
                parameters=parameters,
                returns=returns,
                docstring=docstring,
                is_async=is_async,
                is_method=current_class is not None,
                class_name=current_class,
                imports=imports,
            )

            return metadata

        except Exception as e:
            log.error(f'Failed to process function node in {file_path}: {e}')
            return None

    def process_method_node(
        self, node: 'Node', file_path: str, code: str, imports: List[str], current_class: Optional[str] = None
    ) -> Optional[FunctionEmbedding]:
        """Process a method definition node"""
        try:
            # Get method name
            method_name = 'unknown_method'
            is_async = False

            for child in node.children:
                if child.type == 'property_identifier':
                    method_name = child.text.decode('utf-8')
                elif child.type == 'function' or child.type == 'function_expression':
                    # Check if the function is async
                    func_text = child.text.decode('utf-8')
                    if func_text.startswith('async'):
                        is_async = True

            # Get line numbers
            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1

            # Extract method code
            lines = code.split('\n')
            func_code = '\n'.join(lines[line_start - 1 : line_end])

            # Get function calls
            calls = self.extract_function_calls(node)

            # Get parameters
            parameters = []
            for child in node.children:
                if child.type == 'function' or child.type == 'function_expression':
                    for func_child in child.children:
                        if func_child.type == 'formal_parameters':
                            for param_child in func_child.children:
                                if param_child.type == 'identifier':
                                    parameters.append(param_child.text.decode('utf-8'))

            # Extract JSDoc comment
            docstring = self.extract_jsdoc(node, code)

            # Calculate complexity
            complexity, maintainability = self.calculate_complexity(func_code)

            metadata = FunctionEmbedding(
                id='',
                embedding=[],
                code='',
                function_name=method_name,
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                calls=calls,
                called_by=[],  # Will be populated later
                complexity=complexity,
                maintainability_index=maintainability,
                parameters=parameters,
                returns=None,
                docstring=docstring,
                is_async=is_async,
                is_method=True,
                class_name=current_class,
                imports=imports,
            )

            return metadata

        except Exception as e:
            log.error(f'Failed to process method node: {e}')
            return None

    def extract_jsdoc(self, node: 'Node', code: str) -> Optional[str]:
        """Extract JSDoc comment for a function"""
        try:
            lines = code.split('\n')
            func_line = node.start_point[0]

            # Look for JSDoc comment above the function
            comment_lines = []
            for i in range(func_line - 1, -1, -1):
                line = lines[i].strip()
                if line.endswith('*/'):
                    comment_lines.insert(0, line)
                    # Look for the start of the comment
                    for j in range(i - 1, -1, -1):
                        prev_line = lines[j].strip()
                        comment_lines.insert(0, prev_line)
                        if prev_line.startswith('/**'):
                            break
                    break
                elif line.startswith('*') or line.startswith('/**'):
                    comment_lines.insert(0, line)
                elif line == '':
                    continue
                else:
                    break

            if comment_lines and comment_lines[0].startswith('/**'):
                # Clean up JSDoc comment
                doc_text = '\n'.join(comment_lines)
                # Remove JSDoc formatting
                doc_text = re.sub(r'/\*\*|\*/|\s*\*\s?', '', doc_text)
                return doc_text.strip() if doc_text.strip() else None

            return None

        except Exception as e:
            log.debug(f'Failed to extract JSDoc: {e}')
            return None

    def process_file(self, file_path: str) -> List[FunctionEmbedding]:
        """Process a single JavaScript/TypeScript file"""
        log.info(f'Processing: {file_path}')

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            log.error(f'Failed to read {file_path}: {e}')
            return []

        if not code.strip():
            return []

        # Determine if it's TypeScript
        is_typescript = file_path.endswith(('.ts', '.tsx'))

        # Extract function metadata
        functions_metadata = self.extract_function_metadata(file_path, code, is_typescript)

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
                if func_meta.is_async:
                    embedding_text = (
                        f'async function {func_meta.function_name}({", ".join(func_meta.parameters or [])}) {{\n'
                    )
                else:
                    embedding_text = f'function {func_meta.function_name}({", ".join(func_meta.parameters or [])}) {{\n'

                if func_meta.docstring:
                    embedding_text += f'    /** {func_meta.docstring} */\n'
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
