import os
import json
import math
import re
import subprocess

from .parser import CodeParser, FunctionEmbedding

from typing import List, Optional
from loguru import logger as log


GO_PARSER_BIN_LOC = '/usr/local/bin/go-parser'


class GoCodeProcessor(CodeParser):
    """Processes Go code to extract functions and generate embeddings"""

    def __init__(self, model_name: str, vector_size: int = 1024, embed_model_instance=None):
        super().__init__(model_name=model_name, vector_size=vector_size, embed_model_instance=embed_model_instance)

        # Ensure we have the Go bin and I didn't [REDACTED] _kerfuffle_ the Docker builds
        if self._find_go_parser_binary() is None:
            log.error(f'Could not find go parser binary at: {GO_PARSER_BIN_LOC}')
            raise SystemExit(1)

    def _find_go_parser_binary(self) -> Optional[str]:
        if os.path.isfile(GO_PARSER_BIN_LOC) and os.access(GO_PARSER_BIN_LOC, os.X_OK):
            return GO_PARSER_BIN_LOC
        return None

    def calculate_go_complexity(self, code: str) -> tuple[int, float]:
        """Calculate cyclomatic complexity for Go code"""
        try:
            complexity = 1  # Base complexity

            # Go control flow keywords
            control_keywords = ['if', 'for', 'switch', 'select']
            for keyword in control_keywords:
                # Use word boundaries
                pattern = r'\b' + keyword + r'\b'
                complexity += len(re.findall(pattern, code))

            complexity += len(re.findall(r'\belse\s+if\b', code))
            complexity += len(re.findall(r'\bcase\b', code))
            complexity += len(re.findall(r'\bdefault\b', code))
            complexity += len(re.findall(r'\btype\s+switch\b', code))
            complexity += len(re.findall(r'\brange\b', code))

            # Logical operators
            complexity += len(re.findall(r'&&', code))
            complexity += len(re.findall(r'\|\|', code))

            # Count short if statements
            complexity += len(re.findall(r'\bif\s+[^{]*{[^}]*}(?:\s*else\s*{[^}]*})?', code))

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
            log.error(f'Failed to calculate Go complexity: {e}')
            return 0, 0.0

    def extract_function_metadata(self, file_path: str) -> List[FunctionEmbedding]:
        """Extract metadata for all functions in a Go file"""
        try:
            go_parser_path = self._find_go_parser_binary()
            if not go_parser_path:
                log.warning(f'Go parser binary not found, skipping {file_path}')
                return []

            result = subprocess.run([go_parser_path, file_path], capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                log.warning(f'Go parser failed for {file_path}: {result.stderr}')
                return []

            file_info = json.loads(result.stdout)
            functions = []
            imports = file_info.get('imports', [])

            for func_info in file_info.get('functions', []):
                complexity, maintainability_index = self.calculate_go_complexity(func_info['raw_code'])

                metadata = FunctionEmbedding(
                    id='',
                    embedding=[],
                    code='',
                    function_name=func_info['name'],
                    file_path=file_path,
                    line_start=func_info['start_line'],
                    line_end=func_info['end_line'],
                    calls=func_info['calls'],
                    called_by=[],  # Will be populated later
                    complexity=complexity,
                    maintainability_index=float(maintainability_index),
                    parameters=func_info['parameters'],
                    returns=func_info['returns'] if func_info['returns'] else None,
                    docstring=func_info['docstring'] if func_info['docstring'] else None,
                    is_async=False,  # Goroutines don't really map to this
                    is_method=func_info['is_method'],
                    class_name=func_info['receiver'] if func_info['is_method'] else None,
                    imports=imports,
                )

                functions.append(metadata)

            return functions

        except subprocess.TimeoutExpired:
            log.error(f'Timeout while parsing {file_path}')
            return []
        except json.JSONDecodeError as e:
            log.error(f'Failed to parse JSON output for {file_path}: {e}')
            return []
        except Exception as e:
            log.error(f'Failed to extract metadata from {file_path}: {e}')
            return []

    def process_file(self, file_path: str) -> List[FunctionEmbedding]:
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
        functions_metadata = self.extract_function_metadata(file_path)

        if not functions_metadata:
            return []

        function_embeddings = []

        for func_meta in functions_metadata:
            log.debug(f'Function metadata: {func_meta}')

            try:
                lines = code.split('\n')
                func_code = '\n'.join(lines[func_meta.line_start - 1 : func_meta.line_end])

                embedding_text = f'func {func_meta.function_name}({", ".join(func_meta.parameters or [])})'
                if func_meta.returns:
                    embedding_text += f' {func_meta.returns}'
                embedding_text += ' {\n'

                if func_meta.docstring:
                    embedding_text = f'// {func_meta.docstring}\n' + embedding_text

                embedding_text += func_code + '\n}'

                embedding = self.embed_model.encode(embedding_text).tolist()

                if len(embedding) != self.vector_size:
                    log.error(
                        f"Generated embedding dimension ({len(embedding)}) doesn't match expected size ({self.vector_size})"
                    )
                    continue

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
