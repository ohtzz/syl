import os
import subprocess

from loguru import logger as log
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer

from .constants import CODE_PATH, SUPPORTED_FILETYPES
from .utils import classify_file_type, should_process_file

from syl.parsers.parser import FunctionEmbedding, FileEmbedding
from syl.parsers.python import PythonCodeProcessor
from syl.parsers.javascript import JavaScriptCodeProcessor
from syl.parsers.go import GoCodeProcessor


PYTHON_SKIP_DIRS = ['__pycache__', 'node_modules', 'venv', 'env', '.git', 'vendor', 'target', 'build', 'dist']
COMPRESS_LIMIT = 2048  # 2KB


class DataStoreLoader:
    embeddings: List[float] = None

    def __init__(
        self,
        embed_model: str,
        num_workers: int = 4,
        vector_size: int = 768,
        file_ext_whitelist: Optional[List[str]] = None,
        file_ext_blacklist: Optional[List[str]] = None,
        git_repo_url: str = None,
        git_branch: str = None,
    ):
        self.embed_model = embed_model
        self.num_workers = num_workers
        self._embed_model_instance = SentenceTransformer(self.embed_model)

        if file_ext_whitelist:
            self.file_types = set(
                e
                for e in file_ext_whitelist
                if e in SUPPORTED_FILETYPES and (not file_ext_blacklist or e not in file_ext_blacklist)  # I suppose you could blacklist items from your own whitelist..
            )
        else:
            self.file_types = set(
                e
                for e in SUPPORTED_FILETYPES
                if (not file_ext_blacklist or e not in file_ext_blacklist))
        self.git_repo_url = git_repo_url
        self.git_branch = git_branch
        self.vector_size = vector_size

        # I hope that embed model is cached..
        #
        # Share the model across all parsers so we
        # only have to load and initialize one.
        #
        # Should be _okay_ for threads..
        self.python_parser = PythonCodeProcessor(
            model_name=embed_model,
            vector_size=vector_size,
            embed_model_instance=self._embed_model_instance,
        )
        self.go_parser = GoCodeProcessor(
            model_name=embed_model,
            vector_size=vector_size,
            embed_model_instance=self._embed_model_instance,
        )
        self.javascript_parser = JavaScriptCodeProcessor(
            model_name=embed_model,
            vector_size=vector_size,
            embed_model_instance=self._embed_model_instance,
        )

    def git_clone(self):
        log.info(f'Cloning {self.git_repo_url}...', extra={'repo_url': self.git_repo_url})

        try:
            os.makedirs(CODE_PATH, exist_ok=True)

            # There's probably a Python SDK for this..
            cmd = ['git', 'clone', self.git_repo_url, CODE_PATH]
            _ = subprocess.run(cmd, capture_output=True, text=True, check=True)
            log.info(f'Successfully cloned repository to {CODE_PATH}')

            # Checkout specific branch if one was provided
            if self.git_branch:
                log.info(f'Checking out branch: {self.git_branch}')
                cmd = ['git', '-C', CODE_PATH, 'checkout', self.git_branch]
                _ = subprocess.run(cmd, capture_output=True, text=True, check=True)
                log.info(f'Successfully checked out branch: {self.git_branch}')

        except subprocess.CalledProcessError as e:
            error_msg = f'Git command failed: {e.cmd}'
            if e.stderr:
                error_msg += f'\nError: {e.stderr}'
            log.info(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f'Unexpected error during git clone: {str(e)}'
            log.info(error_msg)
            raise RuntimeError(error_msg)

    def setup(self):
        # Test connection
        self.test_connection()

        # Setup/clone code
        self.setup_code()

    def setup_code(self):
        """Code is either already mounted via Docker, or we git clone"""
        if not self.git_repo_url:
            log.info('Using files from file watcher - no git clone needed')
            return
        self.git_clone()

    def embed_and_load_data(self):
        """Generate embeddings for the code using the given model"""
        self._walk_code_path_and_process_in_batches()

    def _walk_code_path_and_process_in_batches(self, batch_size: int = 100) -> None:
        """Process files in batches to manage memory usage for large projects"""
        directory = Path(CODE_PATH)

        if not directory.exists():
            raise FileNotFoundError(f'Code path does not exist: {CODE_PATH}')

        if not directory.is_dir():
            raise NotADirectoryError(f'Code path is not a directory: {directory}')

        # Collect all files to process
        files_to_process = []
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in PYTHON_SKIP_DIRS]

            root_path = Path(root)
            for file in files:
                file_path = root_path / file

                # Skip hidden files
                if file.startswith('.'):
                    continue

                ext = file_path.suffix.lower()
                if ext in self.file_types and should_process_file(file_path):
                    files_to_process.append(file_path)

        log.info(
            'Collected files to process',
            extra={
                'num_files': len(files_to_process),
                'batch_size': batch_size,
            },
        )

        # Store all functions for call graph building
        # after we're done parsing
        all_functions = []
        all_files_data = []
        all_file_embeddings = []

        # Batch
        for i in range(0, len(files_to_process), batch_size):
            batch_files = files_to_process[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(files_to_process) - 1) // batch_size + 1

            log.info(
                'Processing batch',
                extra={
                    'batch_num': batch_num,
                    'total_batches': total_batches,
                    'batch_files': len(batch_files),
                },
            )

            batch_funcs = []
            batch_files_data = []

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_file = {
                    executor.submit(self._process_single_file, file_path): file_path for file_path in batch_files
                }

                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        if isinstance(result, tuple) and len(result) == 2:
                            funcs, file_data = result
                            batch_funcs.extend(funcs)
                            if file_data:
                                batch_files_data.append(file_data)
                    except Exception as exc:
                        log.error(f'File {file_path} generated an exception: {exc}')

            # Track for call graphs later
            all_functions.extend(batch_funcs)
            all_files_data.extend(batch_files_data)

            batch_file_embeddings = self._generate_file_embeddings(batch_files_data)
            all_file_embeddings.extend(batch_file_embeddings)

        if all_functions:
            log.info(f'Building call graph for {len(all_functions)} functions')
            self._build_call_graph(all_functions)

        if all_functions or all_file_embeddings:
            log.info(
                'Loading functions and file embeddings',
                extra={
                    'num_funcs': len(all_functions),
                    'num_file_embeddings': len(all_file_embeddings),
                },
            )
            self.load_functions(all_functions)
            self.load_file_embeddings(all_file_embeddings)
        else:
            log.warning('No functions or file embeddings found to load')

        log.info(
            'Completed processing all files',
            extra={
                'num_files': len(files_to_process),
            },
        )

    def test_connection(self):
        raise NotImplementedError('Subclass must implement test_connection()')

    def load_functions(self, funcs: List):
        raise NotImplementedError('Subclass must implement load_functions()')

    def load_file_embeddings(self, file_data_with_embeddings: List[FileEmbedding]):
        raise NotImplementedError('Subclass must implement load_file_embeddings()')

    def _process_single_file(self, file_path: Path) -> tuple:
        """Process a single file and return its functions and file data."""
        ext = file_path.suffix.lower()

        # Always create file data for supported files
        file_data = self._create_file_data(file_path)
        if file_data is None:
            return [], None

        # Only extract functions from code files
        if ext == '.py':
            funcs = self.python_parser.process_file(str(file_path))
            return funcs, file_data
        elif ext == '.go':
            funcs = self.go_parser.process_file(str(file_path))
            return funcs, file_data
        elif ext in ['.js', '.jsx', '.ts', '.tsx']:
            funcs = self.javascript_parser.process_file(str(file_path))
            return funcs, file_data
        else:
            # For non-code files (docs, config, data), only store file data
            return [], file_data

    def _create_file_data(self, file_path: Path) -> Optional[FileEmbedding]:
        """Create FileEmbedding object for a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract imports based on file type
            imports = []
            ext = file_path.suffix.lower()
            if ext == '.py':
                imports = self._extract_imports_from_content(content)
            elif ext == '.go':
                imports = self._extract_go_imports_from_content(content)

            file_data = FileEmbedding(
                file_path=str(file_path),
                filename=file_path.name,
                content=content,
                imports=imports,
                file_size=len(content.encode('utf-8')),
                line_count=len(content.split('\n')),
                file_type=classify_file_type(file_path),
            )

            return file_data

        except Exception as e:
            log.error(f'Failed to create file data for {file_path}: {e}')
            return None

    def _extract_imports_from_content(self, content: str) -> List[str]:
        """Extract imports from Python file content"""
        try:
            import ast

            tree = ast.parse(content)
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
        except Exception as e:
            log.warning(f'Failed to extract imports: {e}')
            return []

    def _extract_go_imports_from_content(self, content: str) -> List[str]:
        """Extract imports from Go file content using regex"""
        import re

        imports = []
        try:
            # Match import statements - both single and multi-line
            # Single import: import "package"
            single_import_pattern = r'import\s+"([^"]+)"'
            single_imports = re.findall(single_import_pattern, content)
            imports.extend(single_imports)

            # Multi-line imports: import ( ... )
            multi_import_pattern = r'import\s*\(\s*(.*?)\s*\)'
            multi_imports = re.findall(multi_import_pattern, content, re.DOTALL)
            for block in multi_imports:
                # Extract individual imports from the block
                import_lines = re.findall(r'"([^"]+)"', block)
                imports.extend(import_lines)

            return imports
        except Exception as e:
            log.warning(f'Failed to extract Go imports: {e}')
            return []

    def _build_call_graph(self, all_functions: List[FunctionEmbedding]) -> None:
        """Build call graph and populate called_by relationships for all functions"""
        # Map from function name to function objects
        function_name_map = {}

        for func in all_functions:
            if hasattr(func, 'function_name'):
                func_name = func.function_name
                class_name = getattr(func, 'class_name', None)
            else:
                func_name = getattr(func, 'function_name', str(func))
                class_name = getattr(func, 'class_name', None)

            qualified_name = func_name
            if class_name:
                qualified_name = f'{class_name}.{func_name}'

            # Store both simple and qualified names
            if func_name not in function_name_map:
                function_name_map[func_name] = []
            function_name_map[func_name].append(func)

            if qualified_name != func_name:
                if qualified_name not in function_name_map:
                    function_name_map[qualified_name] = []
                function_name_map[qualified_name].append(func)

        # Build called_by
        for func in all_functions:
            if hasattr(func, 'function_name'):
                calls = func.calls or []
                caller_name = func.function_name
                caller_class = getattr(func, 'class_name', None)
            else:
                calls = getattr(func, 'calls', [])
                caller_name = getattr(func, 'function_name', str(func))
                caller_class = getattr(func, 'class_name', None)

            # Create qualified caller name
            qualified_caller = caller_name
            if caller_class:
                qualified_caller = f'{caller_class}.{caller_name}'

            # For each function this one calls, add this function to its called_by list
            for called_func_name in calls:
                # Attempt to find the called function
                called_functions = []

                # First try exact match
                if called_func_name in function_name_map:
                    called_functions.extend(function_name_map[called_func_name])

                # Try with current class prefix if caller is a method
                if caller_class and not called_functions:
                    qualified_called = f'{caller_class}.{called_func_name}'
                    if qualified_called in function_name_map:
                        called_functions.extend(function_name_map[qualified_called])

                # Update called_by for all matching functions
                for called_func in called_functions:
                    if hasattr(called_func, 'called_by'):
                        called_by_list = called_func.called_by
                    else:
                        if not hasattr(called_func, 'called_by'):
                            called_func.called_by = []
                        called_by_list = called_func.called_by

                    # Add caller to called_by list if not already present
                    if qualified_caller not in called_by_list:
                        called_by_list.append(qualified_caller)

        log.info(f'Built call graph relationships for {len(all_functions)} functions')

    def _generate_file_embeddings(self, files_data: List[FileEmbedding]) -> List[FileEmbedding]:
        files_with_embeddings = []

        for file_data in files_data:
            try:
                # Generate embedding if content
                embedding_floats = None
                if file_data.content:
                    embedding = self._embed_model_instance.encode(file_data.content)
                    embedding_floats = [float(x) for x in embedding]

                    # Compress content if over the limit
                    if file_data.file_size > COMPRESS_LIMIT:
                        file_data.compress_content()
                        file_data.content = ''  # Clear uncompressed

                clean_file_path = file_data.file_path.replace('/', '_').replace('\\', '_')

                # Create a copy of file_data with embedding and ID
                file_with_embedding = FileEmbedding(
                    id=clean_file_path,
                    embedding=embedding_floats,
                    file_path=file_data.file_path,
                    filename=file_data.filename,
                    content=file_data.content,
                    content_compressed=file_data.content_compressed,
                    imports=file_data.imports,
                    file_size=file_data.file_size,
                    line_count=file_data.line_count,
                    file_type=file_data.file_type,
                )

                files_with_embeddings.append(file_with_embedding)

            except Exception as e:
                log.error(f'Failed to generate embedding for file {file_data.file_path}: {e}')
                continue

        return files_with_embeddings


class DataStoreQuery:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._embed_model = None

    @property
    def embed_model(self):
        if self._embed_model is None:
            try:
                log.info('Loading embedding model', extra={'model_name': self.model_name})
                self._embed_model = SentenceTransformer(self.model_name)
            except Exception as e:
                log.error('error loading embedding model', extra={'model_name': self.model_name, 'error': str(e)})
                raise
        return self._embed_model

    def generate_query_embedding(self, query_text: str) -> List[float]:
        try:
            embedding = self.embed_model.encode(query_text)
            return [float(x) for x in embedding.tolist()]
        except Exception as e:
            log.error('error generating embeddings for query', extra={'query_text': query_text, 'error': str(e)})
            raise

    def query_semantic_functions(self, req, limit: int = 10) -> Dict[str, Any]:
        raise NotImplementedError('Subclass must implement query_semantic_functions()')

    def query_semantic_files(self, req, limit: int = 10) -> Dict[str, Any]:
        raise NotImplementedError('Subclass must implement query_semantic_files()')

    def query_filters(self, **filters) -> Dict[str, Any]:
        raise NotImplementedError('Subclass must implement query_filters()')

    def get_file_content(self, file_path: str, start_line: int = None, end_line: int = None) -> bytes:
        raise NotImplementedError('Subclass must implement get_file_content()')

    def get_files_by_name(self, filename: str, max_results: int = 10) -> List[Dict[str, Any]]:
        raise NotImplementedError('Subclass must implement get_files_by_name()')

    def get_function_content(self, function_id: str) -> str:
        raise NotImplementedError('Subclass must implement get_function_content()')

    def get_available_files(self) -> List[str]:
        raise NotImplementedError('Subclass must implement get_available_files()')

    def get_function_callers(self, function_name: str) -> Dict[str, Any]:
        raise NotImplementedError('Subclass must implement get_function_callers()')

    def get_function_called_by(self, function_name: str) -> Dict[str, Any]:
        raise NotImplementedError('Subclass must implement get_function_called_by()')
