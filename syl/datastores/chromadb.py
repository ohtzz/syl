import chromadb
import os
import json

from typing import Dict, Any, List, Optional
from loguru import logger as log
from .datastore import DataStoreLoader, DataStoreQuery
from .utils import generate_file_id, ResultFormatter
from syl.parsers.python import FunctionEmbedding


class ChromaDBLoader(DataStoreLoader):
    def __init__(
        self,
        embed_model: str,
        num_workers: int,
        file_ext_whitelist: Optional[List[str]],
        file_ext_blacklist: Optional[List[str]],
        git_repo_url: Optional[str],
        git_branch: Optional[str],
        vector_size: int,
        persistent_client: bool,
        data_path: Optional[str] = '/data',
        collection_name: str = 'syl',
    ):
        super().__init__(
            embed_model=embed_model,
            num_workers=num_workers,
            file_ext_whitelist=file_ext_whitelist,
            file_ext_blacklist=file_ext_blacklist,
            git_repo_url=git_repo_url,
            git_branch=git_branch,
            vector_size=vector_size,
        )

        if persistent_client:
            self.chroma_client = chromadb.PersistentClient(path=data_path)
        else:
            self.chroma_client = chromadb.EphemeralClient()

        # Funcs
        try:
            self.functions_collection = self.chroma_client.get_collection(
                name=f'{collection_name}_functions', embedding_function=None
            )
            log.info(f'Using existing ChromaDB functions collection: {collection_name}_functions')
        except chromadb.errors.NotFoundError:
            self.functions_collection = self.chroma_client.create_collection(
                name=f'{collection_name}_functions', embedding_function=None
            )
            log.info(f'Created new ChromaDB functions collection: {collection_name}_functions')

        # Files
        try:
            self.files_collection = self.chroma_client.get_collection(
                name=f'{collection_name}_files', embedding_function=None
            )
            log.info(f'Using existing ChromaDB files collection: {collection_name}_files')
        except chromadb.errors.NotFoundError:
            self.files_collection = self.chroma_client.create_collection(
                name=f'{collection_name}_files', embedding_function=None
            )
            log.info(f'Created new ChromaDB files collection: {collection_name}_files')

    def test_connection(self):
        try:
            self.chroma_client.heartbeat()
            log.info('ChromaDB connection test successful')
        except Exception as e:
            log.error(f'ChromaDB connection test failed: {e}')
            raise

    def load_functions(self, funcs: List[FunctionEmbedding]):
        if not funcs:
            log.warning('No functions to load into ChromaDB')
            return

        embeddings = []
        metadatas = []
        documents = []
        ids = []

        for func in funcs:
            embeddings.append(func.embedding)

            metadata = {
                'function_name': func.function_name,
                'file_path': func.file_path,
                'filename': os.path.basename(func.file_path),
                'line_start': func.line_start,
                'line_end': func.line_end,
                'calls': func.calls,
                'called_by': func.called_by,
                'complexity': func.complexity,
                'maintainability_index': func.maintainability_index,
                'parameters': func.parameters,
                'returns': func.returns,
                'docstring': func.docstring,
                'is_async': func.is_async,
                'is_method': func.is_method,
                'class_name': func.class_name,
            }

            # Dump any list values or ChromaDB complains
            for key, value in metadata.items():
                if isinstance(value, list):
                    metadata[key] = json.dumps(value) if value else None

            # Also hates None..
            metadata = {k: v for k, v in metadata.items() if v is not None}

            metadatas.append(metadata)
            documents.append(func.code)
            ids.append(func.id)

        try:
            # First, delete existing functions from the same file to handle updates
            file_paths = list(set(func.file_path for func in funcs))
            for file_path in file_paths:
                try:
                    existing = self.functions_collection.get(where={'file_path': {'$eq': file_path}})
                    if existing['ids']:
                        self.functions_collection.delete(ids=existing['ids'])
                        log.info(f'Deleted {len(existing["ids"])} existing functions from {file_path}')
                except Exception as e:
                    log.warning(f'Could not delete existing functions from {file_path}: {e}')

            # Add new/updated functions
            self.functions_collection.add(embeddings=embeddings, metadatas=metadatas, documents=documents, ids=ids)
            log.info(f'Successfully loaded {len(funcs)} functions to ChromaDB')
        except Exception as e:
            log.error(f'Failed to load functions to ChromaDB: {e}')
            raise

    def load_file_embeddings(self, file_data_with_embeddings):
        """ChromaDB handles file embeddings through the files collection"""
        if not file_data_with_embeddings:
            log.warning('No file embeddings to load into ChromaDB')
            return

        metadatas = []
        documents = []
        ids = []
        embeddings = []

        for file_data in file_data_with_embeddings:
            if (
                file_data.embedding is None
                or (hasattr(file_data.embedding, 'shape') and file_data.embedding.shape == ())
                or str(file_data.embedding) == 'nan'
            ):
                log.warning(f'Skipping file {file_data.file_path} due to invalid embedding: {file_data.embedding}')
                continue

            # Use content if available, otherwise decompress
            content = file_data.content
            if not content and file_data.content_compressed:
                content = file_data.decompress_content()

            metadata = {
                'file_path': file_data.file_path,
                'filename': file_data.filename,
                'imports': file_data.imports or [],
                'file_size': file_data.file_size,
                'line_count': file_data.line_count,
                'file_type': file_data.file_type,
            }

            # Dump any list values or ChromaDB complains
            for key, value in metadata.items():
                if isinstance(value, list):
                    metadata[key] = json.dumps(value) if value else None

            # Also hates None..
            metadata = {k: v for k, v in metadata.items() if v is not None}

            metadatas.append(metadata)
            documents.append(content)
            ids.append(file_data.id)
            embeddings.append(file_data.embedding)

        try:
            if not embeddings:
                log.warning('No valid file embeddings to load into ChromaDB after filtering')
                return

            for file_id in ids:
                try:
                    existing = self.files_collection.get(ids=[file_id])
                    if existing['ids']:
                        self.files_collection.delete(ids=[file_id])
                        log.debug(f'Deleted existing file embedding: {file_id}')
                except Exception:
                    log.debug(f'No existing file embedding to delete: {file_id}')

            self.files_collection.add(metadatas=metadatas, documents=documents, ids=ids, embeddings=embeddings)
            log.info(f'Successfully loaded {len(embeddings)} file embeddings to ChromaDB')
        except Exception as e:
            log.error(f'Failed to load file embeddings to ChromaDB: {e}')
            raise


class ChromaDBQuery(DataStoreQuery):
    def __init__(
        self,
        model_name: str,
        vector_size: int,
        persistent_client: bool,
        data_path: Optional[str] = '/data',
        collection_name: Optional[str] = 'syl',
    ):
        super().__init__(model_name=model_name)

        if persistent_client:
            self.chroma_client = chromadb.PersistentClient(path=data_path)
        else:
            self.chroma_client = chromadb.EphemeralClient()

        # Funcs
        try:
            self.functions_collection = self.chroma_client.get_collection(
                name=f'{collection_name}_functions', embedding_function=None
            )
            log.info(f'Using existing ChromaDB functions collection: {collection_name}_functions')
        except chromadb.errors.NotFoundError:
            self.functions_collection = self.chroma_client.create_collection(
                name=f'{collection_name}_functions', embedding_function=None
            )
            log.info(f'Created new ChromaDB functions collection: {collection_name}_functions')

        # Files
        try:
            self.files_collection = self.chroma_client.get_collection(
                name=f'{collection_name}_files', embedding_function=None
            )
            log.info(f'Using existing ChromaDB files collection: {collection_name}_files')
        except chromadb.errors.NotFoundError:
            self.files_collection = self.chroma_client.create_collection(
                name=f'{collection_name}_files', embedding_function=None
            )
            log.info(f'Created new ChromaDB files collection: {collection_name}_files')

    def query_semantic_functions(self, req, limit: int = 10) -> Dict[str, Any]:
        try:
            query_embeddings = self.generate_query_embedding(req.query)
            where_clause = self._build_where_clause(req)

            query_params = {
                'query_embeddings': [query_embeddings],
                'n_results': min(limit, req.max_results),
                'include': ['metadatas', 'documents', 'distances'],
            }

            if where_clause:
                query_params['where'] = where_clause

            results = self.functions_collection.query(**query_params)

            result_dicts = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}

                    result_dict = {
                        'id': results['ids'][0][i],
                        'function_name': metadata.get('function_name', ''),
                        'file_path': metadata.get('file_path', ''),
                        'filename': metadata.get('filename', ''),
                        'line_start': metadata.get('line_start', 0),
                        'line_end': metadata.get('line_end', 0),
                        'calls': metadata.get('calls', []),
                        'called_by': metadata.get('called_by', []),
                        'complexity': metadata.get('complexity', 0),
                        'maintainability_index': metadata.get('maintainability_index', 0.0),
                        'parameters': metadata.get('parameters', []),
                        'returns': metadata.get('returns', None),
                        'docstring': metadata.get('docstring', None),
                        'is_async': metadata.get('is_async', False),
                        'is_method': metadata.get('is_method', False),
                        'class_name': metadata.get('class_name', None),
                        'distance': results['distances'][0][i] if results['distances'] else 0.0,
                        'document': results['documents'][0][i] if results['documents'] else '',
                    }
                    result_dicts.append(result_dict)

            return ResultFormatter.format_function_results(
                result_dicts, req.query, 'chromadb', req.include_function_content
            )

        except Exception as e:
            log.error(f'Error querying ChromaDB: {e}')
            raise

    def query_semantic_files(self, req, limit: int = 10) -> Dict[str, Any]:
        try:
            query_embeddings = self.generate_query_embedding(req.query)
            where_clause = self._build_file_where_clause(req)

            query_params = {
                'query_embeddings': [query_embeddings],
                'n_results': min(limit, req.max_results),
                'include': ['metadatas', 'documents', 'distances'],
            }

            if where_clause:
                query_params['where'] = where_clause

            results = self.files_collection.query(**query_params)

            result_dicts = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}

                    result_dict = {
                        'id': results['ids'][0][i],
                        'file_path': metadata.get('file_path', ''),
                        'filename': metadata.get('filename', ''),
                        'content': results['documents'][0][i] if results['documents'] else '',
                        'imports': metadata.get('imports', []),
                        'file_size': metadata.get('file_size', 0),
                        'line_count': metadata.get('line_count', 0),
                        'file_type': metadata.get('file_type', 'other'),
                        'distance': results['distances'][0][i] if results['distances'] else 0.0,
                    }
                    result_dicts.append(result_dict)

            return ResultFormatter.format_file_results(result_dicts, req.query, 'chromadb', req.include_file_content)

        except Exception as e:
            log.error(f'Error querying ChromaDB files: {e}')
            raise

    def _build_file_where_clause(self, req) -> Dict[str, Any]:
        """Build ChromaDB where clause for file search from request filters"""
        conditions = []

        if hasattr(req, 'file_type_filter') and req.file_type_filter:
            conditions.append({'file_type': {'$eq': req.file_type_filter}})

        if hasattr(req, 'filename_filter') and req.filename_filter:
            conditions.append({'filename': {'$eq': req.filename_filter}})

        if not conditions:
            return {}
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {'$and': conditions}

    def _build_where_clause(self, req) -> Dict[str, Any]:
        """Build ChromaDB where clause from request filters"""
        conditions = []

        if req.file_filter:
            conditions.append({'filename': {'$eq': req.file_filter}})

        if req.function_filter:
            conditions.append({'function_name': {'$eq': req.function_filter}})

        if req.complexity_filter is not None:
            conditions.append({'complexity': {'$gte': req.complexity_filter}})

        if req.is_async_filter is not None:
            conditions.append({'is_async': {'$eq': req.is_async_filter}})

        if req.is_method_filter is not None:
            conditions.append({'is_method': {'$eq': req.is_method_filter}})

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]
        else:
            return {'$and': conditions}

    def query_filters(self, **filters) -> Dict[str, Any]:
        """Query using metadata filters only (without semantic search)"""
        try:
            where_clause = {}

            # Map common filter names to ChromaDB format
            if 'function_name' in filters:
                where_clause['function_name'] = {'$eq': filters['function_name']}
            if 'filename' in filters:
                where_clause['filename'] = {'$eq': filters['filename']}
            if 'complexity' in filters:
                where_clause['complexity'] = {'$gte': filters['complexity']}
            if 'is_async' in filters:
                where_clause['is_async'] = {'$eq': filters['is_async']}
            if 'is_method' in filters:
                where_clause['is_method'] = {'$eq': filters['is_method']}
            if 'class_name' in filters:
                where_clause['class_name'] = {'$eq': filters['class_name']}

            if not where_clause:
                log.warning('No valid filters provided for query_filters')
                return {'results': [], 'total_count': 0}

            # Use get() method to retrieve documents by metadata
            results = self.functions_collection.get(where=where_clause, include=['metadatas', 'documents'])

            # Convert to expected format
            result_dicts = []
            if results['ids']:
                for i, doc_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i] if results['metadatas'] else {}

                    result_dict = {
                        'id': doc_id,
                        'function_name': metadata.get('function_name', ''),
                        'file_path': metadata.get('file_path', ''),
                        'filename': metadata.get('filename', ''),
                        'line_start': metadata.get('line_start', 0),
                        'line_end': metadata.get('line_end', 0),
                        'calls': metadata.get('calls', []),
                        'called_by': metadata.get('called_by', []),
                        'complexity': metadata.get('complexity', 0),
                        'maintainability_index': metadata.get('maintainability_index', 0.0),
                        'parameters': metadata.get('parameters', []),
                        'returns': metadata.get('returns', None),
                        'docstring': metadata.get('docstring', None),
                        'is_async': metadata.get('is_async', False),
                        'is_method': metadata.get('is_method', False),
                        'class_name': metadata.get('class_name', None),
                        'distance': 0.0,  # No distance for filter-based queries
                        'document': results['documents'][i] if results['documents'] else '',
                    }
                    result_dicts.append(result_dict)

            return ResultFormatter.format_function_results(result_dicts, 'filter-based query', 'chromadb', False)

        except Exception as e:
            log.error(f'Error querying ChromaDB with filters: {e}')
            raise

    def get_file_content(self, file_path: str, start_line: int = None, end_line: int = None) -> bytes:
        """Get file content from ChromaDB files collection"""
        try:
            file_id = generate_file_id(file_path)
            results = self.files_collection.get(ids=[file_id], include=['metadatas', 'documents'])

            if not results['ids']:
                raise FileNotFoundError(f'File not found: {file_path}')

            metadata = results['metadatas'][0]
            document = results['documents'][0]

            if metadata.get('is_compressed', False):
                import gzip
                import base64

                compressed = base64.b64decode(document.encode('ascii'))
                content = gzip.decompress(compressed).decode('utf-8')
            else:
                content = document

            if start_line is None and end_line is None:
                return content.encode('utf-8')

            lines = content.split('\n')
            start_idx = max(0, (start_line - 1) if start_line else 0)
            end_idx = min(len(lines), end_line if end_line else len(lines))

            selected_lines = lines[start_idx:end_idx]
            result_content = '\n'.join(selected_lines)
            return result_content.encode('utf-8')

        except Exception as e:
            log.error(f'Error retrieving file content from ChromaDB for {file_path}: {e}')
            raise

    def get_available_files(self) -> List[str]:
        """Get list of all files stored in ChromaDB"""
        try:
            results = self.files_collection.get(include=['metadatas'])

            file_paths = []
            if results['metadatas']:
                for metadata in results['metadatas']:
                    file_path = metadata.get('file_path')
                    if file_path:
                        file_paths.append(file_path)

            return sorted(file_paths)

        except Exception as e:
            log.error(f'Error getting available files from ChromaDB: {e}')
            raise

    def get_file_functions(self, file_path: str) -> List[Dict[str, Any]]:
        """Get all functions in a specific file with their metadata"""
        try:
            results = self.functions_collection.get(
                where={'file_path': {'$eq': file_path}}, include=['metadatas', 'documents']
            )

            functions = []
            if results['ids']:
                for i, doc_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i] if results['metadatas'] else {}
                    document = results['documents'][i] if results['documents'] else ''

                    function_info = {
                        'id': doc_id,
                        'function_name': metadata.get('function_name', ''),
                        'line_start': metadata.get('line_start', 0),
                        'line_end': metadata.get('line_end', 0),
                        'is_async': metadata.get('is_async', False),
                        'is_method': metadata.get('is_method', False),
                        'class_name': metadata.get('class_name', None),
                        'complexity': metadata.get('complexity', 0),
                        'parameters': metadata.get('parameters', []),
                        'docstring': metadata.get('docstring', None),
                        'code': document,
                    }
                    functions.append(function_info)

            functions.sort(key=lambda x: x['line_start'])
            return functions

        except Exception as e:
            log.error(f'Error getting functions for file {file_path} from ChromaDB: {e}')
            raise

    def get_function_content(self, function_id: str) -> str:
        """Get the code content of a specific function"""
        try:
            results = self.functions_collection.get(ids=[function_id], include=['documents'])

            if not results['ids']:
                raise ValueError(f'Function not found: {function_id}')

            return results['documents'][0]

        except Exception as e:
            log.error(f'Error getting function content for {function_id} from ChromaDB: {e}')
            raise

    def get_function_callers(self, function_name: str) -> Dict[str, Any]:
        """Get all functions that call the specified function"""
        try:
            # Query for functions where the 'calls' array contains the function_name
            results = self.functions_collection.get(
                where={'calls': {'$contains': function_name}}, include=['metadatas', 'documents']
            )

            result_dicts = []
            if results['ids']:
                for i, doc_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i] if results['metadatas'] else {}
                    document = results['documents'][i] if results['documents'] else ''

                    result_dict = {
                        'id': doc_id,
                        'function_name': metadata.get('function_name', ''),
                        'file_path': metadata.get('file_path', ''),
                        'filename': metadata.get('filename', ''),
                        'line_start': metadata.get('line_start', 0),
                        'line_end': metadata.get('line_end', 0),
                        'calls': metadata.get('calls', []),
                        'called_by': metadata.get('called_by', []),
                        'complexity': metadata.get('complexity', 0),
                        'maintainability_index': metadata.get('maintainability_index', 0.0),
                        'parameters': metadata.get('parameters', []),
                        'returns': metadata.get('returns', None),
                        'docstring': metadata.get('docstring', None),
                        'is_async': metadata.get('is_async', False),
                        'is_method': metadata.get('is_method', False),
                        'class_name': metadata.get('class_name', None),
                        'distance': 0.0,  # No distance for relationship queries
                        'document': document,
                    }
                    result_dicts.append(result_dict)

            return ResultFormatter.format_function_results(
                result_dicts, f'Functions that call {function_name}', 'chromadb', False
            )

        except Exception as e:
            log.error(f'Error getting function callers for {function_name} from ChromaDB: {e}')
            raise

    def get_function_called_by(self, function_name: str) -> Dict[str, Any]:
        """Get all functions called by the specified function"""
        try:
            # Query for functions where the 'called_by' array contains the function_name
            results = self.functions_collection.get(
                where={'called_by': {'$contains': function_name}}, include=['metadatas', 'documents']
            )

            result_dicts = []
            if results['ids']:
                for i, doc_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i] if results['metadatas'] else {}
                    document = results['documents'][i] if results['documents'] else ''

                    result_dict = {
                        'id': doc_id,
                        'function_name': metadata.get('function_name', ''),
                        'file_path': metadata.get('file_path', ''),
                        'filename': metadata.get('filename', ''),
                        'line_start': metadata.get('line_start', 0),
                        'line_end': metadata.get('line_end', 0),
                        'calls': metadata.get('calls', []),
                        'called_by': metadata.get('called_by', []),
                        'complexity': metadata.get('complexity', 0),
                        'maintainability_index': metadata.get('maintainability_index', 0.0),
                        'parameters': metadata.get('parameters', []),
                        'returns': metadata.get('returns', None),
                        'docstring': metadata.get('docstring', None),
                        'is_async': metadata.get('is_async', False),
                        'is_method': metadata.get('is_method', False),
                        'class_name': metadata.get('class_name', None),
                        'distance': 0.0,  # No distance for relationship queries
                        'document': document,
                    }
                    result_dicts.append(result_dict)

            return ResultFormatter.format_function_results(
                result_dicts, f'Functions called by {function_name}', 'chromadb', False
            )

        except Exception as e:
            log.error(f'Error getting functions called by {function_name} from ChromaDB: {e}')
            raise
