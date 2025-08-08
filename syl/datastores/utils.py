from pathlib import Path
from typing import Dict, Any, List
from loguru import logger as log

from .constants import (
    CODE_EXTENSIONS,
    DOCS_EXTENSIONS,
    CONFIG_EXTENSIONS,
    DATA_EXTENSIONS,
    SUPPORTED_FILETYPES,
    MAX_DOC_SIZE,
    MAX_CONFIG_SIZE,
    MAX_DATA_SIZE,
)


def classify_file_type(file_path: Path) -> str:
    ext = file_path.suffix.lower()

    if ext in CODE_EXTENSIONS:
        return 'code'
    elif ext in DOCS_EXTENSIONS:
        return 'docs'
    elif ext in CONFIG_EXTENSIONS:
        return 'config'
    elif ext in DATA_EXTENSIONS:
        return 'data'
    else:
        return 'other'


def should_process_file(file_path: Path) -> bool:
    ext = file_path.suffix.lower()

    if ext not in SUPPORTED_FILETYPES:
        log.debug(f'Skipping unsupported filetype: {file_path}')
        return False

    try:
        file_size = file_path.stat().st_size

        # Allow zero-byte files here as the presence of
        # things like __init__.py are useful to know
        # about, just skip generating embeddings for the
        # file later on.
        #
        # if file_size == 0:
        #     return False

        if ext in CODE_EXTENSIONS:
            return True
        elif ext in DOCS_EXTENSIONS:
            return file_size <= MAX_DOC_SIZE
        elif ext in CONFIG_EXTENSIONS:
            return file_size <= MAX_CONFIG_SIZE
        elif ext in DATA_EXTENSIONS:
            return file_size <= MAX_DATA_SIZE
        else:
            return file_size <= MAX_DOC_SIZE

    except OSError:
        log.warning(f'Could not get file size for {file_path}')
        return False


def generate_file_id(file_path: str) -> str:
    """Generate consistent file ID from path"""
    return file_path.replace('/', '_').replace('\\', '_')


def extract_filename(file_path: str) -> str:
    """Extract filename handling both Unix and Windows paths"""
    return Path(file_path).name


class ResultFormatter:
    """Common result formatter for all datastores"""

    @staticmethod
    def _format_returns_field(returns_value):
        """Handle returns field which can be string, list, or None"""
        if returns_value is None:
            return ''
        elif isinstance(returns_value, list):
            # If it's a list, join with commas or take first element
            if len(returns_value) == 0:
                return ''
            elif len(returns_value) == 1:
                return returns_value[0] or ''
            else:
                return ', '.join(str(r) for r in returns_value if r)
        else:
            return str(returns_value) if returns_value else ''

    @staticmethod
    def format_function_results(
        results: List[Dict[str, Any]],
        query_text: str = '',
        datastore_type: str = '',
        include_function_content: bool = False,
    ) -> Dict[str, Any]:
        """Format function query results to common structure"""
        formatted_results: Dict[str, Any] = {
            'query': query_text,
            'datastore': datastore_type,
            'total_results': len(results),
            'results': [],
        }

        for result in results:
            formatted_result = {
                'id': result.get('id', ''),
                'function_name': result.get('function_name', ''),
                'file_path': result.get('file_path', ''),
                'filename': result.get('filename', ''),
                'line_start': result.get('line_start', 0),
                'line_end': result.get('line_end', 0),
                'calls': result.get('calls', []),
                'called_by': result.get('called_by', []),
                'complexity': result.get('complexity', 0),
                'maintainability_index': result.get('maintainability_index', 0.0),
                'parameters': result.get('parameters', []),
                'returns': ResultFormatter._format_returns_field(result.get('returns')),
                'docstring': result.get('docstring', ''),
                'is_async': result.get('is_async', False),
                'is_method': result.get('is_method', False),
                'class_name': result.get('class_name', ''),
                'distance': result.get('distance', 0.0),
            }

            # Include code only if requested
            if include_function_content:
                content = ''

                # Handle different storage formats
                if result.get('document'):
                    # ChromaDB format
                    content = result['document']
                elif result.get('content'):
                    # pgvector uncompressed format
                    content = result['content']
                elif result.get('is_compressed') and result.get('content_compressed'):
                    # pgvector compressed format
                    import gzip
                    import base64

                    try:
                        compressed = base64.b64decode(result['content_compressed'].encode('ascii'))
                        content = gzip.decompress(compressed).decode('utf-8')
                    except Exception as e:
                        log.warning(
                            f'Failed to decompress function content for {result.get("function_name", "unknown")}: {e}'
                        )
                        content = ''

                if content:
                    formatted_result['code'] = content

            formatted_results['results'].append(formatted_result)

        return formatted_results

    @staticmethod
    def format_file_results(
        results: List[Dict[str, Any]],
        query_text: str = '',
        datastore_type: str = '',
        include_file_content: bool = False,
    ) -> Dict[str, Any]:
        """Format file query results to common structure"""
        formatted_results: Dict[str, Any] = {
            'query': query_text,
            'datastore': datastore_type,
            'total_results': len(results),
            'results': [],
        }

        for result in results:
            formatted_result = {
                'id': result.get('id', ''),
                'file_path': result.get('file_path', ''),
                'filename': result.get('filename', ''),
                'file_type': result.get('file_type', 'other'),
                'file_size': result.get('file_size', 0),
                'line_count': result.get('line_count', 0),
                'imports': result.get('imports', []),
                'distance': result.get('distance', 0.0),
            }

            # Include updated_at if it exists
            if result.get('updated_at'):
                formatted_result['updated_at'] = result['updated_at']

            # Include content only if requested
            if include_file_content:
                # Handle compressed content
                content = result.get('content', '')
                if result.get('is_compressed') and result.get('content_compressed'):
                    import gzip
                    import base64

                    try:
                        compressed = base64.b64decode(result['content_compressed'].encode('ascii'))
                        content = gzip.decompress(compressed).decode('utf-8')
                    except Exception as e:
                        log.warning(f'Failed to decompress content for {result.get("file_path", "unknown")}: {e}')
                        content = ''

                formatted_result['content'] = content

            formatted_results['results'].append(formatted_result)

        return formatted_results


class MetadataProcessor:
    """Common metadata processing for different datastore constraints"""

    @staticmethod
    def optimize_for_s3_vector(raw_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize metadata for S3 Vector constraints (max 10 keys, 2KB total)"""
        import json

        optimized = {}

        # Essential fields first
        if raw_metadata.get('function_name'):
            optimized['function_name'] = raw_metadata['function_name'][:100]

        if raw_metadata.get('file_path'):
            optimized['filename'] = extract_filename(raw_metadata['file_path'])

        if raw_metadata.get('complexity') is not None:
            optimized['complexity'] = int(raw_metadata['complexity'])

        # Optional fields
        if raw_metadata.get('calls'):
            optimized['function_calls'] = raw_metadata['calls'][:10]

        if raw_metadata.get('called_by'):
            optimized['called_by'] = raw_metadata['called_by'][:10]

        if raw_metadata.get('is_async') is not None:
            optimized['is_async'] = bool(raw_metadata['is_async'])

        if raw_metadata.get('is_method') is not None:
            optimized['is_method'] = bool(raw_metadata['is_method'])

        if raw_metadata.get('class_name'):
            optimized['class_name'] = raw_metadata['class_name'][:50]

        if 'code' in raw_metadata:
            optimized['bytes'] = len(raw_metadata['code'].encode('utf-8'))

        # Check size constraints
        metadata_json = json.dumps(optimized)
        metadata_size = len(metadata_json.encode('utf-8'))

        if metadata_size > 2048:
            log.warning(f'Metadata too large ({metadata_size} bytes), truncating...')
            # Remove optional fields until under limit
            optional_fields = ['class_name', 'is_async', 'is_method', 'function_calls', 'called_by']
            for field in optional_fields:
                if field in optimized:
                    del optimized[field]
                    metadata_json = json.dumps(optimized)
                    metadata_size = len(metadata_json.encode('utf-8'))
                    if metadata_size <= 2048:
                        break

        log.debug(f'Optimized metadata size: {metadata_size} bytes, keys: {len(optimized)}')
        return optimized
