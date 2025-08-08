from loguru import logger as log
from typing import Dict, Any, Optional, Union
from fastmcp import FastMCP

from .tools import run_async_tool
from syl.common.tools import TOOL_DEFINITIONS


mcp_server = FastMCP('code-search-server')


def _generate_schema_from_pydantic_model(model_class) -> Dict[str, Any]:
    """Generate JSON schema from Pydantic model"""
    return model_class.model_json_schema()


@mcp_server.resource('schema://list_datastores')
def list_datastores_schema() -> Dict[str, Any]:
    """Schema for list_datastores function arguments"""
    return {
        'type': 'object',
        'properties': {},
        'required': [],
        'description': 'No arguments required for listing datastores',
    }


for tool_name, tool_config in TOOL_DEFINITIONS.items():
    model_class = tool_config['model']
    description = tool_config['description']

    def create_schema_resource(model_cls, desc):
        def schema_func() -> Dict[str, Any]:
            schema = _generate_schema_from_pydantic_model(model_cls)
            schema['description'] = desc
            return schema

        return schema_func

    schema_resource = create_schema_resource(model_class, description)
    schema_resource.__name__ = f'{tool_name}_schema'
    schema_resource.__doc__ = f'Schema for {tool_name} function arguments'

    mcp_server.resource(f'schema://{tool_name}')(schema_resource)


@mcp_server.tool()
def list_datastores() -> Dict[str, Any]:
    """List available code-search datastores"""
    try:
        return run_async_tool('list_datastores', arguments=None)
    except Exception as e:
        log.error(f'Error in MCP list_datastores: {str(e)}')
        return {'error': f'Failed to list datastores: {str(e)}'}


@mcp_server.tool()
def search_function_code_semantic(
    datastore_name: str,
    query: str,
    max_results: Optional[Union[int, str]] = 5,
    file_filter: Optional[str] = None,
    function_filter: Optional[str] = None,
    complexity_filter: Optional[Union[int, str]] = None,
    is_async_filter: Optional[Union[bool, str]] = None,   # Handle bool or str because Claude Code sends bools as strs # TODO could be an issue with @mcp_server resources
    is_method_filter: Optional[Union[bool, str]] = None,
    include_function_content: Optional[Union[bool, str]] = False,
) -> Dict[str, Any]:
    """Search functions using semantic similarity"""
    try:
        arguments = {
            'datastore_name': datastore_name,
            'query': query,
            'max_results': _normalize_int(max_results),
            'file_filter': file_filter,
            'function_filter': function_filter,
            'complexity_filter': _normalize_int(complexity_filter),
            'is_async_filter': _normalize_bool(is_async_filter),
            'is_method_filter': _normalize_bool(is_method_filter),
            'include_function_content': _normalize_bool(include_function_content),
        }

        arguments = {k: v for k, v in arguments.items() if v is not None}
        return run_async_tool('search_function_code_semantic', arguments)

    except Exception as e:
        log.error(f'Error in MCP search_function_code_semantic: {str(e)}')
        return {'error': f'Failed to execute function semantic search: {str(e)}'}


@mcp_server.tool()
def search_file_code_semantic(
    datastore_name: str,
    query: str,
    max_results: Optional[Union[int, str]] = 5,
    file_type_filter: Optional[str] = None,
    filename_filter: Optional[str] = None,
    include_file_content: Optional[Union[bool, str]] = False,
) -> Dict[str, Any]:
    """Search files using semantic similarity"""
    try:
        arguments = {
            'datastore_name': datastore_name,
            'query': query,
            'max_results': _normalize_int(max_results),
            'file_type_filter': file_type_filter,
            'filename_filter': filename_filter,
            'include_file_content': _normalize_bool(include_file_content),
        }

        arguments = {k: v for k, v in arguments.items() if v is not None}
        return run_async_tool('search_file_code_semantic', arguments)

    except Exception as e:
        log.error(f'Error in MCP search_file_code_semantic: {str(e)}')
        return {'error': f'Failed to execute file semantic search: {str(e)}'}


@mcp_server.tool()
def search_by_function_name(datastore_name: str, function_name: str) -> Dict[str, Any]:
    """Find a specific function by its exact name"""
    try:
        arguments = {'datastore_name': datastore_name, 'function_name': function_name}
        return run_async_tool('search_by_function_name', arguments)

    except Exception as e:
        log.error(f'Error in MCP search_by_function_name: {str(e)}')
        return {'error': f'Failed to execute function search: {str(e)}'}


@mcp_server.tool()
def get_function_callers(datastore_name: str, function_name: str) -> Dict[str, Any]:
    """Get all functions that call a specific function"""
    try:
        arguments = {'datastore_name': datastore_name, 'function_name': function_name}
        return run_async_tool('get_function_callers', arguments)

    except Exception as e:
        log.error(f'Error in MCP get_function_callers: {str(e)}')
        return {'error': f'Failed to get function callers: {str(e)}'}


@mcp_server.tool()
def get_function_called_by(datastore_name: str, function_name: str) -> Dict[str, Any]:
    """Get all functions called by a specific function"""
    try:
        arguments = {'datastore_name': datastore_name, 'function_name': function_name}
        return run_async_tool('get_function_called_by', arguments)

    except Exception as e:
        log.error(f'Error in MCP get_function_called_by: {str(e)}')
        return {'error': f'Failed to get functions called by: {str(e)}'}


@mcp_server.tool()
def get_complex_functions(datastore_name: str, min_complexity: Optional[Union[int, str]] = 5, max_results: Optional[Union[int, str]] = 5) -> Dict[str, Any]:
    """Get functions above a certain complexity threshold"""
    try:
        arguments = {'datastore_name': datastore_name, 'min_complexity': _normalize_int(min_complexity), 'max_results': _normalize_int(max_results)}
        return run_async_tool('get_complex_functions', arguments)

    except Exception as e:
        log.error(f'Error in MCP get_complex_functions: {str(e)}')
        return {'error': f'Failed to get complex functions: {str(e)}'}


@mcp_server.tool()
def get_async_functions(datastore_name: str) -> Dict[str, Any]:
    """Get all async/await functions in the codebase"""
    try:
        arguments = {'datastore_name': datastore_name}
        return run_async_tool('get_async_functions', arguments)

    except Exception as e:
        log.error(f'Error in MCP get_async_functions: {str(e)}')
        return {'error': f'Failed to get async functions: {str(e)}'}


@mcp_server.tool()
def get_functions_in_file(datastore_name: str, filename: str) -> Dict[str, Any]:
    """Get all functions defined in a specific file"""
    try:
        arguments = {'datastore_name': datastore_name, 'filename': filename}
        return run_async_tool('get_functions_in_file', arguments)

    except Exception as e:
        log.error(f'Error in MCP get_functions_in_file: {str(e)}')
        return {'error': f'Failed to get functions in file: {str(e)}'}


@mcp_server.tool()
def get_file_content(
    datastore_name: str, file_path: str, start_line: Optional[Union[int, str]] = None, end_line: Optional[Union[int, str]] = None
) -> Dict[str, Any]:
    """Retrieve the contents of a file by path, with optional start/end line numbers"""
    try:
        arguments = {
            'datastore_name': datastore_name,
            'file_path': file_path,
            'start_line': _normalize_int(start_line),
            'end_line': _normalize_int(end_line),
        }
        arguments = {k: v for k, v in arguments.items() if v is not None}
        return run_async_tool('get_file_content', arguments)

    except Exception as e:
        log.error(f'Error in MCP get_file_content: {str(e)}')
        return {'error': f'Failed to get file contents: {str(e)}'}


@mcp_server.tool()
def get_files_by_name(datastore_name: str, filename: str, max_results: Optional[Union[int, str]] = 5) -> Dict[str, Any]:
    """Retrieve files by name"""
    try:
        arguments = {
            'datastore_name': datastore_name,
            'filename': filename,
            'max_results': _normalize_int(max_results),
        }
        arguments = {k: v for k, v in arguments.items() if v is not None}
        return run_async_tool('get_files_by_name', arguments)

    except Exception as e:
        log.error(f'Error in MCP get_files_by_name: {str(e)}')
        return {'error': f'Failed to get files by name: {str(e)}'}


def _normalize_bool(value: Union[bool, str, None]) -> Optional[bool]:
    """Convert string boolean values to actual booleans"""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in ('true', '1',):
            return True
        elif value.lower() in ('false', '0'):
            return False
    return value


def _normalize_int(value: Union[int, str, None]) -> Optional[int]:
    """Convert string int values to actual ints"""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    return value
