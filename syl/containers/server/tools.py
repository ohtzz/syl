import asyncio
import concurrent.futures

from loguru import logger as log
from typing import Dict, Any, List, Optional
from mcp.types import Tool

from .registrations import get_registered_datastores

from syl.common import DockerManager
from syl.common.datastores import Status
from syl.common.docker import SYL_INDEX_PREFIX
from syl.common.tools import SearchSemanticFunctionsRequest, SearchSemanticFilesRequest, TOOL_DEFINITIONS


docker = DockerManager()


def run_async_tool(tool_name: str, arguments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Helper function to run async call_tool from synchronous MCP tools"""
    try:
        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, call_tool(tool_name, arguments))
                result = future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run()
            result = asyncio.run(call_tool(tool_name, arguments))

        if 'error' in result:
            return {'error': result['error']}

        return result

    except Exception as e:
        log.error(f'Error in MCP tool {tool_name}: {str(e)}')
        return {'error': f'Failed to execute {tool_name}: {str(e)}'}


def get_datastore_container(datastore_name: str):
    containers = get_datastore_containers()

    log.info(
        'Found datastore containers',
        extra={
            'names': [c.name for c in containers],
        },
    )

    return next((c for c in containers if c.name.removeprefix(SYL_INDEX_PREFIX) == datastore_name), None)


def get_datastore_containers():
    return docker.list_indexing_container()


def get_datastore_names() -> List[str]:
    return [c.name.removeprefix(SYL_INDEX_PREFIX) for c in get_datastore_containers()]


def get_openai_tools() -> List[Dict]:
    """Convert Pydantic models to OpenAI tool format"""
    tools = []
    datastore_names = get_datastore_names()

    for tool_name, tool_def in TOOL_DEFINITIONS.items():
        schema = tool_def['model'].model_json_schema()

        if 'properties' in schema and 'datastore_name' in schema['properties']:
            schema['properties']['datastore_name']['enum'] = datastore_names

        openai_tool = {
            'type': 'function',
            'function': {'name': tool_name, 'description': tool_def['description'], 'parameters': schema},
        }
        tools.append(openai_tool)

    return tools


def get_mcp_tools() -> List[Tool]:
    """Convert Pydantic models to MCP tool format"""
    tools = []
    datastore_names = get_datastore_names()

    for tool_name, tool_def in TOOL_DEFINITIONS.items():
        # Get the schema from the Pydantic model
        schema = tool_def['model'].model_json_schema()

        # Update datastore_name enum with current workers
        if 'properties' in schema and 'datastore_name' in schema['properties']:
            schema['properties']['datastore_name']['enum'] = datastore_names

        mcp_tool = Tool(name=tool_name, description=tool_def['description'], inputSchema=schema)
        tools.append(mcp_tool)

    return tools


async def call_tool(tool_name: str, arguments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Route tool call to appropriate worker"""
    log.info(
        'Calling tool',
        extra={
            'tool': tool_name,
            'arguments': arguments,
        },
    )

    registered_datastores = await get_registered_datastores()

    if tool_name == 'list_datastores':
        return {
            'result': {
                'datastores': [
                    {'name': k, 'description': v.datastore.description}
                    for k, v in registered_datastores.items()
                    if v.status == Status.registered
                ]
            }
        }

    # Everything else requires args
    if not arguments:
        return {'error': 'Tool call arguments missing'}

    try:
        datastore_name = arguments.get('datastore_name')
        if not datastore_name:
            return {'error': 'datastore_name is required'}

        datastore_config = registered_datastores.get(datastore_name)
        if not datastore_config:
            return {'error': 'datastore_name not found'}

        data_conn = datastore_config.datastore._data_provider_conn

        # Route tool calls
        if tool_name == 'search_function_code_semantic':
            req = SearchSemanticFunctionsRequest(**arguments)
            func_list = data_conn.query_semantic_functions(req)
            return {'result': func_list}

        elif tool_name == 'search_file_code_semantic':
            req = SearchSemanticFilesRequest(**arguments)
            file_list = data_conn.query_semantic_files(req)
            return {'result': file_list}

        elif tool_name == 'search_by_function_name':
            func_list = data_conn.query_filters(
                function_name=arguments['function_name'], max_results=arguments.get('max_results', 10)
            )
            return {'result': func_list}

        elif tool_name == 'get_function_callers':
            func_list = data_conn.get_function_callers(arguments['function_name'])
            return {'result': func_list}

        elif tool_name == 'get_function_called_by':
            func_list = data_conn.get_function_called_by(arguments['function_name'])
            return {'result': func_list}

        elif tool_name == 'get_complex_functions':
            func_list = data_conn.query_filters(
                min_complexity=arguments.get('min_complexity', 5), max_results=arguments.get('max_results', 20)
            )
            return {'result': func_list}

        elif tool_name == 'get_async_functions':
            func_list = data_conn.query_filters(is_async=True, max_results=arguments.get('max_results', 10))
            return {'result': func_list}

        elif tool_name == 'get_functions_in_file':
            func_list = data_conn.query_filters(
                filename=arguments['filename'], max_results=arguments.get('max_results', 10)
            )
            return {'result': func_list}

        elif tool_name == 'get_file_content':
            file_content = data_conn.get_file_content(
                file_path=arguments['file_path'], line_start=arguments.get('line_start'), line_end=arguments.get('line_end')
            )
            return {'result': file_content}

        elif tool_name == 'get_files_by_name':
            files = data_conn.get_files_by_name(filename=arguments['filename'], max_results=arguments.get('max_results'))
            return {'result': files}

        else:
            return {'error': f'unknown tool: {tool_name}'}

    except Exception as e:
        log.error(f'Failed to call tool {tool_name}: {str(e)}')
        return {'error': f'Failed to call tool: {str(e)}'}
