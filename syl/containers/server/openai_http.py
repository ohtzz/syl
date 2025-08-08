from loguru import logger as log
from fastapi import FastAPI, HTTPException

from .tools import call_tool

from syl.common.tools import (
    SearchSemanticFunctionsRequest,
    SearchSemanticFilesRequest,
    SearchByFunctionNameRequest,
    GetFunctionCallersRequest,
    GetFunctionCalledByRequest,
    GetComplexFunctionsRequest,
    GetAsyncFunctionsRequest,
    GetFunctionsInFileRequest,
    GetFileContentRequest,
    GetFilesByNameRequest,
    ToolResponse,
)

openai_server = FastAPI(title='OpenAI-Compatible Tools Server')


@openai_server.post('/tools/list_datastores', response_model=ToolResponse)
async def list_datastores_openai():
    result = await call_tool(tool_name='list_datastores', arguments=None)
    return ToolResponse(result=result['result'])


@openai_server.post('/tools/search_function_code_semantic', response_model=ToolResponse)
async def search_function_code_semantic_openai(req: SearchSemanticFunctionsRequest):
    result = await call_tool('search_function_code_semantic', req.model_dump())
    if 'error' in result:
        log.error('error calling search_function_code_semantic', extra={'result': result['error']})
        raise HTTPException(status_code=400, detail=result['error'])
    return ToolResponse(result=result['result'])


@openai_server.post('/tools/search_file_code_semantic', response_model=ToolResponse)
async def search_file_code_semantic_openai(req: SearchSemanticFilesRequest):
    result = await call_tool('search_file_code_semantic', req.model_dump())
    if 'error' in result:
        log.error('error calling search_file_code_semantic', extra={'result': result['error']})
        raise HTTPException(status_code=400, detail=result['error'])
    return ToolResponse(result=result['result'])


@openai_server.post('/tools/search_by_function_name', response_model=ToolResponse)
async def search_by_function_name_openai(req: SearchByFunctionNameRequest):
    result = await call_tool('search_by_function_name', req.model_dump())
    if 'error' in result:
        log.error('error calling search_by_function_name', extra={'result': result})
        raise HTTPException(status_code=400, detail=result['error'])
    return ToolResponse(result=result['result'])


@openai_server.post('/tools/get_function_callers', response_model=ToolResponse)
async def get_function_callers(req: GetFunctionCallersRequest):
    result = await call_tool('get_function_callers', req.model_dump())
    if 'error' in result:
        log.error('error calling get_function_callers', extra={'result': result})
        raise HTTPException(status_code=400, detail=result['error'])
    return ToolResponse(result=result['result'])


@openai_server.post('/tools/get_function_called_by', response_model=ToolResponse)
async def get_function_called_by(req: GetFunctionCalledByRequest):
    result = await call_tool('get_function_called_by', req.model_dump())
    if 'error' in result:
        log.error('error calling get_function_called_by', extra={'result': result})
        raise HTTPException(status_code=400, detail=result['error'])
    return ToolResponse(result=result['result'])


@openai_server.post('/tools/get_complex_functions', response_model=ToolResponse)
async def get_complex_functions(req: GetComplexFunctionsRequest):
    result = await call_tool('get_complex_functions', req.model_dump())
    if 'error' in result:
        log.error('error calling get_complex_functions', extra={'result': result})
        raise HTTPException(status_code=400, detail=result['error'])
    return ToolResponse(result=result['result'])


@openai_server.post('/tools/get_async_functions', response_model=ToolResponse)
async def get_async_functions(req: GetAsyncFunctionsRequest):
    result = await call_tool('get_async_functions', req.model_dump())
    if 'error' in result:
        log.error('error calling get_async_functions', extra={'result': result})
        raise HTTPException(status_code=400, detail=result['error'])
    return ToolResponse(result=result['result'])


@openai_server.post('/tools/get_functions_in_file', response_model=ToolResponse)
async def get_functions_in_file(req: GetFunctionsInFileRequest):
    result = await call_tool('get_functions_in_file', req.model_dump())
    if 'error' in result:
        log.error('error calling get_functions_in_file', extra={'result': result})
        raise HTTPException(status_code=400, detail=result['error'])
    return ToolResponse(result=result['result'])


@openai_server.post('/tools/get_file_content', response_model=ToolResponse)
async def get_file_content_openai(req: GetFileContentRequest):
    result = await call_tool('get_file_content', req.model_dump())
    if 'error' in result:
        log.error('error calling get_file_content', extra={'result': result})
        raise HTTPException(status_code=400, detail=result['error'])
    return ToolResponse(result=result['result'])


@openai_server.post('/tools/get_files_by_name', response_model=ToolResponse)
async def get_files_by_name(req: GetFilesByNameRequest):
    result = await call_tool('get_files_by_name', req.model_dump())
    if 'error' in result:
        log.error('error calling get_files_by_name', extra={'result': result})
        raise HTTPException(status_code=400, detail=result['error'])
    return ToolResponse(result=result['result'])
