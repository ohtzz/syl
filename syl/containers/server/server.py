import asyncio
import uvicorn

from loguru import logger as log

from syl.common import setup_logger

from .control import control_server
from .mcp_http_stream import mcp_server
from .openai_http import openai_server


setup_logger(level='INFO', verbose=True)


async def start_servers():
    """Start all servers concurrently"""

    # Control server
    config_control = uvicorn.Config(control_server, host='0.0.0.0', port=9000)
    server_control = uvicorn.Server(config_control)

    async def run_control():
        await server_control.serve()

    # FastAPI server
    config_openai = uvicorn.Config(openai_server, host='0.0.0.0', port=8000, log_level='info')
    server_openai = uvicorn.Server(config_openai)

    # MCP server
    async def run_mcp():
        await mcp_server.run_streamable_http_async(host='0.0.0.0', port=8001, path='/mcp')

    log.info('Starting control server on port 9000...')
    log.info('Starting OpenAI-compatible server on port 8000...')
    log.info('Starting MCP server on port 8001...')

    # Run all servers concurrently
    await asyncio.gather(server_openai.serve(), run_mcp(), run_control())


def main():
    log.info('Control server: http://0.0.0.0:9000')
    log.info('FastAPI OpenAI server: http://0.0.0.0:8000')
    log.info('MCP server: http://0.0.0.0:8001/mcp')
    asyncio.run(start_servers())


if __name__ == '__main__':
    main()
