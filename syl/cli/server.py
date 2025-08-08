import argparse
import requests
import os

from typing import Dict, Optional
from loguru import logger as log

from syl.common import DockerManager, Datastore
from syl.common.docker import SYL_SERVER_NAME, SYL_SERVER_IMAGE_NAME
from syl.common.datastores import DatastoreRegistration

SYL_SERVER_URL = 'http://localhost:9000'

SYL_SERVER_CONTROL_PORT = 9000
SYL_SERVER_TOOL_HTTP_PORT = 8000
SYL_SERVER_TOOL_MCP_PORT = 8001


def create_control_server(args: argparse.Namespace, docker: DockerManager, override_daemon_mode: Optional[bool] = None):
    log.info('Starting Syl server..', extra={'daemon_mode': not args.no_daemon})

    volumes = {'/var/run/docker.sock': '/var/run/docker.sock'}

    if not args.no_mount_hf:
        volumes[os.path.expanduser(args.hf_cache_dir)] = '/root/.cache/huggingface/hub'

    docker.create_container(
        image=SYL_SERVER_IMAGE_NAME,
        name=SYL_SERVER_NAME,
        daemon_mode=override_daemon_mode if override_daemon_mode is not None else not args.no_daemon,
        volumes=volumes,
        auto_remove=not args.no_rm,
        ports={
            SYL_SERVER_CONTROL_PORT: SYL_SERVER_CONTROL_PORT,  # Control server
            SYL_SERVER_TOOL_HTTP_PORT: SYL_SERVER_TOOL_HTTP_PORT,  # HTTP OpenAI-format
            SYL_SERVER_TOOL_MCP_PORT: SYL_SERVER_TOOL_MCP_PORT,  # MCP
        },
    )


def control_server_running(docker: DockerManager) -> bool:
    """Returns true if the syl-server container is running"""
    return docker.container_is_running(SYL_SERVER_NAME)


def get_registered_datastore(datastore_name: str) -> Optional[DatastoreRegistration]:
    """Returns the registered datastore by name or None"""
    datastores = get_registered_datastores()
    return datastores.get(datastore_name, None)


def get_registered_datastores() -> Dict[str, DatastoreRegistration]:
    """Fetch registered datastores from the Syl server"""
    try:
        resp = requests.get(f'{SYL_SERVER_URL}/datastores', timeout=5)
        if resp.status_code == 200:
            return resp.json()
        else:
            log.warning(
                'error received non-200 status code from syl server',
                extra={
                    'status_code': resp.status_code,
                    'text': resp.text,
                },
            )
            return {}
    except Exception as e:
        log.debug(
            'error connecting to Syl server to get registered datastores',
            extra={
                'error': str(e),
            },
        )
        return {}
