import argparse
import requests

from loguru import logger as log

from syl.common import DockerManager
from syl.common.docker import SYL_SERVER_NAME, datastore_name_from_container_name


def query_datastore(args: argparse.Namespace, docker: DockerManager):
    log.debug(
        'Performing query..',
        extra={
            'datastore_name': args.datastore,
        },
    )

    datastore_name = datastore_name_from_container_name(args.datastore)
    server_ip = docker.get_container_ip(SYL_SERVER_NAME)

    if not server_ip:
        raise RuntimeError(f'error IP address not found for {SYL_SERVER_NAME}')

    if args.files:
        route = 'search_file_code_semantic'
    else:
        route = 'search_function_code_semantic'

    resp = requests.post(
        url=f'http://{server_ip}:8000/tools/{route}',
        json={
            'datastore_name': datastore_name,
            'query': args.query,
            'max_results': args.max_results,
            'file_filter': args.file_filter,
            'function_filter': args.function_filter,
            'complex_filter': args.min_complexity,
            'is_async_filter': args.is_async,
            'is_method_filter': args.is_method,
        },
    )

    resp.raise_for_status()
    print(resp.json())
