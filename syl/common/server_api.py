import requests

from loguru import logger as log
from typing import Optional, List

from syl.common import Datastore
from syl.common.docker import SYL_SERVER_NAME
from syl.common.datastores import Status


def register_datastore(
    datastore: Datastore, status: Status, syl_host: str = SYL_SERVER_NAME, watch_dirs: Optional[List[str]] = None
):
    log.debug(
        'Registering datastore',
        extra={
            'syl_host': syl_host,
            'status': status.name,
            'datastore': datastore.name,
            'watch_dirs': watch_dirs,
        },
    )

    resp = requests.post(
        url=f'http://{syl_host}:9000/datastores',
        json={
            'datastore': datastore.model_dump(mode='json'),
            'status': status.value,
            'file_watch_dirs': watch_dirs,
        },
    )

    if resp.status_code > 299:
        log.error(
            'error received non-2XX response',
            extra={
                'status_code': resp.status_code,
                'response_text': resp.text,
            },
        )
        resp.raise_for_status()

    log.debug(
        'Successfully registered datastore',
        extra={
            'name': datastore.name,
            'status': status,
            'file_watch_dirs': watch_dirs,
        },
    )


def update_datastore_status(name: str, status: Status, syl_host: str = SYL_SERVER_NAME):
    resp = requests.put(url=f'http://{syl_host}:9000/datastores/{name}/status/{status.value}')

    if resp.status_code > 299:
        log.error(
            'error received non-2XX response',
            extra={
                'status_code': resp.status_code,
                'response_text': resp.text,
            },
        )
        resp.raise_for_status()

    log.debug(
        'Successfully updated datastore status',
        extra={
            'name': name,
            'status': status,
        },
    )


def unregister_datastore(name: str, syl_host: str = SYL_SERVER_NAME):
    resp = requests.delete(
        url=f'http://{syl_host}:9000/datastores/{name}',
    )

    if resp.status_code > 299:
        log.error(
            'error received non-2XX response',
            extra={
                'status_code': resp.status_code,
                'response_text': resp.text,
            },
        )
        resp.raise_for_status()

    log.info(
        'Successfully unregistered datastore',
        extra={
            'name': name,
        },
    )
