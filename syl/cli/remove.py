import argparse

from loguru import logger as log

from syl.common import DockerManager
from syl.common.docker import SYL_SERVER_NAME, SYL_INDEX_PREFIX, SYL_WATCHER_PREFIX, datastore_name_from_container_name
from syl.common.server_api import unregister_datastore


def remove(args: argparse.Namespace, docker: DockerManager):
    if args.entity in ('index', 'datastore', 'watcher') and not args.name:
        log.error('error name must be specified')
        return

    if args.entity in ('server', 's'):
        container = docker.get_container_by_name(SYL_SERVER_NAME)

    elif args.entity in ('index', 'i'):
        if not args.name.startswith(SYL_INDEX_PREFIX):
            args.name = SYL_INDEX_PREFIX + args.name
        container = docker.get_container_by_name(args.name)

    elif args.entity in ('datastore', 'd'):
        if not args.name.startswith('syl-'):
            containers = docker.get_datastore_containers_by_name_suffix(args.name)
            if not containers:
                log.warning(
                    'No containers found matching name; attempting deregistration with provided name..',
                    extra={
                        'name': args.name,
                    },
                )

                # Attempt deregistration by
                server_ip = docker.get_container_ip(SYL_SERVER_NAME)
                if not server_ip:
                    raise RuntimeError(f'error IP address not found for {SYL_SERVER_NAME}')
                unregister_datastore(args.name, docker.get_container_ip(SYL_SERVER_NAME))

                return
            elif len(containers) == 1:
                container = containers[0]
            else:
                log.error(
                    'Multiple datastore containers found matching name. You must pass the full name in this case.',
                    extra={'name': args.name, 'container_names': [c.name for c in containers]},
                )
                return
        else:
            container = docker.get_container_by_name(args.name)

    elif args.entity in ('watcher', 'w'):
        if not args.name.startswith(SYL_WATCHER_PREFIX):
            args.name = SYL_WATCHER_PREFIX + args.name
        container = docker.get_container_by_name(args.name)

    else:
        raise ValueError(f'Unknown entity: {args.entity}')

    if not container:
        log.error(
            f'error {args.entity} container not found',
            extra={
                'name': args.name,
            },
        )
        return

    if container.status not in ('running', 'stopped'):
        log.warning(
            'Container is not running or stopped; doing nothing..',
            extra={
                'container': container.name,
                'status': container.status,
            },
        )
        return

    try:
        container.remove(force=True)
    except Exception as e:
        log.error('error removing container', extra={'container': container.name, 'error': str(e)})
        raise e

    log.info('Successfully removed container', extra={'container': container.name})

    if container.name == SYL_SERVER_NAME:
        return

    # Deregister
    server_ip = docker.get_container_ip(SYL_SERVER_NAME)

    if not server_ip:
        raise RuntimeError(f'error IP address not found for {SYL_SERVER_NAME}')

    name = datastore_name_from_container_name(args.name)
    unregister_datastore(name, docker.get_container_ip(SYL_SERVER_NAME))

    # Stop all file watcher containers TODO

    log.info('Successfully deregistered datastore', extra={'datastore': name})
