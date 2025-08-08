import argparse
import time

from loguru import logger as log

from syl.common import DockerManager
from syl.common.docker import SYL_SERVER_NAME, SYL_INDEX_PREFIX, SYL_WATCHER_PREFIX


def restart(args: argparse.Namespace, docker: DockerManager):
    if args.entity in ('datastore', 'watcher') and not args.name:  # TODO no index containers here; check cli args
        print('Error: Name must be specified')
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
                log.error(
                    'No containers found matching name',
                    extra={
                        'name': args.name,
                    },
                )
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

    if container.status != 'running':
        log.warning(
            'Container is not running; doing nothing..',
            extra={
                'container': container.name,
                'status': container.status,
            },
        )
        return

    auto_remove = container.attrs.get('HostConfig', {}).get('AutoRemove', False)
    if auto_remove:
        log.warning(
            "Container is set to auto-remove. Restarting it would delete it. Use 'remove' instead if you want to proceed.",
            extra={
                'container': container.name,
            },
        )
        return

    try:
        container.stop()
        time.sleep(2)
        container.start()
    except Exception as e:
        log.error('error restarting container', extra={'container': container.name, 'error': str(e)})
        raise e

    log.info('Successfully restarted container', extra={'container': container.name})
