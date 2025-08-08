import argparse

from loguru import logger as log

from syl.common.docker import DockerManager


def show_logs(args: argparse.Namespace, docker: DockerManager):
    log.debug('Getting logs of Syl resources..')

    if not args.name.startswith('syl-'):
        containers = docker.get_containers_by_name_suffix(args.name)
        if not containers:
            log.error('No containers found for name', extra={'name': args.name})
            return
        elif len(containers) == 1:
            container_name = containers[0].name
        else:
            log.error(
                'More than one container found for name. You must provide the full container name in this case.',
                extra={'name': args.name, 'container_names': [c.name for c in containers]},
            )
            return
    else:
        container_name = args.name

    if not container_name:
        log.error(f'Container not found: {args.name}')
        return

    max_results = int(args.max_results) if args.max_results else 10

    log.debug(f'Getting logs for container: {container_name}')

    container = docker.get_container_by_name(container_name)
    if not container:
        log.error(f'Container "{container_name}" not found')
        return

    if args.follow:
        log.info(f'Tailing logs for {container_name}. Press Ctrl+C to stop.')
        try:
            docker.get_container_logs(container_name, max_results=max_results, tail=True, follow=True)
        except KeyboardInterrupt:
            log.info('Log tailing stopped by user')
    else:
        logs = docker.get_container_logs(container_name, max_results=max_results, tail=False, follow=False)
        if logs:
            print(logs)
        else:
            log.warning(f'No logs found for container {container_name}')
