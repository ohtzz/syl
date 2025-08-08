import tabulate
import json

from loguru import logger as log

from .server import get_registered_datastores
from syl.common import Colors, DockerManager
from syl.common.datastores import get_registered_status
from syl.common.docker import (
    is_index_container,
    is_datastore_container,
    datastore_name_from_container_name,
    is_watcher_container,
    SYL_SERVER_NAME,
    SYL_INDEX_PREFIX,
    SYL_PGVECTOR_PREFIX,
    SYL_CHROMADB_PREFIX,
    SYL_WATCHER_PREFIX,
    SYL_NETWORK_NAME,
)


def print_json_status(name: str) -> None:
    all_datastores = get_registered_datastores()

    datastores = []

    for p_name, pr in all_datastores.items():
        if p_name.endswith(name):
            datastores.append(pr)

    print(json.dumps(datastores, indent=2))


def _is_syl_image(container_image: str) -> bool:
    """Check if an image is a syl image (not external dependencies like pgvector/chromadb)"""
    if container_image.startswith('ohtz/syl'):
        return True

    if container_image.startswith('sha256:') or len(container_image) == 12:
        return True  # Checked in _is_syl_image_outdated

    non_syl_prefixes = ['pgvector/', 'chromadb/']
    for prefix in non_syl_prefixes:
        if container_image.startswith(prefix):
            return False

    return False


def _is_syl_image_outdated(docker: DockerManager, container_image: str) -> bool:
    """Check if a syl image is outdated by checking if it has the -latest tag"""
    if not _is_syl_image(container_image):
        return False

    # If it's a syl image and has the -latest tag, it's not outdated
    if container_image.startswith('ohtz/syl') and container_image.endswith('-latest'):
        return False

    # If it's a syl image but doesn't have -latest tag (including hashes), it's outdated
    return True


def print_syl_status(
    docker: DockerManager, tabulate_table_fmt: str = 'simple', include_stale_images: bool = False
) -> None:
    log.debug('Getting status of Syl resources..')

    status_items = []

    # Network status
    if docker.network_is_up():
        try:
            # Get CIDR from network IPAM config in case a custom CIDR was provided
            network_cidr = docker.get_network_cidr()
            status_items.append(
                {
                    'name': docker.network_name,
                    'datastore_name': 'syl',
                    'container_id': 'N/A',
                    'registered': 'N/A',
                    'status': 'available',
                    'container_state': 'N/A',
                    'image': 'docker-network',
                    'ip_address': network_cidr,
                }
            )
        except Exception as e:
            log.error('error could not get network cidr', extra={'error': str(e)})
    else:
        status_items.append(
            {
                'name': docker.network_name,
                'datastore_name': 'syl',
                'container_id': 'N/A',
                'registered': 'N/A',
                'status': 'down',
                'container_state': 'N/A',
                'image': 'docker-network',
                'ip_address': 'N/A',
            }
        )

    registered_datastores = get_registered_datastores()

    log.debug(
        'Retrieved registered datastores',
        extra={
            'registered_datastores': registered_datastores,
        },
    )

    # Track which datastores have containers
    datastores_with_containers = set()

    containers = docker.list_containers()
    for container in containers:
        try:
            ip = docker.get_container_ip_from_container(container)
            if not ip:
                ip = 'N/A'
        except Exception:
            ip = 'N/A'

        if container.name:
            # Index and datastore containers
            if is_index_container(container) or is_datastore_container(container):
                datastore_name = datastore_name_from_container_name(container.name)
                datastore = registered_datastores.get(datastore_name, None)
                datastores_with_containers.add(datastore_name)

                registered, status, _ = get_registered_status(container.status, datastore)

            # Watcher containers
            elif is_watcher_container(container):
                datastore_name = datastore_name_from_container_name(container.name)
                datastores_with_containers.add(datastore_name)
                registered = True
                status = 'available'

            # Syl server
            elif container.name == SYL_SERVER_NAME:
                datastore_name = 'syl'
                registered = 'N/A'  # Server itself doesn't need to be registered
                status = 'available'

            # Other
            else:
                datastore_name = container.name
                registered = 'unknown'
                status = 'unknown'

        else:
            datastore_name = container.id
            registered = 'unknown'
            status = 'unknown'

        status_items.append(
            {
                'name': container.name,
                'datastore_name': datastore_name,
                'container_id': container.short_id,
                'registered': str(registered) if registered != 'N/A' else 'N/A',
                'status': status,
                'container_state': container.status,
                'image': container.image.tags[0] if container.image.tags else container.image.short_id,
                'image_id': container.image.id,
                'ip_address': ip,
            }
        )

    # Add registered datastores that don't have local containers
    for datastore_name, datastore_registration in registered_datastores.items():
        if datastore_name not in datastores_with_containers:
            status_items.append(
                {
                    'name': f'(remote) {datastore_name}',
                    'datastore_name': datastore_name,
                    'container_id': 'N/A',
                    'registered': 'True',
                    'status': datastore_registration['status'],
                    'container_state': 'N/A',
                    'image': 'N/A',
                    'image_id': 'N/A',
                    'ip_address': 'N/A',
                }
            )

    if status_items:
        # Sort status items to ensure consistent ordering
        def sort_key(item):
            name = item['name']
            if name == docker.network_name:  # syl-network
                return (0, name)
            elif name == SYL_SERVER_NAME:
                return (1, name)
            elif name.startswith(SYL_WATCHER_PREFIX):
                return (2, name)
            elif name.startswith('(remote)'):
                return (4, name)  # Remote datastores at the end
            else:
                return (3, name)

        status_items.sort(key=sort_key)

        _print_containers_tabulate(
            status_items, docker, table_fmt=tabulate_table_fmt, include_stale_images=include_stale_images
        )

    else:
        log.warning('No containers or network found')


def _print_containers_tabulate(
    containers, docker: DockerManager, table_fmt: str = 'simple', include_stale_images: bool = False
) -> None:
    if not containers:
        return

    table_data = []
    headers = [
        'CONTAINER NAME',
        'DATASTORE',
        'CONTAINER ID',
        'REGISTERED',
        'STATUS',
        'CONTAINER',
        'IMAGE',
        'IP ADDRESS',
    ]

    for container in containers:
        name = container['name']
        if name == SYL_NETWORK_NAME:
            colored_name = Colors.bright_magenta(name)
        elif name == SYL_SERVER_NAME:
            colored_name = Colors.bright_cyan(name)
        elif name.startswith(SYL_INDEX_PREFIX):
            colored_name = Colors.bright_yellow(name)
        elif name.startswith(SYL_PGVECTOR_PREFIX):
            colored_name = Colors.orange(name)
        elif name.startswith(SYL_CHROMADB_PREFIX):
            colored_name = Colors.blue(name)
        elif name.startswith(SYL_WATCHER_PREFIX):
            colored_name = Colors.bright_green(name)
        elif name.startswith('(remote)'):
            colored_name = Colors.bright_white(name)
        else:
            colored_name = name

        registered = container['registered']
        if registered == 'True':
            colored_registered = Colors.bright_green(registered)
        elif registered == 'False':
            colored_registered = Colors.bright_red(registered)
        else:
            colored_registered = registered

        status = container['status']
        if status in ('available', 'registered'):
            status_symbol = '✓'
            colored_status = Colors.bright_green(f'{status_symbol} available')
        elif status == 'indexing':
            status_symbol = '~'
            colored_status = Colors.bright_yellow(f'{status_symbol} {status}')
        else:
            status_symbol = '✗'
            colored_status = Colors.bright_red(f'{status_symbol} {status}')

        container_state = container['container_state']
        if container_state == 'running':
            container_state_symbol = '✓'
            colored_container_status = Colors.bright_green(f'{container_state_symbol} {container_state}')
        elif container_state == 'stopped':
            container_state_symbol = '~'
            colored_container_status = Colors.bright_yellow(f'{container_state_symbol} {container_state}')
        elif container_state == 'N/A':
            colored_container_status = container_state
        else:
            container_state_symbol = '✗'
            colored_container_status = Colors.bright_red(f'{container_state_symbol} {container_state}')

        # Check if image is outdated and color yellow
        if include_stale_images:
            image_text = container['image'][:22] + '...' if len(container['image']) > 25 else container['image']
            # Use image_id for hash-based checking if container is using a hash, otherwise use image tag
            image_to_check = (
                container.get('image_id', container['image'])
                if (container['image'].startswith('sha256:') or len(container['image']) == 12)
                else container['image']
            )
            if _is_syl_image_outdated(docker, image_to_check):
                colored_image = Colors.yellow(image_text)
            elif container['name'].startswith('(remote)') and container['image'] == 'N/A':
                colored_image = Colors.bright_white('N/A')
            else:
                colored_image = image_text

            row = [
                colored_name,
                container['datastore_name'],
                container['container_id'][:12],  # Truncate container ID
                colored_registered,
                colored_status,
                colored_container_status,
                colored_image,
                container['ip_address'],
            ]

        else:
            # Handle image display for remote datastores
            if container['name'].startswith('(remote)') and container['image'] == 'N/A':
                image_display = Colors.bright_white('N/A')
            else:
                image_display = container['image'][:22] + '...' if len(container['image']) > 25 else container['image']

            row = [
                colored_name,
                container['datastore_name'],
                container['container_id'][:12],  # Truncate container ID
                colored_registered,
                colored_status,
                colored_container_status,
                image_display,
                container['ip_address'],
            ]

        table_data.append(row)

    print()
    print(tabulate.tabulate(table_data, headers=headers, tablefmt=table_fmt))
    print()
