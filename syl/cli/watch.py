import argparse
import requests
import os

from loguru import logger as log

from syl.common import DockerManager
from syl.common.docker import SYL_WATCHER_IMAGE_NAME, SYL_SERVER_NAME, SYL_WATCHER_PREFIX


def create_watcher(args: argparse.Namespace, docker: DockerManager):
    # Ensure Docker network is up
    if not docker.network_is_up():
        log.warning('Docker network is not up')
        docker.create_network(driver='bridge')

    # Check if syl server is running
    server_ip = docker.get_container_ip(SYL_SERVER_NAME)
    if not server_ip:
        log.error("Syl server is not running. Please start it first with 'syl run server'")
        return

    # Get datastore configuration from syl server
    log.info(f'Fetching datastore configuration for: {args.name}')
    try:
        resp = requests.get(f'http://{server_ip}:9000/datastores')
        resp.raise_for_status()
        datastores = resp.json()

        datastore_config = datastores.get(args.name)

        if not datastore_config:
            log.error(f"Datastore '{args.name}' not found. Available datastores: {[p.get('name') for p in datastores]}")
            return

    except Exception as e:
        log.error(f'Failed to fetch datastore configuration: {e}')
        return

    log.info(f'Found datastore configuration: {datastore_config.get("datastore_type", "unknown")} datastore')

    watcher_args = [args.name, '--dir', '/watch', '--num-workers', str(args.num_workers)]

    if args.file_ext:
        watcher_args.extend(['--file-ext'] + args.file_ext)

    data_provider = datastore_config.get('data_provider')

    if data_provider == 's3':
        watcher_args.append('--s3-datastore')
        # Add S3 configuration from datastore
        for key, flag in [
            ('s3_bucket', '--s3-bucket'),
            ('s3_region', '--s3-region'),
            ('s3_index', '--s3-index'),
            ('access_key', '--s3-access-key'),
            ('secret_key', '--s3-secret-key'),
            ('aws_profile', '--aws-profile'),
        ]:
            if key in datastore_config:
                watcher_args.extend([flag, str(datastore_config[key])])

    elif data_provider == 'pgvector':
        watcher_args.append('--pgvector-datastore')
        for key, flag in [
            ('pg_host', '--pg-host'),
            ('pg_port', '--pg-port'),
            ('pg_database', '--pg-database'),
            ('pg_func_table', '--pg-func-table'),
            ('pg_file_table', '--pg-file-table'),
            ('pg_user', '--pg-user'),
            ('pg_password', '--pg-password'),
        ]:
            if key in datastore_config:
                watcher_args.extend([flag, str(datastore_config[key])])

    elif data_provider == 'chromadb':
        watcher_args.append('--chromadb-datastore')

        if datastore_config.get('persistent'):
            watcher_args.append('--chromadb-persist')
        for key, flag in [('path', '--chromadb-path'), ('collection', '--chromadb-collection')]:
            if key in datastore_config:
                watcher_args.extend([flag, str(datastore_config[key])])
    else:
        log.error(f'Unsupported datastore type: {datastore_config}')
        return

    if 'embed_model' in datastore_config:
        watcher_args.extend(['--embed-model', datastore_config['embed_model']])
    if 'embed_model_vector_size' in datastore_config:
        watcher_args.extend(['--vector-size', str(datastore_config['embed_model_vector_size'])])

    if not os.path.exists(args.directory):
        log.error(f'Directory does not exist: {args.directory}')
        return
    if not os.path.isdir(args.directory):
        log.error(f'Path is not a directory: {args.directory}')
        return

    container_name = f'{SYL_WATCHER_PREFIX}{args.name}'

    if docker.container_is_running(container_name):
        log.error(f'Watcher container {container_name} is already running.')
        return

    log.info(f'Starting file watcher for datastore: {args.name}')
    log.info(f'Watching directory: {args.directory}')

    volumes = {args.directory: '/watch'}

    docker.create_container(
        image=SYL_WATCHER_IMAGE_NAME,
        name=container_name,
        daemon_mode=not args.no_daemon,
        volumes=volumes,
        env_vars=None,
        extra_args=None,
        script_args=watcher_args,
    )

    log.info(
        'Started file watcher',
        extra={
            'datastore': args.name,
            'directory': args.directory,
            'container': container_name,
        },
    )
