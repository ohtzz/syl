import argparse
import time

from loguru import logger as log

from .index import create_args, create_volumes
from .server import create_control_server, control_server_running, get_registered_datastore
from .watch import create_watcher

from syl.common import DockerManager, Datastore, DataProvider
from syl.common.docker import (
    PGVECTOR_IMAGE,
    SYL_SERVER_NAME,
    SYL_INDEX_IMAGE_NAME,
    SYL_INDEX_PREFIX,
    SYL_PGVECTOR_PREFIX,
    CHROMA_DB_IMAGE, SYL_CHROMADB_PREFIX,
)
from syl.common.datastores import Status
from syl.common.server_api import register_datastore

# Duplicate of datastore constants to avoid slow import
# that loads sentence_transformers (2-3 seconds)
PG_FUNC_TABLE_NAME = 'func_embeddings'
PG_FILE_TABLE_NAME = 'file_embeddings'


def create(args: argparse.Namespace, docker: DockerManager):
    if not docker.network_is_up():
        log.warning('Docker network is not up')
        docker.create_network(driver='bridge', subnet=args.docker_network_cidr)

    if args.entity in ('server', 's'):
        return create_control_server(args, docker)
    elif args.entity in ('datastore', 'd'):
        return create_new_datastore(args, docker)
    elif args.entity in ('watcher', 'w'):
        return create_watcher(args, docker)
    else:
        raise ValueError(f'Unknown entity: {args.entity}')


def create_new_datastore(args: argparse.Namespace, docker: DockerManager):
    rm_container_on_stop = not args.no_rm

    # Start the Syl server if it's not running and user didn't manually exclude
    if not args.no_server and not control_server_running(docker):
        log.info(
            'Starting Syl server in daemon mode as one is not already running..',
            extra={
                'daemon_mode': True,
            },
        )
        create_control_server(args, docker, True)
        time.sleep(3)

    # Names are unique..
    registered_datastore = get_registered_datastore(args.name)
    if registered_datastore is not None:
        log.error(
            f'Container {args.name} is already registered. Remove it first via "syl remove {args.name}" or choose a different name.'
        )
        raise RuntimeError(f'Container {args.name} is already running.')

    if not args.mount_local_code_dir and not args.git_repo_url:
        log.error('You must provide either --mount-local-code-dir code source or --git-repo-url')
        raise RuntimeError()

    # S3 Vector stores limit to 2KB metadata and 10 filterable
    # KV pairs.
    # This means we won't have the code for functions/files and,
    # in some cases, we may have to trim metadata to fit.
    #
    # I probably won't finish support for this as it would require
    # storing larger data separate from the S3 Vector bucket (a
    # regular S3 bucket? A local Postgres container? ...)
    if args.datastore == DataProvider.S3_VECTOR:
        log.warning(
            (
                'You are creating a remote S3 Vector bucket datastore. '
                'Support for this is half-baked/experimental/whatever you want to call it. '
                "You won't have access to full code, all metadata, etc. "
                'See the README.'
            )
        )

    # Start local pgvector data container
    elif args.datastore == DataProvider.PGVECTOR:
        if args.local:
            # Kinda janky, but need args later on for registration
            args.pg_host = f'{SYL_PGVECTOR_PREFIX}{args.name}'
            args.pg_user = 'postgres'
            args.pg_password = args.name
            args.pg_database = 'postgres'
            args.pg_func_table = PG_FUNC_TABLE_NAME
            args.pg_file_table = PG_FILE_TABLE_NAME
            args.pg_port = 5432

            docker.create_container(
                image=PGVECTOR_IMAGE,
                name=args.pg_host,
                daemon_mode=True,
                ports=None,
                volumes=None,
                env_vars={
                    'POSTGRES_USER': args.pg_user,
                    'POSTGRES_PASSWORD': args.pg_password,
                },
                script_args=None,
                auto_remove=rm_container_on_stop,
            )
        else:
            # Require all remote PG args..
            if not all(
                [
                    args.pg_host,
                    args.pg_port,
                    args.pg_database,
                    args.pg_user,
                    args.pg_password,
                    args.pg_func_table,
                    args.pg_file_table,
                ]
            ):
                raise RuntimeError('error all Postgres args are required for remote datastore')

    elif args.datastore == DataProvider.CHROMA_DB:
        if args.local:
            docker.create_container(
                image=CHROMA_DB_IMAGE,
                name=f'{SYL_CHROMADB_PREFIX}{args.name}',
                daemon_mode=True,
                volumes={args.persistent_chroma_dir: './chroma-data:/data'} if args.persistent_chroma_dir else None,
                env_vars=None,
                script_args=None,
                auto_remove=rm_container_on_stop,
            )
        else:
            log.error('Only local ChromaDB datastores are currently supported.')
            raise RuntimeError('Only local ChromaDB datastores are currently supported.')

    script_args = create_args(args)
    volumes = create_volumes(args)

    docker.create_container(
        image=SYL_INDEX_IMAGE_NAME,
        name=f'{SYL_INDEX_PREFIX}{args.name}',
        daemon_mode=not args.no_daemon,
        volumes=volumes,
        script_args=script_args,
        env_vars=None,
        extra_args=None,
    )

    # TODO
    # If --no-daemon is passed then the datastore isn't registered
    # as we attach to the container stdout.
    #
    # This means the datastore shows as 'not found' instead of 'indexing'
    # until indexing completes, and then it gets stuck as 'indexing'
    # as _this_ registration runs _after_ the index container status update
    # to the control server.
    #
    # If we move registration prior to the container creation then
    # we'll need to handle clean up if the creation fails.

    # Register the datastore as indexing.
    #
    # When it completes, it updates its own status and
    # starts any file watchers that were requested.
    register_datastore(
        datastore=Datastore(
            name=args.name,
            description=args.description,
            embed_model=args.embed_model,
            embed_model_vector_size=args.vector_size,
            data_provider=DataProvider(args.datastore),
            # S3 Vector bucket -- remote
            s3_bucket=args.s3_bucket if args.datastore == DataProvider.S3_VECTOR else None,
            s3_index=args.s3_index if args.datastore == DataProvider.S3_VECTOR else None,
            s3_region=args.s3_region if args.datastore == DataProvider.S3_VECTOR else None,
            # pgvector -- local or remote
            pg_host=args.pg_host if args.datastore == DataProvider.PGVECTOR else None,
            pg_port=int(args.pg_port) if args.datastore == DataProvider.PGVECTOR and args.pg_port else None,
            pg_user=args.pg_user if args.datastore == DataProvider.PGVECTOR else None,
            pg_password=args.pg_password if args.datastore == DataProvider.PGVECTOR else None,
            pg_database=args.pg_database if args.datastore == DataProvider.PGVECTOR else None,
            pg_func_table=args.pg_func_table if args.datastore == DataProvider.PGVECTOR else None,
            pg_file_table=args.pg_file_table if args.datastore == DataProvider.PGVECTOR else None,
            # ChromaDB -- local
            # TODO
        ),
        status=Status.indexing,
        watch_dirs=args.watch_dir or None,
        syl_host=docker.get_container_ip(SYL_SERVER_NAME),
    )
