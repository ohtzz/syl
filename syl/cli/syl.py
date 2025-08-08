#!/usr/bin/env python3

import argparse

from loguru import logger as log

from .create import create, PG_FILE_TABLE_NAME, PG_FUNC_TABLE_NAME
from .start import start
from .stop import stop
from .restart import restart
from .status import print_syl_status, print_json_status
from .query import query_datastore
from .logs import show_logs
from .remove import remove
from .pull import pull_images

from syl.common.datastores import valid_name, DEFAULT_MODEL
from syl.common.docker import DockerManager
from syl.common.logger import setup_logger


docker = DockerManager()

setup_logger(level='INFO')


def main():
    parser = argparse.ArgumentParser(description='Manage Syl containers and datastores')
    
    # Create a parent parser with common arguments that will be inherited by all subparsers
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--docker-network-cidr', help='Use a custom Docker network CIDR when starting the Syl server', default='172.20.0.0/16')
    parent_parser.add_argument('--log-level', help='Log level', default='INFO')
    parent_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    # Create a project parent parser with project-specific arguments
    project_parent_parser = argparse.ArgumentParser(add_help=False)
    project_parent_parser.add_argument('--log-level', help='Log level', default='INFO')
    project_parent_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    project_parent_parser.add_argument('--description', help='Datastore project description; useful for LLMs when listing available projects/datastores', required=False)
    project_parent_parser.add_argument('--embed_model', '-e', help='Embedding model', default=DEFAULT_MODEL)
    project_parent_parser.add_argument('--no-daemon', action='store_true', dest='no_daemon', help='Don\'t run in daemon mode')  # not not not not daemon mode
    project_parent_parser.add_argument('--vector-size', help='Vector size; this MUST match your index/table for remote data stores', default=768)
    project_parent_parser.add_argument('--git-repo-url', help='Git repository URL to clone if this is a remote codebase', required=False)
    project_parent_parser.add_argument('--git-branch', help='Git repository branch to checkout after cloning', required=False)
    project_parent_parser.add_argument('--mount-local-code-dir', help='Container mount path for the code directory to index if this is a local codebase', required=False)
    project_parent_parser.add_argument('--no-server', action='store_true', help="Don't start the server if it's not running")
    project_parent_parser.add_argument('--num-workers', help='Number of workers to use for embeddings', default=16)
    project_parent_parser.add_argument('--skip-embed', action='store_true', help='Skip embedding/indexing if the datastore already contains data')
    project_parent_parser.add_argument('--no-register', action='store_true', help='Do not register this datastore with the Syl server')
    project_parent_parser.add_argument('--watch-dir', nargs='*', help='After indexing a datastore, create a directory file watcher to automatically update embeddings for created/modified files')
    project_parent_parser.add_argument('--exclude-ext', nargs='*', help='Exclude certain file extensions')
    
    subparsers = parser.add_subparsers(dest='action', help='Available actions')

    # Pull
    _ = subparsers.add_parser('pull', parents=[parent_parser], help='Pull latest versions of all syl images')

    # Status
    status_parser = subparsers.add_parser('status', aliases=['s'], parents=[parent_parser], help='Show status of Syl resources')
    status_parser.add_argument('--fmt', '-f', default='simple', help='Tabulate table format', choices=['simple', 'plain', 'grid', 'presto', 'tsv', 'fancy_grid', 'heavy_grid', 'mixed_grid', 'double_grid', 'rounded_grid'])
    status_parser.add_argument('--no-image-check', '-n', action='store_true', help='Do not check image tag statuses; passing this arg speeds up the status prints, but you won\'t know if you\'re using an old image')
    status_parser.add_argument('name', nargs='?', help='Optionally only show the JSON status of a specific container or all containers in a project')

    # Logs
    logs_parser = subparsers.add_parser('logs', parents=[parent_parser], help='Show/tail logs for a Syl container')
    logs_parser.add_argument('name', help='Container name or ID')
    logs_parser.add_argument('--max-results', '-m', help='Maximum number of results to show', default=10)
    logs_parser.add_argument('--follow', '-f', action='store_true', help='Tail logs')

    # Query
    query_parser = subparsers.add_parser('query', parents=[parent_parser], help='Perform a query against a datastore')
    query_parser.add_argument('datastore', help='Datastore name')
    query_parser.add_argument('--query', '-q', help='Query keywords', required=True)
    query_parser.add_argument('--distance-metric', help='Search method; not applicable to S3 (it uses the metric created on the bucket index)', default='cosine', choices=['cosine', 'euclidean'])
    query_parser.add_argument('--max-results', '-m', type=int, default=10, help='Maximum number of results to return', required=False)
    query_parser.add_argument('--file-filter', help='Filter results by file name', required=False)
    query_parser.add_argument('--function-filter', help='Filter results by function name', required=False)
    query_parser.add_argument('--min-complexity', '-c', type=int, help='Minimum complexity to return', required=False)
    query_parser.add_argument('--is-async', '-a', action='store_true', help='Filter results by async functions')
    query_parser.add_argument('--is-method', action='store_true', help='Filter results by methods')
    query_parser.add_argument('--files', action='store_true', help='Query the files table (default is functions)')

    # Start
    start_parser = subparsers.add_parser('start', parents=[parent_parser], help='Start a stopped Syl container')
    start_parser.add_argument('entity', help='Entity to start (shorthand first letter)', choices=['server', 'datastore', 'watcher', 's', 'p', 'd', 'w'])
    start_parser.add_argument('name', nargs='?', help='Container name or ID')

    # Stop
    stop_parser = subparsers.add_parser('stop', parents=[parent_parser], help='Stop a running Syl container')
    stop_parser.add_argument('entity', help='Entity to stop', choices=['server', 'datastore', 'watcher', 's', 'p', 'd', 'w'])
    stop_parser.add_argument('name', nargs='?', help='Container name or ID')

    # Restart
    restart_parser = subparsers.add_parser('restart', parents=[parent_parser], help='Restart a running Syl container')
    restart_parser.add_argument('entity', help='Entity to restart', choices=['server', 'datastore', 'watcher', 's', 'p', 'd', 'w'])
    restart_parser.add_argument('name', nargs='?', help='Container name or ID')

    # Remove
    remove_parser = subparsers.add_parser('remove', aliases=['rm'], parents=[parent_parser], help='Remove a Syl container')
    remove_parser.add_argument('entity', help='Entity to remove', choices=['server', 'index', 'datastore', 'watcher', 's', 'p', 'd', 'w'])
    remove_parser.add_argument('name', nargs='?', help='Container name or ID')

    # Create
    create_parser = subparsers.add_parser('create', parents=[parent_parser], help='Create the Syl server, a new datastore, or file watcher')
    create_parser.add_argument('--no-rm', action='store_true', help='Do not auto-remove this datastore\'s containers when stopped')
    create_parser.add_argument('--hf-cache-dir', help='HuggingFace model cache directory to mount', default='~/.cache/huggingface/hub')
    create_parser.add_argument('--no-mount-hf', action='store_true', help='Do not automatically mount the default HuggingFace model cache directory')
    create_subparsers = create_parser.add_subparsers(dest='entity', help='Entity to create')

    # Create server
    create_server_parser = create_subparsers.add_parser('server', aliases=['s'], parents=[parent_parser], help='Create the Syl server. This is done automatically when the first datastore is created.')
    create_server_parser.add_argument('--no-daemon', action='store_true', help='Don\'t run in daemon mode')

    # Create
    create_project_parser = create_subparsers.add_parser('datastore', aliases=['d'], parents=[parent_parser], help='Create a new datastore')
    
    # Datastores
    datastore_parser = create_project_parser.add_subparsers(dest='datastore', help='Type of datastore to use')

    # Create project S3 datastore arguments
    s3vector_datastore = datastore_parser.add_parser('s3vector', parents=[project_parent_parser], help='Use remote S3 Vector bucket for embeddings')
    s3vector_datastore.add_argument('name', help='project name')
    s3vector_datastore.add_argument('--s3-bucket', help='S3 bucket name')
    s3vector_datastore.add_argument('--s3-region', help='S3 region')
    s3vector_datastore.add_argument('--s3-index', help='S3 Vector bucket index')
    s3vector_datastore.add_argument('--aws-access-key', help='AWS access key')
    s3vector_datastore.add_argument('--aws-secret-key', help='AWS secret key')
    s3vector_datastore.add_argument('--aws-profile', help='AWS profile name to use if config dir is mounted')
    s3vector_datastore.add_argument('--aws-mount-config-dir', action='store_true', help='Mount aws config directory to the container (read-only)')

    # Create project Postgres datastore arguments
    pgvector_datastore = datastore_parser.add_parser('pgvector', parents=[project_parent_parser], help='Use local or remote pgvector for embeddings')
    pgvector_datastore.add_argument('name', help='pgvector datastore name')
    pgvector_datastore.add_argument('--local', action='store_true', help='Create a local pgvector container to store the embeddings')
    pgvector_datastore.add_argument('--pg-func-table', help='Custom function embeddings table name', required=False, default=PG_FUNC_TABLE_NAME)
    pgvector_datastore.add_argument('--pg-file-table', help='Custom file embeddings table name', required=False, default=PG_FILE_TABLE_NAME)
    pgvector_datastore.add_argument('--pg-host', help='Remote postgres host')
    pgvector_datastore.add_argument('--pg-port', type=int, help='Remote postgres port', default=5432)
    pgvector_datastore.add_argument('--pg-database', help='Remote postgres database name', default='postgres')
    pgvector_datastore.add_argument('--pg-user', help='Remote postgres user')
    pgvector_datastore.add_argument('--pg-password', help='Remote postgres password')

    # Create project ChromaDB datastore arguments
    chromadb_datastore = datastore_parser.add_parser('chromadb', parents=[project_parent_parser], help='Use local ChromaDB for embeddings')
    chromadb_datastore.add_argument('name', help='ChromaDB datastore name')
    chromadb_datastore.add_argument('--local', action='store_true', help='Create a local ChromaDB container to store the embeddings')
    chromadb_datastore.add_argument('--persistent-chroma', help='Run ChromaDB in persistent mode (default is in-memory)')
    chromadb_datastore.add_argument('--persistent-chroma-dir', help='ChromaDB persistent data directory; only applicable with --persistent-chroma')

    # Create file watcher
    create_watcher_parser = create_subparsers.add_parser('watcher', aliases=['w'], parents=[parent_parser], help='Create a new file watcher')
    create_watcher_parser.add_argument('datastore_name', help='Datastore name to watch')
    create_watcher_parser.add_argument('--directory', help='Directory to monitor for changes', required=True)
    create_watcher_parser.add_argument('--daemon', '-d', action='store_true', help='Run watcher container in daemon mode')
    create_watcher_parser.add_argument('--file-ext', nargs='+', help='File extensions to watch', default=['.py', '.go', '.js'])
    create_watcher_parser.add_argument('--num-workers', type=int, help='Number of worker threads', default=4)

    args = parser.parse_args()

    setup_logger(level=args.log_level, verbose=args.verbose)

    if hasattr(args, 'name') \
            and args.name is not None \
            and not valid_name(args.name):

        log.error(('invalid datastore name; must be alphanumeric and contain only letters, '
                   'numbers, dashes and underscores'), extra={
            'datastore_name': args.name,
        })
        raise ValueError(('invalid datastore name; must be alphanumeric and contain only '
                         'letters, numbers, dashes and underscores'))

    if args.action in ('status', 's'):
        if args.name is not None:
            return print_json_status(args.name)
        else:
            return print_syl_status(
                docker=docker,
                tabulate_table_fmt=args.fmt,
                include_stale_images=not args.no_image_check
            )

    elif args.action == 'logs':
        return show_logs(args, docker)

    elif args.action == 'query':
        return query_datastore(args, docker)

    elif args.action == 'create':
       return create(args, docker)

    elif args.action == 'start':
        return start(args, docker)

    elif args.action == 'stop':
        return stop(args, docker)

    elif args.action == 'restart':
        return restart(args, docker)

    elif args.action in ('remove', 'rm'):
        return remove(args, docker)

    elif args.action == 'pull':
        return pull_images(docker)

    else:
        log.error('unknown action', extra={
            'action': args.action,
        })
        raise ValueError(f'unknown action: {args.action}')


if __name__ == '__main__':
    main()
