import argparse
import os
import time
import signal
import threading

from pathlib import Path
from typing import Set, Dict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from loguru import logger as log

from syl.common.logger import setup_logger
from syl.common.datastores import DEFAULT_MODEL
from syl.datastores.s3_vector import S3VectorLoader
from syl.datastores.pgvector import PostgreSQLLoader
from syl.datastores.chromadb import ChromaDBLoader
from syl.datastores.constants import SUPPORTED_FILETYPES
from syl.datastores.utils import should_process_file


setup_logger(level='INFO', verbose=True)


SUPPORTED_EXTENSIONS = SUPPORTED_FILETYPES


class CodeFileHandler(FileSystemEventHandler):
    """File system event handler for code files"""

    def __init__(self, data_loader, watch_directory: str, file_extensions: Set[str] = None):
        super().__init__()
        self.data_loader = data_loader
        self.watch_directory = Path(watch_directory).resolve()
        self.file_extensions = file_extensions or SUPPORTED_EXTENSIONS
        self.processing_queue: Set[str] = set()
        self.last_processed: Dict[str, float] = {}
        self.debounce_delay = 5.0

        log.info(f'Watching directory: {self.watch_directory}')
        log.info(f'Supported extensions: {self.file_extensions}')

    def should_process_file_watcher(self, file_path: str) -> bool:
        path = Path(file_path)

        if path.suffix.lower() not in self.file_extensions:
            return False

        return should_process_file(path)

    def on_modified(self, event):
        if not event.is_directory:
            self.handle_file_change(event.src_path, 'modified')

    def on_created(self, event):
        if not event.is_directory:
            self.handle_file_change(event.src_path, 'created')

    def on_moved(self, event):
        if not event.is_directory:
            self.handle_file_change(event.dest_path, 'moved')

    def handle_file_change(self, file_path: str, event_type: str):
        """Handle file change with debounce"""
        if not self.should_process_file_watcher(file_path):
            return

        abs_path = str(Path(file_path).resolve())
        current_time = time.time()

        if abs_path in self.last_processed:
            if current_time - self.last_processed[abs_path] < self.debounce_delay:
                log.debug(f'Skipping {abs_path} - recently processed')
                return

        log.info(f'File {event_type}: {abs_path}')
        self.processing_queue.add(abs_path)
        self.last_processed[abs_path] = current_time

        # Process the file after a short delay to handle rapid successive changes
        timer = threading.Timer(self.debounce_delay, self.process_file, args=[abs_path])
        timer.start()

    def process_file(self, file_path: str):
        """Process a single file"""
        if file_path not in self.processing_queue:
            return

        try:
            log.info(f'Processing file: {file_path}')

            # Convert absolute path to relative path for processing
            rel_path = os.path.relpath(file_path, self.watch_directory)

            result = self.data_loader._process_single_file(Path(file_path))

            funcs, file_data = result

            if funcs:
                log.info(f'Loading {len(funcs)} functions from {rel_path}')
                self.data_loader.load_functions(funcs)

            if file_data:
                log.info(f'Loading file data for {rel_path}')

                # Generate embedding for the file if it doesn't have one
                if not hasattr(file_data, 'embedding') or file_data.embedding is None:
                    content = file_data.content or file_data.decompress_content()
                    if content:
                        file_data.embedding = self.data_loader.generate_embedding(content)
                        self.data_loader.load_file_embeddings([file_data])
                    else:
                        log.warning(f'No content available for file embedding: {rel_path}')
                else:
                    self.data_loader.load_file_embeddings([file_data])

            log.info(f'Successfully processed: {rel_path}')

        except Exception as e:
            log.error(f'Error processing file {file_path}: {e}')
        finally:
            self.processing_queue.discard(file_path)


shutdown_requested = False


def signal_handler(sig, frame):
    global shutdown_requested
    signal_name = 'SIGTERM' if sig == signal.SIGTERM else 'SIGINT'
    log.info(f'Received {signal_name} signal, stopping file watcher...')
    shutdown_requested = True


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def main():
    parser = argparse.ArgumentParser(description='Starts a filewatcher to update embeddings')
    parser.add_argument('name', help='Name of the datastore')
    parser.add_argument('--embed-model', '-e', help='Embedding model', default=DEFAULT_MODEL)
    parser.add_argument(
        '--vector-size',
        help='Vector size to use for embeddings; MUST match datastore and be supported by model',
        type=int,
    )
    parser.add_argument('--dir', help='Directory to monitor', required=True)
    parser.add_argument(
        '--file-ext', nargs='+', help='File extensions to watch (default: .py .go .js)', default=['.py', '.go', '.js']
    )
    parser.add_argument('--num-workers', type=int, help='Number of worker threads', default=4)

    # Datastore selection (mutually exclusive)
    datastore_group = parser.add_mutually_exclusive_group(required=True)
    datastore_group.add_argument('--s3-datastore', action='store_true', help='Update S3 Vector bucket datastore')
    datastore_group.add_argument('--pgvector-datastore', action='store_true', help='Update Postgres pgvector datastore')
    datastore_group.add_argument('--chromadb-datastore', action='store_true', help='Update ChromaDB datastore')

    # S3 datastore arguments
    parser.add_argument('--s3-bucket', help='S3 bucket name')
    parser.add_argument('--s3-region', help='S3 region')
    parser.add_argument('--s3-index', help='S3 Vector bucket index')
    parser.add_argument('--aws-access-key', help='S3 access key')
    parser.add_argument('--aws-secret-key', help='S3 secret key')
    parser.add_argument('--aws-profile', help='AWS profile name to use if config dir is mounted')

    # Postgres datastore arguments
    parser.add_argument('--pg-host', help='Postgres host')
    parser.add_argument('--pg-port', type=int, help='Postgres port')
    parser.add_argument('--pg-database', help='Postgres database name')
    parser.add_argument('--pg-func-table', help='Function embeddings table name')
    parser.add_argument('--pg-file-table', help='File embeddings table name')
    parser.add_argument('--pg-user', help='Postgres user')
    parser.add_argument('--pg-password', help='Postgres password')

    # ChromaDB datastore arguments
    parser.add_argument('--chromadb-persist', action='store_true', help='Use persistent ChromaDB')
    parser.add_argument('--chromadb-path', help='ChromaDB data path', default='./chroma')
    parser.add_argument('--chromadb-collection', help='ChromaDB collection name', default='syl')

    args = parser.parse_args()

    if args.s3_datastore:
        if not args.s3_bucket:
            raise RuntimeError('S3 bucket name must be specified')
        elif not args.s3_index:
            raise RuntimeError('S3 index must be specified')

        if (not args.s3_access_key or not args.s3_secret_key) and not args.aws_profile:
            raise RuntimeError('Either AWS access keys or AWS profile must be specified')

        data_loader = S3VectorLoader(
            embed_model=args.embed_model,
            num_workers=args.num_workers,
            vector_size=args.vector_size,
            file_ext_whitelist=args.file_ext,
            s3_bucket_name=args.s3_bucket,
            s3_index_name=args.s3_index,
            aws_access_key_id=args.aws_access_key,
            aws_secret_key=args.aws_secret_key,
            aws_profile=args.aws_profile,
            region=args.s3_region,
            git_repo_url=None,
            git_branch=None,
            file_ext_blacklist=None,
        )

    elif args.pgvector_datastore:
        if not args.pg_host:
            raise RuntimeError('Postgres host must be specified')
        elif not args.pg_port:
            raise RuntimeError('Postgres port must be specified')
        elif not args.pg_database:
            raise RuntimeError('Postgres database must be specified')
        elif not args.pg_user:
            raise RuntimeError('Postgres user must be specified')
        elif not args.pg_password:
            raise RuntimeError('Postgres password must be specified')
        elif not args.pg_func_table:
            raise RuntimeError('Function embeddings table name must be specified')
        elif not args.pg_file_table:
            raise RuntimeError('File embeddings table name must be specified')

        data_loader = PostgreSQLLoader(
            git_repo_url=None,
            git_branch=None,
            embed_model=args.embed_model,
            num_workers=args.num_workers,
            vector_size=args.vector_size,
            file_ext_whitelist=args.file_ext,
            connection_string=f'postgresql://{args.pg_user}:{args.pg_password}@{args.pg_host}:{args.pg_port}/{args.pg_database}',
            func_table_name=args.pg_func_table,
            file_table_name=args.pg_file_table,
        )

    elif args.chromadb_datastore:
        if not args.vector_size:
            raise RuntimeError('Vector size must be specified for ChromaDB')

        data_loader = ChromaDBLoader(
            embed_model=args.embed_model,
            num_workers=args.num_workers,
            vector_size=args.vector_size,
            file_ext_whitelist=args.file_ext,
            persistent_client=args.chromadb_persist,
            data_path=args.chromadb_path,
            collection_name=args.chromadb_collection,
            git_repo_url=None,
            git_branch=None,
            file_ext_blacklist=None,
        )

    else:
        raise RuntimeError('Please specify a datastore: --s3-datastore, --pgvector-datastore, or --chromadb-datastore')

    # Validate directory exists
    if not os.path.exists(args.dir):
        raise RuntimeError(f'Directory does not exist: {args.dir}')

    if not os.path.isdir(args.dir):
        raise RuntimeError(f'Path is not a directory: {args.dir}')

    # Test conn..
    log.info('Testing datastore connection...')
    data_loader.test_connection()
    log.info('Datastore connection successful')

    log.info(f'Starting file watcher for datastore: {args.name}')

    file_extensions = set(ext if ext.startswith('.') else f'.{ext}' for ext in args.file_ext)
    event_handler = CodeFileHandler(data_loader=data_loader, watch_directory=args.dir, file_extensions=file_extensions)

    observer = Observer()
    observer.schedule(event_handler, args.dir, recursive=True)

    try:
        observer.start()
        log.info(f'File watcher started. Monitoring {args.dir} for changes...')
        log.info('Running indefinitely. Use SIGTERM or SIGINT to stop.')

        # Keep the main thread alive
        while observer.is_alive() and not shutdown_requested:
            observer.join(1)

        if shutdown_requested:
            log.info('Shutdown requested, stopping observer...')

    except Exception as e:
        log.error(f'Error in file watcher: {e}')
        raise
    finally:
        observer.stop()
        observer.join()
        if hasattr(data_loader, 'close'):
            data_loader.close()
        log.info('File watcher stopped')


if __name__ == '__main__':
    main()
