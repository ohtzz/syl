import argparse

from datetime import datetime
from loguru import logger as log

from syl.common.logger import setup_logger
from syl.common.datastores import Status, DEFAULT_MODEL
from syl.common.server_api import update_datastore_status
from syl.datastores import S3VectorLoader, PostgreSQLLoader
from syl.datastores.chromadb import ChromaDBLoader
from syl.datastores.constants import PG_FUNC_TABLE_NAME, PG_FILE_TABLE_NAME

setup_logger(level='INFO', verbose=True)


def main():
    parser = argparse.ArgumentParser(description='Load and query vector embeddings and metadata')
    parser.add_argument('name', help='Name of the datastore')
    parser.add_argument('--log-level', help='Log level', default='INFO')
    parser.add_argument('--embed-model', '-e', help='Embedding model', default=DEFAULT_MODEL)
    parser.add_argument('--num-workers', '-n', help='Number of worker threads to spawn for embedding', type=int)
    parser.add_argument('--only-file-ext', help='Only generate embeddings and store these file types')
    parser.add_argument('--exclude-ext', nargs='*', help='Exclude certain file extensions')
    parser.add_argument(
        '--git-repo-url', help='Git repository URL to clone if this is a remote datastore', required=False
    )
    parser.add_argument('--git-branch', help='Git repository branch to checkout after cloning', required=False)
    parser.add_argument(
        '--vector-size',
        help='Vector size to use for embeddings; MUST match datastore and be supported by model',
        type=int,
    )
    parser.add_argument(
        '--skip-embed', action='store_true', help='Skip embedding if the datastore already contains data'
    )
    parser.add_argument('--no-register', action='store_true', help='Do not register this datastore with the Syl server')

    # Datastore selection (mutually exclusive)
    datastore_group = parser.add_mutually_exclusive_group(required=True)
    datastore_group.add_argument('--s3-datastore', action='store_true', help='Use existing S3 Vector bucket datastore')
    datastore_group.add_argument(
        '--pgvector-datastore', action='store_true', help='Use existing or Syl Postgres pgvector datastore'
    )
    datastore_group.add_argument(
        '--chromadb-datastore', action='store_true', help='Use existing Syl ChromaDB datastore'
    )

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
    parser.add_argument('--pg-func-table', help='Function embeddings table name', default=PG_FUNC_TABLE_NAME)
    parser.add_argument('--pg-file-table', help='File embeddings table name', default=PG_FILE_TABLE_NAME)
    parser.add_argument('--pg-user', help='Postgres user')
    parser.add_argument('--pg-password', help='Postgres password')

    args = parser.parse_args()

    setup_logger(level=args.log_level)

    if args.s3_datastore:
        if not args.s3_bucket:
            raise RuntimeError('S3 bucket name must be specified')
        elif not args.s3_index:
            raise RuntimeError('S3 index must be specified')

        data_loader = S3VectorLoader(
            embed_model=args.embed_model,
            num_workers=args.num_workers,
            vector_size=args.vector_size,
            file_ext_whitelist=args.only_file_ext,
            file_ext_blacklist=args.exclude_ext,
            git_repo_url=args.git_repo_url,
            git_branch=args.git_branch,
            s3_bucket_name=args.s3_bucket,
            s3_index_name=args.s3_index,
            aws_access_key_id=args.aws_access_key,
            aws_secret_key=args.aws_secret_key,
            aws_profile=args.aws_profile,
            region=args.s3_region,
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
            embed_model=args.embed_model,
            num_workers=args.num_workers,
            vector_size=args.vector_size,
            file_ext_whitelist=args.only_file_ext,
            file_ext_blacklist=args.exclude_ext,
            git_repo_url=args.git_repo_url,
            git_branch=args.git_branch,
            connection_string=f'postgresql://{args.pg_user}:{args.pg_password}@{args.pg_host}:{args.pg_port}/{args.pg_database}',
            func_table_name=args.pg_func_table,
            file_table_name=args.pg_file_table,
        )

    elif args.chromadb_datastore:
        data_loader = ChromaDBLoader(
            embed_model=args.embed_model,
            num_workers=args.num_workers,
            vector_size=args.vector_size,
            file_ext_whitelist=args.only_file_ext,
            file_ext_blacklist=args.exclude_ext,
            git_repo_url=args.git_repo_url,
            git_branch=args.git_branch,
            persistent_client=False,  # TODO
        )

    else:
        raise RuntimeError('Please specify either --s3-datastore or --pgvector datastore')

    status = Status.registered

    if not args.skip_embed:
        # Setup data loader items; also tests connection to ensure we can actually reach
        # the S3 Vector bucket/PG database before wasting time embedding stuff..

        try:
            log.info('Setting up data loader..')
            start_time = datetime.now()

            data_loader.setup()

            end_time = datetime.now()
            log.info(
                'Setup data loader!',
                extra={
                    'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': f'{((end_time - start_time).total_seconds() / 60):.2f} mins',
                },
            )

            # Embed and load data
            log.info('Generating embeddings..')
            start_time = datetime.now()

            data_loader.embed_and_load_data()

            end_time = datetime.now()
            log.info(
                'Generated embeddings!',
                extra={
                    'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': f'{((end_time - start_time).total_seconds() / 60):.2f} mins',
                },
            )

        except Exception as e:
            log.error('Indexing failed. Registering as failed..')
            update_datastore_status(name=args.name, status=Status.failed)
            raise e

    log.info('Finished indexing datastore data')

    if not args.no_register:
        log.info('Updating datastore status with Syl server..', extra={'datastore_name': args.name})
        update_datastore_status(name=args.name, status=status)
        log.info('Successfully updated datastore')


if __name__ == '__main__':
    main()
