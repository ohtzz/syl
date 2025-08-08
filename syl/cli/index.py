import argparse
import os

from typing import List, Dict
from uuid import uuid4

from syl.common import DataProvider


# Duplicate of datastore code path const
# because importing that const slows CLI
# by 2-3 seconds as it loads heavy items
# and I don't want to fix it right now.
#
# Sorry future self.
CODE_PATH = '/code'


def generate_datastore_id():
    return str(uuid4()).split('-')[0]


def create_args(args: argparse.Namespace) -> List[str]:
    index_args = create_common_args(args)

    if args.datastore == DataProvider.PGVECTOR:
        # Local or remote pgvector
        index_args.extend(
            [
                '--pgvector-datastore',
                '--pg-host',
                args.pg_host,
                '--pg-port',
                str(args.pg_port),
                '--pg-database',
                args.pg_database,
                '--pg-func-table',
                args.pg_func_table,
                '--pg-file-table',
                args.pg_file_table,
                '--pg-user',
                args.pg_user,
                '--pg-password',
                args.pg_password,
            ]
        )

    elif args.datastore == DataProvider.CHROMA_DB:
        index_args.extend(['--chromadb-datastore'])

    elif args.datastore == DataProvider.S3_VECTOR:
        index_args.extend(
            [
                '--s3-datastore',
                '--s3-bucket',
                args.s3_bucket,
                '--s3-region',
                args.s3_region,
                '--s3-index',
                args.s3_index,
            ]
        )

        if args.aws_mount_config_dir:
            pass
        elif args.aws_profile:
            index_args.extend(['--aws-profile', args.aws_profile])
        elif args.aws_access_key and args.aws_secret_key:
            index_args.extend(
                [
                    '--aws-access-key',
                    args.aws_access_key,
                    '--aws-secret-key',
                    args.aws_secret_key,
                ]
            )
        else:
            raise RuntimeError(
                'error either --aws-mount-config-dir, --aws-profile, or --aws-access-key and --aws-secret-key are required'
            )

    else:
        raise ValueError('error unknown datastore option')

    return index_args


def create_common_args(args: argparse.Namespace) -> List[str]:
    index_args = [
        args.name,
        '--num-workers',
        str(args.num_workers),
        '--vector-size',
        str(args.vector_size),
        '--log-level',
        args.log_level,
    ]

    if args.embed_model:
        index_args.extend(['--embed-model', args.embed_model])

    if args.skip_embed:
        index_args.extend(['--skip-embed'])

    if args.no_register:
        index_args.append('--no-register')

    if args.git_repo_url:
        index_args.extend(['--git-repo-url', args.git_repo_url])

    if args.git_branch:
        index_args.extend(['--git-branch', args.git_branch])

    return index_args


def create_volumes(args: argparse.Namespace) -> Dict[str, str]:
    volumes: Dict[str, str] = {}

    if args.mount_local_code_dir:
        volumes[args.mount_local_code_dir] = CODE_PATH

    # HF model cache; makes both server and indexing containers faster as
    # the control server must load the models as well to execute queries
    if not args.no_mount_hf:
        volumes[os.path.expanduser(args.hf_cache_dir)] = '/root/.cache/huggingface/hub/'

    if args.datastore == DataProvider.S3_VECTOR and args.aws_mount_config_dir:
        volumes[os.path.expanduser('~/.aws')] = '/root/.aws'

    return volumes
