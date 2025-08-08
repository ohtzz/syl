import re

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, PrivateAttr
from typing import Optional, List, Tuple, Dict, Any


DATASTORE_NAME_RE = r'^[a-zA-Z\d_-]+$'
DEFAULT_MODEL = 'BAAI/bge-base-en-v1.5'  # Seems better than MSFT code-bert base


class DataProvider(str, Enum):
    S3_VECTOR = 's3vector'
    PGVECTOR = 'pgvector'
    CHROMA_DB = 'chromadb'


class Datastore(BaseModel):
    name: str
    description: Optional[str] = None
    embed_model: str
    embed_model_vector_size: int
    data_provider: DataProvider
    _data_provider_conn: Optional[dict] = PrivateAttr(default=None)

    s3_bucket: Optional[str] = None
    s3_index: Optional[str] = None
    s3_region: Optional[str] = None

    pg_host: Optional[str] = None  # syl-pgvector-$PROJECT or remote..
    pg_port: Optional[int] = None
    pg_user: Optional[str] = None
    pg_password: Optional[str] = None
    pg_database: Optional[str] = None
    pg_func_table: Optional[str] = None
    pg_file_table: Optional[str] = None

    chroma_persistent: Optional[bool] = False

    def pg_conn_str(self) -> str:
        return f'postgresql://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_database}'


class Status(str, Enum):
    registered = 'registered'
    failed = 'failed'
    indexing = 'indexing'
    unknown = 'unknown'
    other = 'other'


class DatastoreRegistration(BaseModel):
    datastore: Datastore
    status: Status
    file_watch_dirs: Optional[List[str]] = None
    first_seen: Optional[datetime] = None
    registered_at: Optional[datetime] = None


def valid_name(name: str) -> bool:
    """Validates whether a name (datastore, server, etc.) is valid.
    We use the same charset for everything.
    """
    return re.match(DATASTORE_NAME_RE, name) is not None


def get_registered_status(container_status: str, ds_reg: DatastoreRegistration) -> Tuple[bool, str, str]:
    """Returns the registration, status, and container state

    registered: boolean indicating whether the datastore.status == Status.registered
    status: 'available' if the above status is registered, otherwise the value of datastore.status
    container: Whatever container_status was passed in
    """

    if ds_reg is None:
        return False, 'not found', container_status

    _status = ds_reg.get('status', 'status not found')
    registered = _status == Status.registered
    status = _status
    if registered and container_status == 'running':
        status = 'available'

    return registered, status, container_status

    # Not sure how I broke these types..
    # registered = ds_reg.status == Status.registered
    #
    # status = ds_reg.status
    # if registered and container_status == 'running':
    #     status = 'available'
    #
    # return registered, status, container_status
