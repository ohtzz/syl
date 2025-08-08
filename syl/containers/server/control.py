from loguru import logger as log
from fastapi import FastAPI
from pydantic import BaseModel

from .registrations import get_registered_datastores

from syl.common import DataProvider
from syl.common.datastores import DatastoreRegistration, Status
from syl.containers.server.registrations import (
    remove_registered_datastore,
    update_registered_datastore_status,
    add_registered_datastore,
)
from syl.datastores.pgvector import PostgreSQLQuery
from syl.datastores.s3_vector import S3VectorQuery
from syl.datastores.chromadb import ChromaDBQuery

control_server = FastAPI(title='Syl Control Server')


class DeleteDatastore(BaseModel):
    name: str


@control_server.get('/datastores')
async def list_registered_datastores():
    return await get_registered_datastores()


@control_server.post('/datastores')
async def post_register_datastore(req: DatastoreRegistration):
    registered_datastores = await get_registered_datastores()

    if req.datastore.name in registered_datastores:
        log.warning('Datastore name already registered; overwriting..', extra={'datastore_name': req.datastore.name})

    # Create query instance based on data provider.
    #
    # This will pull the model (hopefully) before a request comes in.
    if req.datastore.data_provider == DataProvider.S3_VECTOR:
        req.datastore._data_provider_conn = S3VectorQuery(
            model_name=req.datastore.embed_model,
            bucket_name=req.datastore.s3_bucket,
            index_name=req.datastore.s3_index,
            region=req.datastore.s3_region,
        )

    elif req.datastore.data_provider == DataProvider.PGVECTOR:
        req.datastore._data_provider_conn = PostgreSQLQuery(
            model_name=req.datastore.embed_model,
            connection_string=req.datastore.pg_conn_str(),
            func_table_name=req.datastore.pg_func_table,
            file_table_name=req.datastore.pg_file_table,
            vector_size=req.datastore.embed_model_vector_size,
        )

    elif req.datastore.data_provider == DataProvider.CHROMA_DB:
        req.datastore._data_provider_conn = ChromaDBQuery(
            model_name=req.datastore.embed_model,
            vector_size=req.datastore.embed_model_vector_size,
            persistent_client=req.datastore.chroma_persistent,
            collection_name=req.datastore.name,
        )

    else:
        return {'error': 'unknown datastore data provider'}

    await add_registered_datastore(req.datastore, req.status, req.file_watch_dirs)
    log.info('Successfully registered datastore', extra={'datastore_name': req.datastore.name})

    return {'result': req.datastore.name}


@control_server.put('/datastores/{name}/status/{status}')
async def update_datastore_status(name: str, status: Status):
    try:
        # Update status
        await update_registered_datastore_status(name, status)

        # Get datastore to launch any file watchers
        # TODO

    except KeyError:
        return {'error': 'unknown datastore'}

    return {'result': status.name}


@control_server.delete('/datastores/{name}')
async def delete_registered_datastore(name: str):
    registered_datastores = await get_registered_datastores()

    if name not in registered_datastores:
        log.error('Datastore name not registered', extra={'datastore_name': name})
        return {'error': 'datastore name not registered'}

    await remove_registered_datastore(name)
    log.info('Successfully deregistered datastore', extra={'datastore_name': name})

    # Remove file watchers
    # TODO

    return {'result': name}
