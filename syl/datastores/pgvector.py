import gzip
import base64
import numpy as np

from datetime import datetime
from typing import Dict, List, Any, Optional

from loguru import logger as log
from sqlalchemy import create_engine, Column, String, Integer, Boolean, DateTime, Text, ARRAY, Index, text, event
from sqlalchemy.orm import declarative_base, sessionmaker, Query
from sqlalchemy.orm.decl_api import DeclarativeMeta
from pgvector.sqlalchemy import Vector
from pgvector.psycopg2 import register_vector

from syl.parsers.parser import FileEmbedding
from syl.datastores.datastore import DataStoreLoader, DataStoreQuery

from .constants import PG_FUNC_TABLE_NAME, PG_FILE_TABLE_NAME
from .utils import extract_filename, ResultFormatter
from syl.common.tools import SearchSemanticFunctionsRequest

Base: DeclarativeMeta = declarative_base()


class FunctionEmbeddingTable(Base):
    __tablename__ = PG_FUNC_TABLE_NAME
    __table_args__ = (
        Index(
            'idx_code_embedding_cosine',
            'embedding',
            postgresql_using='ivfflat',
            postgresql_ops={'embedding': 'vector_cosine_ops'},
        ),
        Index(
            'idx_code_embedding_euclidean',
            'embedding',
            postgresql_using='ivfflat',
            postgresql_ops={'embedding': 'vector_l2_ops'},
        ),
        Index('idx_code_filename', 'filename'),
        Index('idx_code_function_name', 'function_name'),
        Index('idx_code_complexity', 'complexity'),
        {'extend_existing': True},
    )

    id = Column(String, primary_key=True)
    embedding = Column(Vector(768))  # pgvector type with default dimension
    content = Column(Text)
    content_compressed = Column(Text)  # b64 gzip
    filename = Column(String)
    file_path = Column(String)
    function_name = Column(String)
    line_start = Column(Integer)
    line_end = Column(Integer)
    calls = Column(ARRAY(String))
    called_by = Column(ARRAY(String))
    complexity = Column(Integer)
    maintainability_index = Column(Integer)
    parameters = Column(ARRAY(String))
    returns = Column(ARRAY(String))
    docstring = Column(Text)
    is_async = Column(Boolean, default=False)
    is_method = Column(Boolean, default=False)
    class_name = Column(String)
    is_compressed = Column(Boolean, default=False)
    updated_at = Column(DateTime, default=datetime.utcnow)


class FileEmbeddingTable(Base):
    __tablename__ = PG_FILE_TABLE_NAME
    __table_args__ = (
        Index(
            'idx_file_embedding_cosine',
            'embedding',
            postgresql_using='ivfflat',
            postgresql_ops={'embedding': 'vector_cosine_ops'},
        ),
        Index(
            'idx_file_embedding_euclidean',
            'embedding',
            postgresql_using='ivfflat',
            postgresql_ops={'embedding': 'vector_l2_ops'},
        ),
        Index('idx_file_filename', 'filename'),
        Index('idx_file_file_type', 'file_type'),
        Index('idx_file_file_path', 'file_path'),
        {'extend_existing': True},
    )

    id = Column(String, primary_key=True)
    embedding = Column(Vector(768))  # pgvector type with default dimension
    file_path = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    content = Column(Text)
    content_compressed = Column(Text)  # b64 gzip
    imports = Column(ARRAY(String))
    file_size = Column(Integer)
    line_count = Column(Integer)
    file_type = Column(String, nullable=False, default='other')  # code, docs, config, data, other
    is_compressed = Column(Boolean, default=False)
    updated_at = Column(DateTime, default=datetime.utcnow)


def create_function_embedding_table(table_name: str, vector_size: int = 768):
    """Factory function to create FunctionEmbeddingTable classes with different table names"""

    class_name = f'FunctionEmbeddingTable_{table_name}'

    return type(
        class_name,
        (Base,),
        {
            '__tablename__': table_name,
            '__table_args__': (
                Index(
                    f'idx_{table_name}_embedding_cosine',
                    'embedding',
                    postgresql_using='ivfflat',
                    postgresql_ops={'embedding': 'vector_cosine_ops'},
                ),
                Index(
                    f'idx_{table_name}_embedding_euclidean',
                    'embedding',
                    postgresql_using='ivfflat',
                    postgresql_ops={'embedding': 'vector_l2_ops'},
                ),
                Index(f'idx_{table_name}_filename', 'filename'),
                Index(f'idx_{table_name}_function_name', 'function_name'),
                Index(f'idx_{table_name}_complexity', 'complexity'),
                {'extend_existing': True},
            ),
            'id': Column(String, primary_key=True),
            'embedding': Column(Vector(vector_size)),
            'content': Column(Text),
            'content_compressed': Column(Text),
            'is_compressed': Column(Boolean, default=False),
            'filename': Column(String),
            'file_path': Column(String),
            'function_name': Column(String),
            'line_start': Column(Integer),
            'line_end': Column(Integer),
            'calls': Column(ARRAY(String)),
            'called_by': Column(ARRAY(String)),
            'complexity': Column(Integer),
            'maintainability_index': Column(Integer),
            'parameters': Column(ARRAY(String)),
            'returns': Column(ARRAY(String)),
            'docstring': Column(Text),
            'is_async': Column(Boolean, default=False),
            'is_method': Column(Boolean, default=False),
            'class_name': Column(String),
            'updated_at': Column(DateTime, default=datetime.utcnow),
        },
    )


def create_file_embedding_table(table_name: str, vector_size: int = 768):
    """Factory function to create FileEmbeddingTable classes with different table names"""

    class_name = f'FileEmbeddingTable_{table_name}'

    return type(
        class_name,
        (Base,),
        {
            '__tablename__': table_name,
            '__table_args__': (
                Index(
                    f'idx_{table_name}_embedding_cosine',
                    'embedding',
                    postgresql_using='ivfflat',
                    postgresql_ops={'embedding': 'vector_cosine_ops'},
                ),
                Index(
                    f'idx_{table_name}_embedding_euclidean',
                    'embedding',
                    postgresql_using='ivfflat',
                    postgresql_ops={'embedding': 'vector_l2_ops'},
                ),
                Index(f'idx_{table_name}_filename', 'filename'),
                Index(f'idx_{table_name}_file_type', 'file_type'),
                Index(f'idx_{table_name}_file_path', 'file_path'),
                {'extend_existing': True},
            ),
            'id': Column(String, primary_key=True),
            'embedding': Column(Vector(vector_size)),
            'file_path': Column(String, nullable=False),
            'filename': Column(String, nullable=False),
            'content': Column(Text),
            'content_compressed': Column(Text),  # b64 gzip
            'imports': Column(ARRAY(String)),
            'file_size': Column(Integer),
            'line_count': Column(Integer),
            'file_type': Column(String, nullable=False, default='other'),  # code, docs, config, data, other
            'is_compressed': Column(Boolean, default=False),
            'updated_at': Column(DateTime, default=datetime.utcnow),
        },
    )


class PostgreSQLLoader(DataStoreLoader):
    def __init__(
        self,
        embed_model: str,
        num_workers: int,
        file_ext_whitelist: Optional[List[str]],
        file_ext_blacklist: Optional[List[str]],
        git_repo_url: Optional[str],
        git_branch: Optional[str],
        connection_string: str,
        vector_size: int,
        func_table_name: str = PG_FUNC_TABLE_NAME,
        file_table_name: str = PG_FILE_TABLE_NAME,
    ):
        super().__init__(
            embed_model=embed_model,
            num_workers=num_workers,
            file_ext_whitelist=file_ext_whitelist,
            file_ext_blacklist=file_ext_blacklist,
            git_repo_url=git_repo_url,
            git_branch=git_branch,
            vector_size=vector_size,
        )

        self.connection_string = connection_string
        self.func_table_name = func_table_name
        self.file_table_name = file_table_name
        self.vector_size = vector_size

        try:
            self.engine = create_engine(connection_string, echo=False)
            self.SessionLocal = sessionmaker(bind=self.engine)

            # Create separate model classes for this instance to avoid global modification.
            # There's probably a better way to handle this..
            # TODO read sql alchemy docs if I get bored enough
            self.FunctionEmbeddingTable = create_function_embedding_table(func_table_name, vector_size)
            self.FileEmbeddingTable = create_file_embedding_table(file_table_name, vector_size)

            log.info('Connected to PostgreSQL database via SQLAlchemy')
            self._setup_database()

            # Register pgvector with psycopg after extension is created
            with self.engine.connect() as conn:
                register_vector(conn.connection.dbapi_connection)

            # Also set up event listener for future connections
            @event.listens_for(self.engine, 'connect')
            def connect(dbapi_connection, connection_record):
                register_vector(dbapi_connection)

        except Exception as e:
            log.error(f'Failed to connect to PostgreSQL: {e}')
            raise

    def _setup_database(self):
        """Create database schema and enable pgvector extension"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))
                conn.commit()

                # Create only the custom tables for this instance
                self.FunctionEmbeddingTable.__table__.create(self.engine, checkfirst=True)
                self.FileEmbeddingTable.__table__.create(self.engine, checkfirst=True)

                log.info(f'Database schema setup completed for tables: {self.func_table_name}, {self.file_table_name}')

        except Exception as e:
            log.error(f'error setting up database: {e}')
            raise

    def load_functions(self, funcs: List):
        """Load function embeddings into PostgreSQL"""
        postgres_records = []
        for func in funcs:
            try:
                record = self.prepare_function_record(func)
                postgres_records.append(record)
            except Exception as e:
                log.error(f'Failed to prepare record for function {getattr(func, "id", "unknown")}: {e}')
                continue

        if postgres_records:
            self.upload_functions_to_postgres(postgres_records)
        else:
            log.warning('No valid function records to upload to PostgreSQL')

    def load_file_embeddings(self, file_data_with_embeddings: List[FileEmbedding]):
        """Load file embeddings into file_embeddings table"""
        if not file_data_with_embeddings:
            log.warning('No file embeddings to load into PostgreSQL')
            return

        embedding_records = []
        for file_data in file_data_with_embeddings:
            try:
                record = self.prepare_file_embedding_record(file_data)
                embedding_records.append(record)
            except Exception as e:
                log.error(f'error preparing file embedding record {file_data.file_path}: {e}')
                continue

        if embedding_records:
            self.upload_file_embeddings_to_postgres(embedding_records)
        else:
            log.warning('No valid file embedding records to upload to PostgreSQL')

    def prepare_function_record(self, function_data) -> 'FunctionEmbeddingTable':
        """Convert functions to SQLAlchemy models"""

        def get_field(obj, field_name, default=None):
            """Helper to get field from either dataclass or dict"""
            if hasattr(obj, field_name):
                return getattr(obj, field_name)
            elif hasattr(obj, 'get'):
                return obj.get(field_name, default)
            else:
                return default

        def ensure_list(value):
            """Ensure value is a list for PG array fields"""
            if value is None:
                return None
            elif isinstance(value, list):
                return value
            elif isinstance(value, str):
                return [value]
            else:
                return list(value) if hasattr(value, '__iter__') else [str(value)]

        embedding = get_field(function_data, 'embedding', [])
        func_id = get_field(function_data, 'id', '')
        code = get_field(function_data, 'code', '')

        if not embedding:
            raise ValueError('No embedding found in function data')

        # Convert embedding to list for pgvector
        if isinstance(embedding, np.ndarray):
            embedding_list = embedding.tolist()
        else:
            embedding_list = list(embedding)

        # Handle content compression
        content = None
        content_compressed = None
        is_compressed = False

        if code:
            # Compress if code is large (>1KB)
            if len(code.encode('utf-8')) > 1024:
                compressed = gzip.compress(code.encode('utf-8'))
                content_compressed = base64.b64encode(compressed).decode('ascii')
                is_compressed = True
            else:
                content = code

        record = self.FunctionEmbeddingTable(
            id=func_id,
            embedding=embedding_list,
            content=content,
            content_compressed=content_compressed,
            is_compressed=is_compressed,
            function_name=get_field(function_data, 'function_name'),
            file_path=get_field(function_data, 'file_path'),
            filename=extract_filename(get_field(function_data, 'file_path', '')),
            line_start=get_field(function_data, 'line_start'),
            line_end=get_field(function_data, 'line_end'),
            calls=ensure_list(get_field(function_data, 'calls')),
            called_by=ensure_list(get_field(function_data, 'called_by')),
            complexity=get_field(function_data, 'complexity'),
            maintainability_index=get_field(function_data, 'maintainability_index'),
            parameters=ensure_list(get_field(function_data, 'parameters')),
            returns=ensure_list(get_field(function_data, 'returns')),
            docstring=get_field(function_data, 'docstring'),
            is_async=get_field(function_data, 'is_async', False),
            is_method=get_field(function_data, 'is_method', False),
            class_name=get_field(function_data, 'class_name'),
            updated_at=datetime.utcnow(),
        )

        return record

    def prepare_file_embedding_record(self, file_data: FileEmbedding):
        """Convert FileEmbedding with embedding to SQLAlchemy model"""
        # Handle content compression
        content = None
        content_compressed = None
        is_compressed = False

        if file_data.content_compressed:
            content_compressed = file_data.content_compressed
            is_compressed = True
        else:
            content = file_data.content

        record = self.FileEmbeddingTable(
            id=file_data.id,
            embedding=file_data.embedding,
            file_path=file_data.file_path,
            filename=file_data.filename,
            content=content,
            content_compressed=content_compressed,
            imports=file_data.imports or [],
            file_size=file_data.file_size,
            line_count=file_data.line_count,
            file_type=file_data.file_type,
            is_compressed=is_compressed,
            updated_at=datetime.utcnow(),
        )

        return record

    def upload_functions_to_postgres(self, records: List, batch_size: int = 100):
        log.info(f'Uploading {len(records)} function records to PostgreSQL')

        total_uploaded = 0
        failed_uploads = []

        try:
            session = self.SessionLocal()

            # First delete existing functions from the same files
            file_paths = list(set(record.file_path for record in records if record.file_path))
            if file_paths:
                for file_path in file_paths:
                    deleted_count = (
                        session.query(self.FunctionEmbeddingTable)
                        .filter(self.FunctionEmbeddingTable.file_path == file_path)
                        .delete()
                    )
                    if deleted_count > 0:
                        log.info(f'Deleted {deleted_count} existing functions from {file_path}')
                session.commit()

            # Batch
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]

                try:
                    log.info(
                        f'Uploading function batch {i // batch_size + 1}/{(len(records) - 1) // batch_size + 1} ({len(batch)} records)...'
                    )

                    for record in batch:
                        # Merge for upsert
                        session.merge(record)

                    session.commit()
                    total_uploaded += len(batch)
                    log.info(f'Successfully uploaded function batch {i // batch_size + 1}')

                except Exception as e:
                    log.error(f'Failed to upload function batch starting at index {i}: {e}')
                    session.rollback()
                    failed_uploads.extend([record.id for record in batch])
                    continue

        except Exception as e:
            log.error(f'Database operation failed: {e}')
            raise
        finally:
            session.close()

        log.info(f'Function upload complete: {total_uploaded}/{len(records)} successful')

        if failed_uploads:
            log.warning(f'Failed to upload {len(failed_uploads)} function records')
            return False, failed_uploads

        return True, []

    def upload_file_embeddings_to_postgres(self, records: List, batch_size: int = 100):
        log.info(f'Uploading {len(records)} file embedding records to PostgreSQL')

        total_uploaded = 0
        failed_uploads = []

        try:
            session = self.SessionLocal()

            # First delete existing file embeddings from the same files
            file_paths = list(set(record.file_path for record in records if record.file_path))
            if file_paths:
                for file_path in file_paths:
                    deleted_count = (
                        session.query(self.FileEmbeddingTable).filter(self.FileEmbeddingTable.file_path == file_path).delete()
                    )
                    if deleted_count > 0:
                        log.info(f'Deleted {deleted_count} existing file embeddings for {file_path}')
                session.commit()

            # Batch upload
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]

                try:
                    log.info(
                        f'Uploading file embedding batch {i // batch_size + 1}/{(len(records) - 1) // batch_size + 1} ({len(batch)} records)...'
                    )

                    for record in batch:
                        # Use merge for upsert functionality
                        session.merge(record)

                    session.commit()
                    total_uploaded += len(batch)
                    log.info(f'Successfully uploaded file embedding batch {i // batch_size + 1}')

                except Exception as e:
                    log.error(f'Failed to upload file embedding batch starting at index {i}: {e}')
                    session.rollback()
                    failed_uploads.extend([record.id for record in batch])
                    continue

        except Exception as e:
            log.error(f'Database operation failed: {e}')
            raise
        finally:
            session.close()

        log.info(f'File embedding upload complete: {total_uploaded}/{len(records)} successful')

        if failed_uploads:
            log.warning(f'Failed to upload {len(failed_uploads)} file embedding records')
            return False, failed_uploads

        return True, []

    def test_connection(self):
        try:
            with self.engine.connect() as conn:
                conn.execute(text('SELECT 1'))
            log.info('Database connection test successful')
            return True
        except Exception as e:
            log.error(f'Database connection test failed: {e}')
            return False

    def close(self):
        if hasattr(self, 'engine'):
            self.engine.dispose()
            log.info('Closed PostgreSQL connection')


class PostgreSQLQuery(DataStoreQuery):
    def __init__(
        self,
        connection_string: str,
        func_table_name: str,
        file_table_name: str,
        model_name: str,
        distance_metric: str = 'cosine',
        vector_size: int = 768,
    ):
        super().__init__(model_name=model_name)

        self.connection_string = connection_string
        self.func_table_name = func_table_name
        self.file_table_name = file_table_name
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.vector_size = vector_size

        try:
            self.engine = create_engine(connection_string, echo=True)

            # Register pgvector with psycopg
            @event.listens_for(self.engine, 'connect')
            def connect(dbapi_connection, connection_record):
                register_vector(dbapi_connection)

            self.SessionLocal = sessionmaker(bind=self.engine)

            # Create separate model classes for this instance to avoid global modification.
            # There's probably a better way to handle this..
            self.FunctionEmbeddingTable = create_function_embedding_table(func_table_name, vector_size)
            self.FileEmbeddingTable = create_file_embedding_table(file_table_name, vector_size)

            log.info('Connected to PostgreSQL database via SQLAlchemy')
        except Exception as e:
            log.error(f'Failed to connect to PostgreSQL: {e}')
            raise

    def get_file_content(self, file_path: str, line_start: Optional[int] = None, line_end: Optional[int] = None) -> bytes:
        try:
            session = self.SessionLocal()
            result = session.query(self.FileEmbeddingTable).filter(self.FileEmbeddingTable.file_path == file_path).first()

            if not result:
                raise FileNotFoundError(f'File not found: {file_path}')

            if result.is_compressed and result.content_compressed:
                compressed = base64.b64decode(result.content_compressed.encode('ascii'))
                content = gzip.decompress(compressed).decode('utf-8')
            else:
                content = result.content or ''

            if line_start is None and line_end is None:
                return content.encode('utf-8')

            lines = content.split('\n')
            start_idx = max(0, (line_start - 1) if line_start else 0)
            end_idx = min(len(lines), line_end if line_end else len(lines))

            selected_lines = lines[start_idx:end_idx]
            result_content = '\n'.join(selected_lines)
            return result_content.encode('utf-8')

        except Exception as e:
            log.error(f'Failed to retrieve file content: {e}')
            raise
        finally:
            session.close()

    def get_files_by_name(self, filename: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Retrieve files by name"""
        try:
            session = self.SessionLocal()
            results = session.query(self.FileEmbeddingTable).filter(
                self.FileEmbeddingTable.filename == filename).limit(max_results).all()

            return [self._convert_file_result_to_dict(result) for result in results]

        except Exception as e:
            log.error(f'Failed to retrieve files by name: {e}')
            raise
        finally:
            session.close()

    @staticmethod
    def _convert_func_result_to_dict(result, distance: float = 0.0, include_function_content: Optional[bool] = False) -> Dict[str, Any]:
        """Helper function to convert SQLAlchemy result to dictionary"""

        # Slim dict
        d = {
            'file_path': result.file_path,
            'filename': result.filename,
            'function_name': result.function_name,
            'line_start': result.line_start,
            'line_end': result.line_end,
            'calls': result.calls,
            'called_by': result.called_by,
            'complexity': result.complexity,
            'maintainability_index': result.maintainability_index,
            'parameters': result.parameters,
            'returns': result.returns,
            'docstring': result.docstring,
            'is_async': result.is_async,
            'is_method': result.is_method,
            'class_name': result.class_name,
        }

        if distance > 0 :
            d['distance']: distance

        if include_function_content:
            if result.is_compressed and result.content_compressed:
                compressed = base64.b64decode(result.content_compressed.encode('ascii'))
                content = gzip.decompress(compressed).decode('utf-8')
            else:
                content = result.content or ''
            d['content'] = content

        return d

    @staticmethod
    def _convert_file_result_to_dict(result, distance: float = 0.0, include_file_content: Optional[bool] = False) -> Dict[str, Any]:
        """Helper function to convert SQLAlchemy result to dictionary"""

        # Slim dict
        d = {
            'file_path': result.file_path,
            'filename': result.filename,
            'imports': result.imports,
            'file_size': result.file_size,
            'line_count': result.line_count,
            'file_type': result.file_type,
            'updated_at': result.updated_at,
        }

        if distance > 0 :
            d['distance']: distance

        if include_file_content:
            if result.is_compressed and result.content_compressed:
                compressed = base64.b64decode(result.content_compressed.encode('ascii'))
                content = gzip.decompress(compressed).decode('utf-8')
            else:
                content = result.content or ''
            d['content'] = content

        return d

    def query_semantic_functions(self, req: SearchSemanticFunctionsRequest, limit: int = 10) -> Dict[str, Any]:
        """Queries functions with optional semantic search returning slim dicts"""

        sql_filters = req.to_sqlalchemy_filter(self.FunctionEmbeddingTable)
        limit = req.max_results or 10

        log.info(
            'Querying Postgres function vectors',
            extra={
                'raw_query': req.query,
                'full_request': req,
                'limit': limit,
                'sql_request_filters': sql_filters.get_filters(),
            },
        )

        try:
            session = self.SessionLocal()
            query = session.query(self.FunctionEmbeddingTable)

            if sql_filters:
                query = sql_filters.apply_to_query(query)

            if req.query:
                query_embedding = self.generate_query_embedding(req.query)

                if self.distance_metric == 'cosine':
                    distance_expr = self.FunctionEmbeddingTable.embedding.cosine_distance(query_embedding)
                elif self.distance_metric == 'euclidean':
                    distance_expr = self.FunctionEmbeddingTable.embedding.l2_distance(query_embedding)
                else:
                    raise ValueError(f'Unknown distance metric: {self.distance_metric}')

                query = query.add_columns(distance_expr.label('distance')).order_by(distance_expr)
                results = query.limit(limit).all()

                result_dicts = []
                for result, distance in results:
                    result_dict = self._convert_func_result_to_dict(result, float(distance),
                                                                    req.include_function_content)
                    result_dicts.append(result_dict)

            # No semantic search, just filtered results
            else:
                results = query.limit(limit).all()
                result_dicts = [self._convert_func_result_to_dict(result, 0.0, req.include_function_content)
                                for result in results]

            log.debug(
                'Raw results',
                extra={
                    'num_results': len(results),
                },
            )

            return result_dicts

        except Exception as e:
            log.error(f'Error querying function vectors table: {e}')
            raise
        finally:
            session.close()

    def query_semantic_files(self, req, limit: int = 10) -> Dict[str, Any]:
        sql_filters = req.to_sqlalchemy_filter(self.FileEmbeddingTable)
        limit = req.max_results or 10

        log.info(
            'Querying Postgres file vectors',
            extra={
                'raw_query': req.query,
                'full_request': req,
                'limit': limit,
                'sql_request_filters': sql_filters.get_filters(),
            },
        )

        try:
            session = self.SessionLocal()
            query = session.query(self.FileEmbeddingTable)

            if sql_filters:
                query = sql_filters.apply_to_query(query)

            if req.query:
                query_embedding = self.generate_query_embedding(req.query)

                if self.distance_metric == 'cosine':
                    distance_expr = self.FileEmbeddingTable.embedding.cosine_distance(query_embedding)
                elif self.distance_metric == 'euclidean':
                    distance_expr = self.FileEmbeddingTable.embedding.l2_distance(query_embedding)
                else:
                    raise ValueError(f'Unknown distance metric: {self.distance_metric}')

                query = query.add_columns(distance_expr.label('distance')).order_by(distance_expr)
                results = query.limit(limit).all()

                result_dicts = []
                for result, distance in results:
                    result_dict = self._convert_file_result_to_dict(result, float(distance),
                                                                    req.include_file_content)
                    result_dicts.append(result_dict)

            # No semantic search, just filtered results
            else:
                results = query.limit(limit).all()
                result_dicts = [self._convert_file_result_to_dict(result, 0.0, req.include_file_content)
                                for result in results]

            log.debug(
                'Raw results',
                extra={
                    'num_results': len(results),
                },
            )

            return result_dicts

        except Exception as e:
            log.error(f'Error querying file vectors table: {e}')
            raise
        finally:
            session.close()

    def query_filters(self, **filters) -> Dict[str, Any]:
        try:
            session = self.SessionLocal()
            query: Query[Any] = session.query(self.FunctionEmbeddingTable)

            if 'min_complexity' in filters:
                query = query.filter(self.FunctionEmbeddingTable.complexity >= filters['min_complexity'])
            if 'function_name' in filters:
                query = query.filter(self.FunctionEmbeddingTable.function_name == filters['function_name'])
            if 'filename' in filters:
                filename = filters['filename']
                query = query.filter(
                    (self.FunctionEmbeddingTable.file_path.like(f'%/{filename}'))
                    | (self.FunctionEmbeddingTable.file_path.like(f'%{filename}'))
                )
            if 'is_async' in filters:
                query = query.filter(self.FunctionEmbeddingTable.is_async == filters['is_async'])
            if 'is_method' in filters:
                query = query.filter(self.FunctionEmbeddingTable.is_method == filters['is_method'])

            max_results = filters.get('max_results', 20)
            results = query.limit(max_results).all()

            result_dicts = []
            for result in results:
                result_dict = {
                    'id': result.id,
                    'function_name': result.function_name,
                    'file_path': result.file_path,
                    'filename': result.filename,
                    'line_start': result.line_start,
                    'line_end': result.line_end,
                    'calls': result.calls,
                    'called_by': result.called_by,
                    'complexity': result.complexity,
                    'maintainability_index': result.maintainability_index,
                    'parameters': result.parameters,
                    'returns': result.returns,
                    'docstring': result.docstring,
                    'is_async': result.is_async,
                    'is_method': result.is_method,
                    'class_name': result.class_name,
                    'updated_at': result.updated_at,
                    'distance': 0.0,  # No distance for filter-based queries
                }
                result_dicts.append(result_dict)

            formatted_results = ResultFormatter.format_function_results(
                result_dicts, 'filter-based query', 'postgresql', False
            )
            return formatted_results

        except Exception as e:
            log.error(f'Failed to query with filters: {e}')
            raise
        finally:
            session.close()

    def get_function_content(self, function_id: str) -> str:
        try:
            session = self.SessionLocal()

            result = (
                session.query(self.FunctionEmbeddingTable).filter(self.FunctionEmbeddingTable.id == function_id).first()
            )

            if not result:
                raise ValueError(f'Function not found: {function_id}')

            # Handle content decompression
            if result.content:
                return result.content
            elif result.is_compressed and result.content_compressed:
                try:
                    compressed = base64.b64decode(result.content_compressed.encode('ascii'))
                    return gzip.decompress(compressed).decode('utf-8')
                except Exception as e:
                    log.error(f'Failed to decompress function content for {function_id}: {e}')
                    return ''
            else:
                log.warning(f'No code found for function {function_id}')
                return ''

        except Exception as e:
            log.error(f'Error getting function content for {function_id}: {e}')
            raise
        finally:
            session.close()

    def get_available_files(self) -> List[str]:
        try:
            session = self.SessionLocal()

            results = session.query(self.FileEmbeddingTable.file_path).all()
            file_paths = [result[0] for result in results]

            return sorted(file_paths)

        except Exception as e:
            log.error(f'Error getting available files: {e}')
            raise
        finally:
            session.close()

    def get_function_callers(self, function_name: str) -> Dict[str, Any]:
        try:
            session = self.SessionLocal()

            results = (
                session.query(self.FunctionEmbeddingTable)
                .filter(self.FunctionEmbeddingTable.calls.contains([function_name]))
                .all()
            )

            result_dicts = []
            for result in results:
                result_dict = {
                    'id': result.id,
                    'function_name': result.function_name,
                    'file_path': result.file_path,
                    'filename': result.filename,
                    'line_start': result.line_start,
                    'line_end': result.line_end,
                    'calls': result.calls,
                    'called_by': result.called_by,
                    'complexity': result.complexity,
                    'maintainability_index': result.maintainability_index,
                    'parameters': result.parameters,
                    'returns': result.returns,
                    'docstring': result.docstring,
                    'is_async': result.is_async,
                    'is_method': result.is_method,
                    'class_name': result.class_name,
                    'updated_at': result.updated_at,
                    'distance': 0.0,  # No distance for relationship queries
                }
                result_dicts.append(result_dict)

            formatted_results = ResultFormatter.format_function_results(
                result_dicts, f'Functions that call {function_name}', 'postgresql', False
            )
            return formatted_results

        except Exception as e:
            log.error(f'Error getting function callers for {function_name}: {e}')
            raise
        finally:
            session.close()

    def get_function_called_by(self, function_name: str) -> Dict[str, Any]:
        try:
            session = self.SessionLocal()

            results = (
                session.query(self.FunctionEmbeddingTable)
                .filter(self.FunctionEmbeddingTable.called_by.contains([function_name]))
                .all()
            )

            result_dicts = []
            for result in results:
                result_dict = {
                    'id': result.id,
                    'function_name': result.function_name,
                    'file_path': result.file_path,
                    'filename': result.filename,
                    'line_start': result.line_start,
                    'line_end': result.line_end,
                    'calls': result.calls,
                    'called_by': result.called_by,
                    'complexity': result.complexity,
                    'maintainability_index': result.maintainability_index,
                    'parameters': result.parameters,
                    'returns': result.returns,
                    'docstring': result.docstring,
                    'is_async': result.is_async,
                    'is_method': result.is_method,
                    'class_name': result.class_name,
                    'updated_at': result.updated_at,
                    'distance': 0.0,  # No distance for relationship queries
                }
                result_dicts.append(result_dict)

            formatted_results = ResultFormatter.format_function_results(
                result_dicts, f'Functions called by {function_name}', 'postgresql', False
            )
            return formatted_results

        except Exception as e:
            log.error(f'Error getting functions called by {function_name}: {e}')
            raise
        finally:
            session.close()

    def close(self):
        if hasattr(self, 'engine'):
            self.engine.dispose()
            log.info('Closed PostgreSQL connection')
