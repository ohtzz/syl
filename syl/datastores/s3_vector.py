import boto3
import json

from loguru import logger as log
from typing import Dict, List, Any, Optional
from botocore.exceptions import ClientError

from syl.datastores.datastore import DataStoreLoader, DataStoreQuery
from .utils import MetadataProcessor, ResultFormatter
from ..parsers.parser import FunctionEmbedding

DEFAULT_BATCH_SIZE = 100


class S3VectorLoader(DataStoreLoader):
    """Loads embeddings into S3 Vector bucket with optimized metadata"""

    def __init__(
        self,
        embed_model: str,
        num_workers: int,
        file_ext_whitelist: Optional[List[str]],
        file_ext_blacklist: Optional[List[str]],
        git_repo_url: Optional[str],
        git_branch: Optional[str],
        s3_bucket_name: str,
        vector_size: int,
        s3_index_name: str,
        aws_access_key_id: Optional[str],
        aws_secret_key: Optional[str],
        aws_profile: Optional[str] = None,
        region: Optional[str] = None,
    ):
        self.s3_bucket_name = s3_bucket_name
        self.s3_index_name = s3_index_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_key = aws_secret_key
        self.aws_profile = aws_profile
        self.region = region
        self.s3_vector_client = self._create_s3_vector_client()

        super().__init__(
            embed_model=embed_model,
            num_workers=num_workers,
            file_ext_whitelist=file_ext_whitelist,
            file_ext_blacklist=file_ext_blacklist,
            git_repo_url=git_repo_url,
            git_branch=git_branch,
            vector_size=vector_size,
        )

    def _create_s3_vector_client(self):
        try:
            # Mounted with profile
            if self.aws_profile is not None:
                boto3.setup_default_session(profile_name=self.aws_profile)
                return boto3.client('s3vectors', region_name=self.region)
            # Access keys
            elif self.aws_access_key_id is not None and self.aws_secret_key is not None:
                return boto3.client(
                    's3vectors',
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_key,
                    region_name=self.region,
                )
            # Default to mount default profile/configuration
            else:
                return boto3.client('s3vectors', region_name=self.region)

        except Exception as e:
            log.error(f'Failed to initialize S3 Vector client: {e}')
            raise

    def get_s3_vector_index(self):
        resp = self.s3_vector_client.get_index(vectorBucketName=self.s3_bucket_name, indexName=self.s3_index_name)
        return resp['index']

    def put_s3_vectors(self, vector_records: List[Dict[str, Any]]):
        """Upload vector records to S3 Vector bucket"""
        log.info(f'Uploading {len(vector_records)} vectors to S3 Vector bucket')

        total_uploaded = 0
        failed_uploads = []

        # Batch upload
        for i in range(0, len(vector_records), DEFAULT_BATCH_SIZE):
            batch = vector_records[i : i + DEFAULT_BATCH_SIZE]

            try:
                log.info(
                    f'Uploading batch {i // DEFAULT_BATCH_SIZE + 1}/{(len(vector_records) - 1) // DEFAULT_BATCH_SIZE + 1} ({len(batch)} vectors)...'
                )

                # Validate before pushing
                for record in batch:
                    self.validate_vector_record(record)

                # Upload
                _ = self.s3_vector_client.put_vectors(
                    vectorBucketName=self.s3_bucket_name, indexName=self.s3_index_name, vectors=batch
                )

                log.info(f'Successfully uploaded batch {i // DEFAULT_BATCH_SIZE + 1} ({len(batch)} vectors)')
                total_uploaded += len(batch)

            except ClientError as e:
                error_code = e.response['Error']['Code']
                error_message = e.response['Error']['Message']
                log.error(f'AWS error uploading batch starting at index {i}: {error_code} - {error_message}')
                failed_uploads.extend([record['key'] for record in batch])
                continue
            except Exception as e:
                log.error(f'Failed to upload batch starting at index {i}: {e}')
                failed_uploads.extend([record['key'] for record in batch])
                continue

        log.info(f'Upload complete: {total_uploaded}/{len(vector_records)} successful')

        if failed_uploads:
            log.warning(f'Failed to upload {len(failed_uploads)} vectors')
            log.warning(
                f'Failed vector keys: {failed_uploads[:10]}...'
                if len(failed_uploads) > 10
                else f'Failed vector keys: {failed_uploads}'
            )
            return False, failed_uploads

        return True, []

    def test_connection(self):
        try:
            self.get_s3_vector_index()
        except Exception as e:
            log.error(f'Failed to get S3 Vector bucket as connection test: {e}')

    def load_functions(self, funcs: List[FunctionEmbedding]) -> bool:
        """Load function embeddings into S3 Vector store"""
        try:
            if not funcs:
                log.error('No functions provided')
                return False

            # Convert to vector records
            log.info('Converting to S3 Vector records...')
            vector_records = []
            conversion_errors = []

            for i, func_data in enumerate(funcs):
                try:
                    vector_record = self.create_vector_record(func_data)
                    vector_records.append(vector_record)
                except Exception as e:
                    log.warning(f'Failed to convert function {i}: {e}')
                    conversion_errors.append(i)
                    continue

            if conversion_errors:
                log.warning(f'Failed to convert {len(conversion_errors)} functions')

            if not vector_records:
                log.error('No valid vector records created')
                return False

            log.info(f'Created {len(vector_records)} vector records')

            # Upload to S3 Vector
            success, failed_ids = self.put_s3_vectors(vector_records)

            if success:
                log.info('Successfully uploaded all vectors to S3 Vector bucket')
                return True
            else:
                log.error(f'Upload failed for {len(failed_ids)} vectors')
                return False

        except Exception as e:
            log.error(f'Process failed: {e}')
            return False

    def load_file_embeddings(self, file_data_with_embeddings):
        """S3Vector does not support file embeddings - only function embeddings"""
        log.warning('S3Vector datastore does not support file embeddings, skipping file embedding loading')

    def create_vector_record(self, function_data: FunctionEmbedding) -> Dict[str, Any]:
        """Create a vector record for S3 Vector bucket"""

        if not function_data.embedding:
            raise ValueError('No embedding found in function data')

        # Ensure floats
        embedding_float = [float(x) for x in function_data.embedding]

        raw_metadata = {
            'function_name': function_data.function_name,
            'file_path': function_data.file_path,
            'line_start': function_data.line_start,
            'line_end': function_data.line_end,
            'calls': function_data.calls,
            'called_by': function_data.called_by,
            'complexity': function_data.complexity,
            'maintainability_index': function_data.maintainability_index,
            'parameters': function_data.parameters,
            'returns': function_data.returns,
            'docstring': function_data.docstring,
            'is_async': function_data.is_async,
            'is_method': function_data.is_method,
            'class_name': function_data.class_name,
            'imports': function_data.imports,
        }
        if function_data.code:
            raw_metadata['code'] = function_data.code

        optimized_metadata = MetadataProcessor.optimize_for_s3_vector(raw_metadata)

        vector_record = {
            'key': function_data.id,
            'data': {'float32': embedding_float},
            'metadata': optimized_metadata,
        }

        return vector_record

    def validate_vector_record(self, record: Dict[str, Any]):
        """Validate vector record meets S3 Vector constraints"""

        # Check required fields
        if 'key' not in record or not record['key']:
            raise ValueError("Vector record missing required 'key' field")

        if 'data' not in record or 'float32' not in record['data']:
            raise ValueError("Vector record missing required 'data.float32' field")

        # Check embedding dimension consistency
        embedding = record['data']['float32']
        if not isinstance(embedding, list):
            raise ValueError("Vector 'data.float32' must be a list")

        # Validate float32 values
        for i, val in enumerate(embedding):
            if not isinstance(val, (int, float)):
                raise ValueError(f'Vector value at index {i} is not a number: {val}')
            if not (-3.4e38 <= val <= 3.4e38):  # float32 range
                raise ValueError(f'Vector value at index {i} out of float32 range: {val}')

        # Check metadata limits
        metadata = record.get('metadata', {})

        # Max 10 keys limit
        if len(metadata) > 10:
            raise ValueError(f'Metadata has {len(metadata)} keys, maximum is 10')

        # Max 2KB size limit
        metadata_json = json.dumps(metadata)
        metadata_size = len(metadata_json.encode('utf-8'))
        if metadata_size > 2048:
            raise ValueError(f'Metadata size is {metadata_size} bytes, maximum is 2048')

        log.debug(
            f'Validated record {record["key"]}: {len(embedding)} dims, {len(metadata)} metadata keys, {metadata_size} bytes'
        )


class S3VectorQuery(DataStoreQuery):
    """Query S3 Vector bucket for code embeddings"""

    def __init__(self, bucket_name: str, index_name: str, model_name: str, region: str = 'us-east-1'):
        super().__init__(model_name=model_name)

        self.bucket_name = bucket_name
        self.index_name = index_name
        self.region = region

        # Initialize S3 Vector client
        try:
            self.s3_vector_client = boto3.client('s3vectors', region_name=region)
            log.info(f'Connected to S3 Vector in region: {region}')
        except Exception as e:
            log.error(f'Failed to initialize S3 Vector client: {e}')
            raise

    def query_s3_vector_bucket(
        self, query_text: str, limit: int = 10, filter_dict: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Search vectors in S3 Vector bucket"""

        # Generate query embedding
        log.info(f"Generating embedding for query: '{query_text}'")
        query_embedding = self.generate_query_embedding(query_text)

        try:
            # Query S3 Vector
            log.info(f'Querying S3 Vector bucket: {self.bucket_name}/{self.index_name}')
            response = self.s3_vector_client.query_vectors(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
                topK=limit,
                queryVector={'float32': query_embedding},
                filter=filter_dict,
                returnMetadata=True,
                returnDistance=True,
            )

            return response

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            log.error(f'AWS error during query: {error_code} - {error_message}')
            raise
        except Exception as e:
            log.error(f'Failed to query vectors: {e}')
            raise

    def format_results(
        self, response: Dict[str, Any], query_text: str, include_function_content: bool = False
    ) -> Dict[str, Any]:
        return ResultFormatter.format_function_results(
            [
                {
                    'id': match.get('key', ''),
                    'distance': match.get('distance', 0.0),
                    'metadata': match.get('metadata', {}),
                    **match.get('metadata', {}),
                }
                for match in response.get('vectors', [])
            ],
            query_text,
            's3vector',
            include_function_content,
        )

    def query_semantic_functions(self, req, limit: int = 10) -> Dict[str, Any]:
        resp = self.query_s3_vector_bucket(req.query, req.max_results, req.to_s3_vector_filters())
        return self.format_results(resp, req.query, req.include_function_content)

    def query_semantic_files(self, req, limit: int = 10) -> Dict[str, Any]:
        """S3Vector does not support file semantic search"""
        raise NotImplementedError('File semantic search is not supported with S3Vector datastore')

    def query_filters(self, **filters) -> Dict[str, Any]:
        """S3Vector requires a query text, so filter-only queries are not supported"""
        raise NotImplementedError(
            'S3Vector (AWS) requires a query text for semantic search. Filter-only queries are not supported. Use the semantic search endpoint with filters instead.'
        )

    def get_file_content(self, file_path: str, start_line: int = None, end_line: int = None) -> bytes:
        """S3Vector does not support file content retrieval"""
        raise NotImplementedError('File content retrieval is not supported with S3Vector datastore')

    def get_function_content(self, function_id: str) -> str:
        """S3Vector does not support function content retrieval"""
        raise NotImplementedError('Function content retrieval is not supported with S3Vector datastore')

    def get_available_files(self) -> List[str]:
        """S3Vector does not support file listing"""
        raise NotImplementedError('File listing is not supported with S3Vector datastore')

    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """S3Vector does not support file metadata retrieval"""
        raise NotImplementedError('File metadata retrieval is not supported with S3Vector datastore')

    def get_function_callers(self, function_name: str) -> Dict[str, Any]:
        """S3Vector does not support call graph queries"""
        raise NotImplementedError('Function caller queries are not supported with S3Vector datastore')

    def get_function_called_by(self, function_name: str) -> Dict[str, Any]:
        """S3Vector does not support call graph queries"""
        raise NotImplementedError('Function called-by queries are not supported with S3Vector datastore')
