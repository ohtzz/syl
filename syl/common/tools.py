from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
from collections.abc import Callable
from sqlalchemy import and_
from sqlalchemy.orm import Query


@dataclass
class S3FilterMapping:
    field_name: str
    operator: str
    transform: Optional[Callable] = None


class SQLOperator(str, Enum):
    EQ = '='
    GT = '>'
    GTE = '>='
    LT = '<'
    LTE = '<='
    NE = '!='
    LIKE = 'like'
    ILIKE = 'ilike'
    IN = 'in'
    NOT_IN = "not_in"
    ANY = '= any'
    CONTAINS = '@>'
    CONTAINED_BY = '<@'
    JSONB_EXISTS = '?'
    JSONB_PATH_EXISTS = '@?'


class SQLAlchemyFilter:
    """Wrapper for building SQLAlchemy filters using column objects directly"""

    def __init__(self):
        self.filters: List[Any] = []

    def add_filter(self, column, operator: SQLOperator, value: Any):
        """Add a filter condition using SQLAlchemy column objects"""
        if value is None or value == '':
            return

        if operator == SQLOperator.EQ:
            condition = column == value
        elif operator == SQLOperator.NE:
            condition = column != value
        elif operator == SQLOperator.GT:
            condition = column > value
        elif operator == SQLOperator.GTE:
            condition = column >= value
        elif operator == SQLOperator.LT:
            condition = column < value
        elif operator == SQLOperator.LTE:
            condition = column <= value
        elif operator == SQLOperator.LIKE:
            condition = column.like(value)
        elif operator == SQLOperator.ILIKE:
            condition = column.ilike(value)
        elif operator == SQLOperator.IN:
            condition = column.in_(value)
        elif operator == SQLOperator.NOT_IN:
            condition = ~column.in_(value)
        else:
            raise ValueError(f"Unsupported operator: {operator}")

        self.filters.append(condition)

    def apply_to_query(self, query: Query) -> Query:
        """Apply all filters to a SQLAlchemy query"""
        if not self.filters:
            return query

        return query.filter(and_(*self.filters))

    def get_filters(self) -> List[Any]:
        """Get the raw filter conditions"""
        return self.filters


class SearchSemanticFunctionsRequest(BaseModel):
    """Search for functions using semantic similarity"""

    datastore_name: str = Field(description='Datastore name to route the request to')
    query: str = Field(description="Search query describing the code functionality you're looking for")
    max_results: int = Field(default=10, description='Maximum number of results to return (default: 10, max: 30)')
    file_filter: Optional[str] = Field(default=None, description="Filter to specific filename (e.g. 'auth.py')")
    function_filter: Optional[str] = Field(default=None, description='Filter to specific function name')

    complexity_filter: Optional[int] = Field(
        default=None, description='Filter to functions with at least this complexity score'
    )
    is_async_filter: Optional[bool] = Field(
        default=None, description='Filter to async functions (true) or non-async (false)'
    )
    is_method_filter: Optional[bool] = Field(
        default=None, description='Filter to class methods (true) or standalone functions (false)'
    )
    include_function_content: bool = Field(
        default=False, description='Include function source code in results (increases response size)'
    )

    _S3_FILTER_MAPPINGS = {
        'file_filter': S3FilterMapping('filename', '$eq'),
        'function_filter': S3FilterMapping('function_name', '$eq'),
        'complexity_filter': S3FilterMapping('complexity', '$gte'),
        'is_async_filter': S3FilterMapping('is_async', '$eq'),
        'is_method_filter': S3FilterMapping('is_method', '$eq'),
    }

    def to_sqlalchemy_filter(self, table_class) -> SQLAlchemyFilter:
        """Convert request filters to SQLAlchemy filter using table column objects"""
        sql_filter = SQLAlchemyFilter()

        # Map request fields to table columns and operators
        filter_mappings = [
            ('file_filter', table_class.filename, SQLOperator.EQ),
            ('function_filter', table_class.function_name, SQLOperator.EQ),
            ('complexity_filter', table_class.complexity, SQLOperator.GTE),
            ('is_async_filter', table_class.is_async, SQLOperator.EQ),
            ('is_method_filter', table_class.is_method, SQLOperator.EQ),
        ]

        for field_name, column, operator in filter_mappings:
            value = getattr(self, field_name, None)
            if value is not None and value != '':
                sql_filter.add_filter(column, operator, value)

        return sql_filter

    def to_s3_vector_filters(self):
        filter_conditions = []

        for field_name, mapping in self._S3_FILTER_MAPPINGS.items():
            value = getattr(self, field_name, None)

            if value is None or value == '':
                continue

            if mapping.transform:
                value = mapping.transform(value)

            condition = {mapping.field_name: {mapping.operator: value}}
            filter_conditions.append(condition)

        if not filter_conditions:
            return None

        if len(filter_conditions) == 1:
            return filter_conditions[0]
        else:
            return {'$and': filter_conditions}

    def to_chromadb_filters(self):
        ...


class SearchSemanticFilesRequest(BaseModel):
    """Search for files using semantic similarity"""

    datastore_name: str = Field(description='Datastore name to route the request to')
    query: str = Field(description="Search query describing the file content or purpose you're looking for")
    max_results: int = Field(default=10, description='Maximum number of results to return (default: 10, max: 30)')
    file_type_filter: Optional[str] = Field(
        default=None, description="Filter to specific file type (e.g. 'code', 'docs', 'config', 'data')"
    )
    filename_filter: Optional[str] = Field(
        default=None, description="Filter to specific filename pattern (e.g. 'auth.py')"
    )
    include_file_content: bool = Field(
        default=False, description='Include full file content in results (increases response size significantly)'
    )

    _S3_FILTER_MAPPINGS = {
        'file_type_filter': S3FilterMapping('file_type', '$eq'),
        'filename_filter': S3FilterMapping('filename', '$eq'),
    }

    def to_sqlalchemy_filter(self, table_class) -> SQLAlchemyFilter:
        """Convert request filters to SQLAlchemy filter using table column objects"""
        sql_filter = SQLAlchemyFilter()

        filter_mappings = [
            ('file_type_filter', table_class.file_type, SQLOperator.EQ),
            ('filename_filter', table_class.filename, SQLOperator.EQ),
        ]

        for field_name, column, operator in filter_mappings:
            value = getattr(self, field_name, None)
            if value is not None and value != '':
                sql_filter.add_filter(column, operator, value)

        return sql_filter

    def to_s3_vector_filters(self) -> Optional[Dict]:
        ...  # No files for S3 yet...


class SearchByFunctionNameRequest(BaseModel):
    datastore_name: str = Field(description='Datastore name to route the request to')
    function_name: str = Field(description='Exact function name to search for')
    max_results: int = Field(default=10, description='Maximum number of results to return (default: 10)')


class GetFunctionCallersRequest(BaseModel):
    datastore_name: str = Field(description='Datastore name to route the request to')
    function_name: str = Field(description='Function name to find callers for')


class GetFunctionCalledByRequest(BaseModel):
    datastore_name: str = Field(description='Datastore name to route the request to')
    function_name: str = Field(description='Function name to find called functions for')


class GetComplexFunctionsRequest(BaseModel):
    datastore_name: str = Field(description='Datastore name to route the request to')
    min_complexity: int = Field(default=5, description='Minimum complexity score (default: 5)')
    max_results: int = Field(default=20, description='Maximum results to return (default: 20)')


class GetAsyncFunctionsRequest(BaseModel):
    datastore_name: str = Field(description='Datastore name to route the request to')
    max_results: int = Field(default=10, description='Maximum number of results to return (default: 10)')


class GetFunctionsInFileRequest(BaseModel):
    datastore_name: str = Field(description='Datastore name to route the request to')
    filename: str = Field(description="Filename to search in (e.g. 'auth.py')")
    max_results: int = Field(default=10, description='Maximum number of results to return (default: 10)')


class GetFileContentRequest(BaseModel):
    datastore_name: str = Field(description='Datastore name to route the request to')
    file_path: str = Field(description="Get the full content of a file or content between line numbers")
    start_line: Optional[int] = Field(default=None, description='Starting line number (1-based, optional)', ge=1)
    end_line: Optional[int] = Field(default=None, description='Ending line number (1-based, optional)', ge=1)


class GetFilesByNameRequest(BaseModel):
    datastore_name: str = Field(description='Datastore name to route the request to')
    filename: str = Field(description="The name of the file to retrieve (e.g. 'auth.py')")
    max_results: int = Field(default=5, description='Maximum number of results to return (default: 5)')


class ToolResponse(BaseModel):
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


TOOL_DEFINITIONS: Dict[str, Any] = {
    'search_function_code_semantic': {
        'description': 'Search codebase using semantic similarity. Finds functions based on meaning.',
        'model': SearchSemanticFunctionsRequest,
    },
    'search_file_code_semantic': {
        'description': 'Search files using semantic similarity. Finds files based on content or purpose.',
        'model': SearchSemanticFilesRequest,
    },
    'search_by_function_name': {
        'description': 'Find a specific function by its exact name',
        'model': SearchByFunctionNameRequest,
    },
    'get_function_callers': {
        'description': 'Find all functions that call a specific function (call graph analysis)',
        'model': GetFunctionCallersRequest,
    },
    'get_function_called_by': {
        'description': 'Find all functions called by a specific function (reverse call graph)',
        'model': GetFunctionCalledByRequest,
    },
    'get_complex_functions': {
        'description': 'Find functions above a certain complexity threshold',
        'model': GetComplexFunctionsRequest,
    },
    'get_async_functions': {
        'description': 'Find all async/await functions in the codebase',
        'model': GetAsyncFunctionsRequest,
    },
    'get_functions_in_file': {
        'description': 'Find all functions defined in a specific file',
        'model': GetFunctionsInFileRequest,
    },
    'get_file_content': {
        'description': 'Retrieve the contents of a file by name, with optional start/end line numbers',
        'model': GetFileContentRequest,
    },
    'get_files_by_name': {
        'description': 'Retrieve files by name',
        'model': GetFilesByNameRequest,
    },
}
