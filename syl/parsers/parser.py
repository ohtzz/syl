import gzip
import base64
import json
import hashlib

from loguru import logger as log
from threading import Lock
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FunctionEmbedding:
    id: str
    embedding: List[float]
    code: str
    function_name: str
    file_path: str
    line_start: int
    line_end: int
    calls: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)
    complexity: int = 0
    maintainability_index: float = 0.0
    parameters: List[str] = field(default_factory=list)
    returns: Optional[str] = None
    docstring: Optional[str] = None
    is_async: bool = False
    is_method: bool = False
    class_name: Optional[str] = None
    imports: List[str] = field(default_factory=list)


@dataclass
class FileEmbedding:
    file_path: str
    filename: str
    content: str  # Full file content
    content_compressed: Optional[str] = None  # b64 gzip
    imports: List[str] = None
    file_size: int = 0
    line_count: int = 0
    file_type: str = 'other'  # code, docs, config, data, other
    embedding: Optional[List[float]] = None
    id: Optional[str] = None

    def compress_content(self):
        """Compress to gzip and base64 encode"""
        if self.content:
            compressed = gzip.compress(self.content.encode('utf-8'))
            self.content_compressed = base64.b64encode(compressed).decode('ascii')

    def decompress_content(self) -> str:
        """Decompress content from base64 gzipped format"""
        if self.content_compressed:
            compressed = base64.b64decode(self.content_compressed.encode('ascii'))
            return gzip.decompress(compressed).decode('utf-8')
        return self.content or ''

    def has_embedding(self) -> bool:
        """Check if this file data has an embedding"""
        return self.embedding is not None and len(self.embedding) > 0


class CodeParser:
    def __init__(self, model_name: str, vector_size: int, embed_model_instance=None):
        self.model_name = model_name
        self.vector_size = vector_size
        self._embed_model = embed_model_instance
        self._model_lock = Lock()

    @property
    def embed_model(self):
        if self._embed_model is not None:
            return self._embed_model

        with self._model_lock:
            # Janky double check as we blast threads at this at the start..
            if self._embed_model is not None:
                return self._embed_model

            try:
                log.info(f'Loading embedding model: {self.model_name}')
                self._embed_model = SentenceTransformer(self.model_name)

                # Validate that the model's vector size matches expected size
                model_dimension = self._embed_model.get_sentence_embedding_dimension()
                if model_dimension != self.vector_size:
                    log.error(
                        f"Model dimension ({model_dimension}) doesn't match expected vector_size ({self.vector_size}). Please use --vector-size {model_dimension} or choose a different model."
                    )
                    raise ValueError(
                        f'Model produces {model_dimension}-dimensional embeddings but {self.vector_size} was requested'
                    )
                else:
                    log.info(f'Model dimension matches expected vector size: {self.vector_size}')

            except Exception as e:
                log.error(f'Failed to load model {self.model_name}: {e}')
                raise SystemExit(1)

        return self._embed_model

    def process_file(self, file_path: str) -> List[FunctionEmbedding]:
        raise NotImplementedError('Subclasses must implement this method')

    @staticmethod
    def generate_function_id(metadata: FunctionEmbedding) -> str:
        """Generate unique ID for function"""
        unique_string = f'{metadata.file_path}:{metadata.function_name}:{metadata.line_start}'
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]

    def save_to_json(self, embeddings: List[FunctionEmbedding], output_path: str):
        """Save embeddings and metadata to JSON file"""
        log.info(f'Saving to {output_path}')

        # Convert to serializable format
        data = {
            'metadata': {
                'total_functions': len(embeddings),
                'embedding_model': self.embed_model.get_sentence_embedding_dimension(),
                'embedding_dimension': len(embeddings[0].embedding) if embeddings else 0,
            },
            'functions': [],
        }

        for embedding in embeddings:
            func_data = {
                'id': embedding.id,
                'embedding': embedding.embedding,
                'code': embedding.code,
                'function_name': embedding.function_name,
                'file_path': embedding.file_path,
                'line_start': embedding.line_start,
                'line_end': embedding.line_end,
                'calls': embedding.calls,
                'called_by': embedding.called_by,
                'complexity': embedding.complexity,
                'maintainability_index': embedding.maintainability_index,
                'parameters': embedding.parameters,
                'returns': embedding.returns,
                'docstring': embedding.docstring,
                'is_async': embedding.is_async,
                'is_method': embedding.is_method,
                'class_name': embedding.class_name,
                'imports': embedding.imports,
            }
            data['functions'].append(func_data)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        log.info(f'Saved {len(embeddings)} function embeddings to {output_path}')
