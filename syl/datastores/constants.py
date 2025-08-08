CODE_PATH = '/code'

PG_FUNC_TABLE_NAME = 'func_embeddings'
PG_FILE_TABLE_NAME = 'file_embeddings'

CODE_EXTENSIONS = {
    '.py',
    '.go',
    '.js',
    '.ts',
    '.jsx',
    '.tsx',
    '.java',
    '.c',
    '.cpp',
    '.h',
    '.hpp',
    '.cs',
    '.php',
    '.rb',
    '.rs',
    '.kt',
    '.swift',
    '.scala',
    '.clj',
    '.r',
    '.m',
    '.sql',
}
DOCS_EXTENSIONS = {'.md', '.txt', '.rst', '.asciidoc', '.adoc', '.org', '.tex', '.rtf'}
CONFIG_EXTENSIONS = {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.xml', '.properties'}
DATA_EXTENSIONS = {'.csv', '.tsv', '.log'}

SUPPORTED_FILETYPES = CODE_EXTENSIONS | DOCS_EXTENSIONS | CONFIG_EXTENSIONS | DATA_EXTENSIONS

MAX_DOC_SIZE = 1024 * 1024  # 1MB for documentation files
MAX_CONFIG_SIZE = 512 * 1024  # 512KB for config files
MAX_DATA_SIZE = 2 * 1024 * 1024  # 2MB for data files
