try:
    import chromadb
except Exception as e:
    __import__('pysqlite3')
    import sys

    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print(e)

from . import parsing
from . import pipelines