# My notes: vector databases

Chroma is an open-source embedding database that runs locally and is easy to start with in Python. Good for prototypes and small projects.

FAISS is a library from Meta for fast similarity search over dense vectors. It is not a full database: no built-in metadata filtering or persistence server.

Pinecone is a fully managed, hosted vector database. You pay for it, but there is no infrastructure to run.

pgvector adds vector similarity search to PostgreSQL, so you can keep embeddings next to your normal relational data.

Choosing: for a local student project, Chroma or FAISS; for production with existing Postgres, pgvector; for zero-ops at scale, a managed service.
