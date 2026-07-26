"""
src/rag: Retrieval-Augmented Generation module for FIA regulation lookup.

Public interface:
    RagRetriever    - initialise once; call .query() at inference time
    RegulationChunk - dataclass returned by every query
    query_rag_tool  - LangChain @tool wrapper around RagRetriever singleton
"""
