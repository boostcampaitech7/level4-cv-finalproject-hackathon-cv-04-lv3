from pydantic import BaseModel, Field
from typing import List

class NewsDocument(BaseModel):
    page_content: str
    metadata: dict = None

class ScriptItem(BaseModel):
    start: int
    end: int
    text: str

class SimilaritySchema(BaseModel):
    query: List[ScriptItem]
    k: int = Field(default=4, ge=1, description="Number of documents to retrieve")
    max_token: int = Field(default=3000, ge=1, description="Maximum number of tokens for LLM")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Temperature for LLM")
    chain_type: str = Field(default="stuff", description="Chain type for RetrievalQA")

class SimilarityThresholdSchema(BaseModel):
    query: List[ScriptItem]
    k: int = Field(default=4, ge=1, description="Number of documents to retrieve")
    max_token: int = Field(default=3000, ge=1, description="Maximum number of tokens for LLM")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Temperature for LLM")
    chain_type: str = Field(default="stuff", description="Chain type for RetrievalQA")
    score_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold for document retrieval"
    )
    search_type: str = Field(
        default="similarity_score_threshold",
        description="Search type for retriever"
    )

class MMRSchema(BaseModel):
    query: List[ScriptItem]
    k: int = Field(default=4, ge=1, description="Final number of documents to retrieve")
    fetch_k: int = Field(default=20, ge=1, description="Number of documents to fetch before filtering")
    lambda_mult: float = Field(
        default=0.5, 
        ge=0.0, 
        le=1.0, 
        description="Lambda multiplier for diversity (0 = max diversity, 1 = max relevance)"
    )
    max_token: int = Field(default=3000, ge=1, description="Maximum number of tokens for LLM")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Temperature for LLM")
    chain_type: str = Field(default="stuff", description="Chain type for RetrievalQA")
