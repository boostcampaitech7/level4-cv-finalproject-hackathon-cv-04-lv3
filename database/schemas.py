from pydantic import BaseModel

class NewsDocument(BaseModel):
    page_content: str
    metadata: dict = None