from pydantic import BaseModel


class BrainConfig(BaseModel):
    """
    Configuration for BrainAgent:
      - directory: root folder of your corpus
      - brain_name:  key for registry lookup
      - chunk_size:  lines per chunk (-1 = whole file)
    """
    directory: str
    brain_name: str
    chunk_size: int = -1
