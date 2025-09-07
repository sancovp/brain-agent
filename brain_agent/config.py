from pydantic import BaseModel
from typing import Literal


class BrainConfig(BaseModel):
    """
    Configuration for BrainAgent:
      - brain_name: key for registry lookup
      - neuron_source_type: how to load neurons ("registry_keys", "entire_registry", "directory", "file")
      - neuron_source: registry name, directory path, or file path
      - chunk_max: max characters per chunk (for file chunking)
    """
    brain_name: str = None
    neuron_source_type: Literal["registry_keys", "entire_registry", "directory", "file"] = "directory"
    neuron_source: str = None
    chunk_max: int = 30000
    
    # Backwards compatibility fields
    directory: str = None
    chunk_size: int = -1
    
    def model_post_init(self, __context):
        """Handle backwards compatibility and validation."""
        # Handle old directory-based configs
        if self.directory and not self.neuron_source:
            self.neuron_source_type = "directory"
            self.neuron_source = self.directory
            if self.chunk_size != -1:
                self.chunk_max = self.chunk_size
        
        # Ensure neuron_source is set
        if not self.neuron_source and self.directory:
            self.neuron_source = self.directory
