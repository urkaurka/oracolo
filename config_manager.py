import yaml
from pathlib import Path

class ConfigManager:
    def __init__(self, config_path: str = "../config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    @property
    def chroma_dir(self) -> str:
        return self.config['vector_store']['persist_directory']

    @property
    def pdf_resources_dir(self) -> str:
        return self.config['resources']['pdf_directory']