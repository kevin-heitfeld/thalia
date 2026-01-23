"""
Configuration file loader for literature management.

Supports YAML configuration files for queries and collections.
"""

from pathlib import Path
from typing import Dict, List, Optional

import yaml


class ConfigLoader:
    """
    Load and validate configuration files for literature management.

    Examples
    --------
    >>> config = ConfigLoader()
    >>> query_config = config.load_query("vertex_operators")
    >>> print(query_config["name"])
    'Vertex Operator Normalizations'
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize config loader.

        Parameters
        ----------
        config_dir : str, optional
            Path to config directory (default: project_root/config)
        """
        if config_dir is None:
            # Default to project root / config
            self.config_dir = Path(__file__).parent.parent.parent.parent / "config"
        else:
            self.config_dir = Path(config_dir)

        self.queries_dir = self.config_dir / "queries"
        self.collections_dir = self.config_dir / "collections"

    def load_query(self, name: str) -> Dict:
        """
        Load query configuration file.

        Parameters
        ----------
        name : str
            Query config name (without .yaml extension)

        Returns
        -------
        dict
            Query configuration
        """
        path = self.queries_dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Query config not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return config

    def load_collection(self, name: str) -> Dict:
        """
        Load collection configuration file.

        Parameters
        ----------
        name : str
            Collection config name (without .yaml extension)

        Returns
        -------
        dict
            Collection configuration
        """
        path = self.collections_dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Collection config not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return config

    def list_queries(self) -> List[str]:
        """List available query configurations."""
        if not self.queries_dir.exists():
            return []
        return [f.stem for f in self.queries_dir.glob("*.yaml")]

    def list_collections(self) -> List[str]:
        """List available collection configurations."""
        if not self.collections_dir.exists():
            return []
        return [f.stem for f in self.collections_dir.glob("*.yaml")]
