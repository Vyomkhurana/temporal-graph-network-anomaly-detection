"""
Configuration loader utility for the TGN project.

Loads and validates YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """
    Configuration manager for the project.
    
    Loads YAML config files and provides easy access to settings.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config if config else {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key using dot notation.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'system.device')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            >>> config = Config('configs/default.yaml')
            >>> device = config.get('system.device')  # Returns 'cuda'
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key using dot notation.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            output_path: Path to save config. If None, overwrites original file.
        """
        save_path = Path(output_path) if output_path else self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        print(f"Configuration saved to: {save_path}")
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(path='{self.config_path}', keys={list(self.config.keys())})"


def load_config(config_path: str = "configs/default.yaml") -> Config:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    return Config(config_path)
