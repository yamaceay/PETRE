"""
Configuration management for PETRE with immutable dataclasses.

Provides centralized configuration with validation, canonicalization,
and device detection. Built once at application start and passed down.
"""

from __future__ import annotations

import os
import json
import ntpath
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch
import logging

try:
    from .import_utils import smart_import
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from import_utils import smart_import

# Import interfaces for type checking
interfaces_imports = smart_import('interfaces', ['ConfigurationError'])
ConfigurationError = interfaces_imports['ConfigurationError']


@dataclass(frozen=True)
class AppConfig:
    """
    Immutable configuration for PETRE application.

    Contains all static dependencies and configuration that should be
    built once at application start and passed down to components.
    """
    # Core paths
    output_base_folder_path: str
    data_file_path: str
    tri_pipeline_path: str

    # Column names
    individual_name_column: str
    original_text_column: str

    # Anonymization parameters
    ks: List[int]

    # Annotation configuration - can be file path OR generation method
    starting_anonymization_path: Optional[str] = None
    annotation_method: Optional[str] = None  # 'spacy_ner3', 'spacy_ner4', 'spacy_ner7', 'presidio', 'combined'
    annotation_confidence_threshold: float = 0.7
    annotation_entity_types: Optional[List[str]] = None

    # Optional configuration with defaults
    mask_text: str = ""
    use_mask_all_instances: bool = True
    explainability_mode: str = "SHAP"  # "SHAP" or "Greedy"
    use_chunking: bool = True

    # Derived configuration (computed)
    starting_annon_name: str = field(init=False)
    output_folder_path: str = field(init=False)
    device: torch.device = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization validation and derived value computation."""
        # Validate annotation configuration
        if not self.starting_anonymization_path and not self.annotation_method:
            raise ConfigurationError("Either starting_anonymization_path or annotation_method must be provided")

        if self.starting_anonymization_path and self.annotation_method:
            raise ConfigurationError("Provide either starting_anonymization_path OR annotation_method, not both")

        # Validate k values
        if not isinstance(self.ks, list) or len(self.ks) == 0:
            raise ConfigurationError("Setting 'ks' must be a non-empty list")

        if not all(isinstance(k, int) and k > 0 for k in self.ks):
            raise ConfigurationError("All values in 'ks' must be positive integers")

        # Validate explainability mode
        if self.explainability_mode not in ["SHAP", "Greedy"]:
            raise ConfigurationError(f"explainability_mode must be 'SHAP' or 'Greedy', got '{self.explainability_mode}'")

        # Validate file paths
        if not os.path.isfile(self.data_file_path):
            raise ConfigurationError(f"Data file not found: {self.data_file_path}")

        # Only validate annotation file if using file-based approach
        if self.starting_anonymization_path and not os.path.isfile(self.starting_anonymization_path):
            raise ConfigurationError(f"Starting anonymization file not found: {self.starting_anonymization_path}")

        if not os.path.isdir(self.tri_pipeline_path):
            raise ConfigurationError(f"TRI pipeline path not found: {self.tri_pipeline_path}")

        # Compute derived values using object.__setattr__ for frozen dataclass
        # Only compute starting_annon_name if we have a file path
        if self.starting_anonymization_path:
            object.__setattr__(self, 'starting_annon_name', self._compute_starting_annon_name())
        else:
            object.__setattr__(self, 'starting_annon_name', 'dynamic_annotations')

        object.__setattr__(self, 'output_folder_path', self._compute_output_folder_path())
        object.__setattr__(self, 'device', self._detect_device())

        # Sort ks in ascending order
        sorted_ks = sorted(self.ks)
        object.__setattr__(self, 'ks', sorted_ks)

        # Create output directory if it doesn't exist
        os.makedirs(self.output_folder_path, exist_ok=True)

    def _compute_starting_annon_name(self) -> str:
        """Extract filename without extension from starting anonymization path."""
        if not self.starting_anonymization_path:
            return 'dynamic_annotations'
        head, tail = ntpath.split(self.starting_anonymization_path)
        filename = tail or ntpath.basename(head)
        return os.path.splitext(filename)[0]

    def _compute_output_folder_path(self) -> str:
        """Compute output folder path using base path and anonymization name."""
        return os.path.join(self.output_base_folder_path, self.starting_annon_name)

    def _detect_device(self) -> torch.device:
        """Detect the best available device for computation."""
        if torch.cuda.is_available():
            # Set CUDA debugging environment variable
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            return torch.device("cuda:0")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Apple Silicon MPS support
            return torch.device("mps")
        else:
            return torch.device("cpu")

    @property
    def device_type(self) -> str:
        """Get device type as string."""
        return self.device.type

    @property
    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self.device.type in ['cuda', 'mps']


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Dictionary containing configuration values

    Raises:
        ConfigurationError: If file doesn't exist or is invalid
    """
    if not config_path.endswith(".json"):
        raise ConfigurationError(f"Configuration file must be JSON format: {config_path}")

    if not os.path.isfile(config_path):
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in configuration file: {e}") from e
    except Exception as e:
        raise ConfigurationError(f"Error reading configuration file: {e}") from e

    return config_data


def validate_config_data(config_data: Dict[str, Any]) -> None:
    """
    Validate required configuration fields.

    Args:
        config_data: Dictionary containing configuration

    Raises:
        ConfigurationError: If required fields are missing
    """
    # Always required fields
    required_fields = [
        "output_base_folder_path",
        "data_file_path",
        "individual_name_column",
        "original_text_column",
        "tri_pipeline_path",
        "ks"
    ]

    missing_fields = [field for field in required_fields if field not in config_data]
    if missing_fields:
        raise ConfigurationError(f"Missing required configuration fields: {missing_fields}")

    # Validate annotation source - either file-based OR method-based
    has_annotation_file = "starting_anonymization_path" in config_data
    has_annotation_method = "annotation_method" in config_data

    if not has_annotation_file and not has_annotation_method:
        raise ConfigurationError(
            "Must specify either 'starting_anonymization_path' (file-based) or "
            "'annotation_method' (dynamic generation)"
        )

    if has_annotation_file and has_annotation_method:
        raise ConfigurationError(
            "Cannot specify both 'starting_anonymization_path' and 'annotation_method'. "
            "Choose either file-based or dynamic annotation generation."
        )

    # Validate field types for present fields
    type_validations = {
        "output_base_folder_path": str,
        "data_file_path": str,
        "individual_name_column": str,
        "original_text_column": str,
        "starting_anonymization_path": str,
        "tri_pipeline_path": str,
        "ks": list,
        "annotation_method": str,
        "annotation_confidence_threshold": (int, float),
        "annotation_entity_types": list
    }

    for field, expected_type in type_validations.items():
        if field in config_data and not isinstance(config_data[field], expected_type):
            raise ConfigurationError(f"Field '{field}' must be of type {expected_type.__name__}")


def create_app_config(config_path: str, **overrides) -> AppConfig:
    """
    Create AppConfig from JSON file with optional overrides.

    Args:
        config_path: Path to JSON configuration file
        **overrides: Optional configuration overrides

    Returns:
        Configured AppConfig instance

    Raises:
        ConfigurationError: If configuration is invalid
    """
    # Load from file
    config_data = load_config_from_file(config_path)

    # Apply overrides
    config_data.update(overrides)

    # Validate
    validate_config_data(config_data)

    # Extract only known fields for AppConfig
    config_fields = {
        field.name for field in AppConfig.__dataclass_fields__.values()
        if field.init  # Only include fields that are part of __init__
    }

    filtered_config = {
        key: value for key, value in config_data.items()
        if key in config_fields
    }

    # Log any unknown fields
    unknown_fields = set(config_data.keys()) - config_fields
    if unknown_fields:
        logging.warning(f"Unknown configuration fields ignored: {unknown_fields}")

    return AppConfig(**filtered_config)


def canonicalize_config_value(value: Any, default: Any) -> Any:
    """
    Canonicalize configuration value with default fallback.

    Args:
        value: Value to canonicalize (may be None)
        default: Default value to use if value is None

    Returns:
        Canonicalized value
    """
    return default if value is None else value


# Convenience functions for common configuration patterns
def get_config_with_defaults(**kwargs) -> Dict[str, Any]:
    """Get configuration dictionary with standard defaults applied."""
    defaults = {
        'mask_text': "",
        'use_mask_all_instances': True,
        'explainability_mode': "SHAP",
        'use_chunking': True
    }

    # Apply defaults for missing keys
    for key, default_value in defaults.items():
        if key not in kwargs:
            kwargs[key] = default_value

    return kwargs


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.

    Args:
        base_config: Base configuration
        override_config: Override configuration (takes precedence)

    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    merged.update(override_config)
    return merged


__all__ = [
    'AppConfig',
    'ConfigurationError',
    'load_config_from_file',
    'validate_config_data',
    'create_app_config',
    'canonicalize_config_value',
    'get_config_with_defaults',
    'merge_configs'
]