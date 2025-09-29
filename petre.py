#!/usr/bin/env python3
"""
PETRE - Privacy-preserving Entities Transparency and Re-identification Evaluation

Backward compatibility wrapper that maintains the original API while using the 
new modular architecture internally. This allows existing code to continue 
working without modifications while benefiting from the improved architecture.

Usage:
    # Original API (maintained for backward compatibility)
    from petre import PETRE
    petre = PETRE(output_base_folder_path="outputs", ...)
    petre.run()
    
    # Or CLI usage
    python petre.py config.json
"""

import json
import logging
import os
import sys
from typing import List, Optional, Dict, Any

# Import the new modular architecture
try:
    # Import from package if installed
    from . import (
        AppConfig,
        create_app_config,
        create_petre_orchestrator,
        run_petre_from_config,
        run_petre_from_dict,
        PETREError,
        ConfigurationError
    )
except ImportError:
    # Import from local files if running as script
    from import_utils import smart_import_single
    
    AppConfig = smart_import_single('config', 'AppConfig')
    create_app_config = smart_import_single('config', 'create_app_config')
    create_petre_orchestrator = smart_import_single('main', 'create_petre_orchestrator')
    run_petre_from_config = smart_import_single('main', 'run_petre_from_config')
    run_petre_from_dict = smart_import_single('main', 'run_petre_from_dict')
    PETREError = smart_import_single('interfaces', 'PETREError')
    ConfigurationError = smart_import_single('interfaces', 'ConfigurationError')


class PETRE:
    """
    Backward compatibility wrapper for the original PETRE API.
    
    This class maintains the original API while internally using the new 
    modular architecture. All original functionality is preserved.
    
    Example:
        >>> petre = PETRE(
        ...     output_base_folder_path="outputs",
        ...     data_file_path="data.json",
        ...     individual_name_column="name",
        ...     original_text_column="text",
        ...     starting_anonymization_path="annotations.json",
        ...     tri_pipeline_path="./models/tri",
        ...     ks=[2, 3, 5]
        ... )
        >>> petre.run()
    """
    
    def __init__(
        self,
        output_base_folder_path: str,
        data_file_path: str,
        individual_name_column: str,
        original_text_column: str,
        starting_anonymization_path: str,
        tri_pipeline_path: str,
        ks: List[int],
        mask_text: str = "",
        use_mask_all_instances: bool = True,
        explainability_mode: str = "SHAP",
        use_chunking: bool = True
    ):
        """
        Initialize PETRE with the original API parameters.
        
        Args:
            output_base_folder_path: Base folder for output results
            data_file_path: Path to dataset file
            individual_name_column: Column name for individual names
            original_text_column: Column name for original text
            starting_anonymization_path: Path to initial annotations
            tri_pipeline_path: Path to TRI model pipeline
            ks: List of k values for k-anonymity
            mask_text: Text to use for masking (default: "")
            use_mask_all_instances: Whether to mask all instances (default: True)
            explainability_mode: Explanation method - "SHAP" or "Greedy" (default: "SHAP")
            use_chunking: Whether to use chunking (default: True)
        """
        self._config_params = {
            'output_base_folder_path': output_base_folder_path,
            'data_file_path': data_file_path,
            'individual_name_column': individual_name_column,
            'original_text_column': original_text_column,
            'starting_anonymization_path': starting_anonymization_path,
            'tri_pipeline_path': tri_pipeline_path,
            'ks': ks,
            'mask_text': mask_text,
            'use_mask_all_instances': use_mask_all_instances,
            'explainability_mode': explainability_mode,
            'use_chunking': use_chunking
        }
        
        self._config = None
        self._orchestrator = None
        self._is_initialized = False
        
    def set_configs(self, **kwargs) -> None:
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
            
        Example:
            >>> petre.set_configs(ks=[2, 3, 5, 7], mask_text="[REDACTED]")
        """
        self._config_params.update(kwargs)
        # Reset initialization state when config changes
        self._config = None
        self._orchestrator = None
        self._is_initialized = False
        
    def _ensure_initialized(self) -> None:
        """Ensure the orchestrator is initialized with current configuration."""
        if not self._is_initialized:
            try:
                # Create configuration from current parameters
                self._config = AppConfig(**self._config_params)
                
                # Create orchestrator using new architecture
                self._orchestrator = create_petre_orchestrator(self._config)
                
                # Initialize the orchestrator
                self._orchestrator.initialize()
                
                self._is_initialized = True
                
            except Exception as e:
                raise PETREError(f"Failed to initialize PETRE: {e}") from e
    
    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute PETRE evaluation with current configuration.
        
        Args:
            verbose: Whether to enable verbose logging (default: True)
            
        Returns:
            Dictionary containing evaluation results
            
        Raises:
            PETREError: If execution fails
            ValidationError: If configuration is invalid
            
        Example:
            >>> results = petre.run(verbose=True)
        """
        # Configure logging based on verbose setting
        log_level = logging.INFO if verbose else logging.WARNING
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        try:
            # Ensure we're initialized
            self._ensure_initialized()
            
            # Run incremental execution with the new architecture
            if verbose:
                logging.info("Starting PETRE evaluation...")
                logging.info(f"Configuration: {self._config}")
            
            results = self._orchestrator.run_incremental_execution(self._config.ks)
            
            if verbose:
                logging.info("PETRE evaluation completed successfully")
            
            return results
            
        except Exception as e:
            error_msg = f"PETRE execution failed: {e}"
            logging.error(error_msg)
            raise PETREError(error_msg) from e
    
    @property
    def config(self):
        """Get the current configuration object."""
        return self._config
    
    def validate_configuration(self) -> bool:
        """
        Validate the current configuration without running execution.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # Create config to trigger validation
            config = AppConfig(**self._config_params)
            return True
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}") from e


def main():
    """
    Command-line interface entry point.
    
    Maintains backward compatibility with the original CLI interface.
    
    Usage:
        python petre.py config.json
    """
    if len(sys.argv) != 2:
        print("Usage: python petre.py <config_file_path>")
        sys.exit(1)
    
    config_file_path = sys.argv[1]
    
    # Validate config file exists
    if not os.path.exists(config_file_path):
        print(f"Error: Configuration file '{config_file_path}' not found")
        sys.exit(1)
    
    try:
        # Load configuration from file
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Create PETRE instance with loaded configuration
        petre = PETRE(**config_data)
        
        # Validate configuration
        petre.validate_configuration()
        
        # Run evaluation
        results = petre.run(verbose=True)
        
        print("PETRE evaluation completed successfully")
        print(f"Results written to: {petre.config.output_base_folder_path}")
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)
    except ConfigurationError as e:
        print(f"Error: Configuration validation failed: {e}")
        sys.exit(1)
    except PETREError as e:
        print(f"Error: PETRE execution failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Unexpected error: {e}")
        sys.exit(1)


# Alternative entry points for different use cases
def run_petre_cli(config_file_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Run PETRE from configuration file (programmatic CLI interface).
    
    Args:
        config_file_path: Path to JSON configuration file
        verbose: Whether to enable verbose logging
        
    Returns:
        Dictionary containing evaluation results
        
    Example:
        >>> results = run_petre_cli('config.json', verbose=True)
    """
    # Use the new architecture directly for better error handling
    return run_petre_from_config(config_file_path)


def create_petre_from_dict(config_dict: Dict[str, Any]) -> PETRE:
    """
    Create PETRE instance from configuration dictionary.
    
    Args:
        config_dict: Configuration parameters as dictionary
        
    Returns:
        PETRE instance
        
    Example:
        >>> config = {"output_base_folder_path": "outputs", ...}
        >>> petre = create_petre_from_dict(config)
    """
    return PETRE(**config_dict)


def validate_petre_config(config_file_path: str) -> bool:
    """
    Validate PETRE configuration file without running execution.
    
    Args:
        config_file_path: Path to JSON configuration file
        
    Returns:
        True if configuration is valid
        
        Raises:
        ConfigurationError: If configuration is invalid    Example:
        >>> is_valid = validate_petre_config('config.json')
    """
    try:
        config = create_app_config(config_file_path)
        return True
    except Exception as e:
        raise ConfigurationError(f"Configuration validation failed: {e}") from e


# Maintain compatibility with original module-level functions
def run_from_config(config_path: str, **overrides) -> Dict[str, Any]:
    """Legacy function for backward compatibility."""
    if overrides:
        return run_petre_from_dict({**create_app_config(config_path).__dict__, **overrides})
    return run_petre_from_config(config_path)


if __name__ == "__main__":
    main()
