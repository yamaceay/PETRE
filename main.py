"""
Main application orchestration for PETRE.

Provides dual execution support (script + module) with smart import pattern,
error handling, and clean dependency injection.
"""

from __future__ import annotations

import sys
import os
import logging
from typing import Dict, Any, Optional

# Smart import pattern for dual execution support
try:
    from .import_utils import smart_import
except ImportError:
    # Direct script execution
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from import_utils import smart_import

# Import all dependencies using smart import
cli_imports = smart_import('cli', [
    'parse_arguments', 'validate_arguments', 'configure_logging',
    'create_configuration', 'print_configuration_summary', 
    'print_results_summary', 'handle_error'
])
config_imports = smart_import('config', ['AppConfig'])
interfaces_imports = smart_import('interfaces', ['PETREError'])
core_imports = smart_import('core', ['PETREComponentFactory'])

# Extract imports for clean usage
parse_arguments = cli_imports['parse_arguments']
validate_arguments = cli_imports['validate_arguments']
configure_logging = cli_imports['configure_logging']
create_configuration = cli_imports['create_configuration']
print_configuration_summary = cli_imports['print_configuration_summary']
print_results_summary = cli_imports['print_results_summary']
handle_error = cli_imports['handle_error']

AppConfig = config_imports['AppConfig']
PETREError = interfaces_imports['PETREError']
PETREComponentFactory = core_imports['PETREComponentFactory']


def create_petre_orchestrator(config: AppConfig):
    """
    Create and configure PETRE orchestrator with all dependencies.
    
    Args:
        config: Application configuration
        
    Returns:
        Configured PETRE orchestrator
    """
    factory = PETREComponentFactory()
    orchestrator = factory.create_orchestrator(config)
    return orchestrator


def run_petre_workflow(config: AppConfig, dry_run: bool = False) -> Dict[str, Any]:
    """
    Run the complete PETRE workflow.
    
    Args:
        config: Application configuration
        dry_run: If True, only validate setup without running anonymization
        
    Returns:
        Dictionary containing results
        
    Raises:
        PETREError: If workflow execution fails
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting PETRE workflow")
        logger.info("Configuration: %s", config.starting_annon_name)
        logger.info("Device: %s", config.device)
        logger.info("K values: %s", config.ks)
        
        # Create orchestrator
        orchestrator = create_petre_orchestrator(config)
        
        if dry_run:
            logger.info("Dry run mode - validating setup only")
            orchestrator.initialize()
            logger.info("Setup validation completed successfully")
            return {"status": "validated", "message": "Configuration and setup are valid"}
        
        # Initialize components
        logger.info("Initializing PETRE components")
        orchestrator.initialize()
        
        # Run incremental execution for all k values
        logger.info("Running incremental anonymization")
        orchestrator.run_incremental_execution(config.ks)
        
        logger.info("PETRE workflow completed successfully")
        
        # Placeholder results - would be populated by actual implementation
        results = {
            "status": "completed",
            "k_values": config.ks,
            "output_directory": config.output_folder_path,
            "starting_anonymization": config.starting_annon_name
        }
        
        return results
        
    except Exception as e:
        logger.error("PETRE workflow failed: %s", e)
        if isinstance(e, PETREError):
            raise
        else:
            raise PETREError(f"Workflow execution failed: {e}") from e


def main(args: Optional[list] = None) -> int:
    """
    Main application entry point supporting both execution methods.
    
    Args:
        args: Optional command-line arguments (defaults to sys.argv)
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Parse and validate arguments
        parsed_args = parse_arguments(args)
        validate_arguments(parsed_args)
        
        # Configure logging early
        configure_logging(parsed_args)
        logger = logging.getLogger(__name__)
        
        logger.info("PETRE application starting")
        logger.debug("Arguments: %s", vars(parsed_args))
        
        # Create configuration
        config = create_configuration(parsed_args)
        
        # Print configuration summary
        print_configuration_summary(config, parsed_args)
        
        # Run the workflow
        results = run_petre_workflow(config, dry_run=parsed_args.dry_run)
        
        # Print results summary
        print_results_summary(results, parsed_args)
        
        logger.info("PETRE application completed successfully")
        return 0
        
    except Exception as e:
        # Handle all errors through CLI error handler
        return handle_error(e, parsed_args if 'parsed_args' in locals() else None)


# Support for both direct execution and module execution
if __name__ == "__main__":
    # Direct execution: python3 main.py
    exit_code = main()
    sys.exit(exit_code)


# Module execution entry point
def cli_main() -> None:
    """Entry point for python3 -m petre.main"""
    exit_code = main()
    sys.exit(exit_code)


# Programmatic interface for external usage
def run_petre_from_config(config_path: str, **overrides) -> Dict[str, Any]:
    """
    Run PETRE programmatically from configuration file.
    
    Args:
        config_path: Path to JSON configuration file
        **overrides: Configuration overrides
        
    Returns:
        Dictionary containing results
        
    Raises:
        PETREError: If execution fails
    """
    # Import here to avoid circular dependencies
    config_module = smart_import('config', ['create_app_config'])
    create_app_config = config_module['create_app_config']
    
    # Create configuration
    config = create_app_config(config_path, **overrides)
    
    # Run workflow
    return run_petre_workflow(config)


def run_petre_from_dict(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run PETRE programmatically from configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Dictionary containing results
        
    Raises:
        PETREError: If execution fails
    """
    # Create temporary config file or handle dict config
    # For now, raise NotImplementedError
    raise NotImplementedError("Dictionary configuration not yet implemented")


# Legacy compatibility function
def run_petre_legacy(**kwargs) -> Any:
    """
    Legacy interface for backward compatibility.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        PETRE instance or results (for backward compatibility)
    """
    # This would create a PETRE instance compatible with the original API
    # Implementation depends on backward compatibility requirements
    raise NotImplementedError("Legacy interface not yet implemented")


# Application metadata
__version__ = "2.0.0"
__author__ = "PETRE Development Team"
__description__ = "Privacy-preserving Entities Transparency and Re-identification Evaluation"


# Convenience functions for common workflows
def validate_configuration(config_path: str) -> bool:
    """
    Validate a configuration file without running the workflow.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        True if configuration is valid
        
    Raises:
        PETREError: If configuration is invalid
    """
    try:
        config_module = smart_import('config', ['create_app_config'])
        create_app_config = config_module['create_app_config']
        
        config = create_app_config(config_path)
        
        # Run dry-run validation
        run_petre_workflow(config, dry_run=True)
        
        return True
        
    except Exception as e:
        raise PETREError(f"Configuration validation failed: {e}") from e


def get_default_configuration() -> Dict[str, Any]:
    """
    Get default configuration template.
    
    Returns:
        Dictionary with default configuration structure
    """
    return {
        "output_base_folder_path": "./outputs",
        "data_file_path": "./data/dataset.json",
        "individual_name_column": "name",
        "original_text_column": "text", 
        "starting_anonymization_path": "./data/annotations.json",
        "tri_pipeline_path": "./models/tri-pipeline",
        "ks": [2, 3, 5],
        "mask_text": "",
        "use_mask_all_instances": True,
        "explainability_mode": "SHAP",
        "use_chunking": True
    }


# Export public API
__all__ = [
    # Main functions
    'main',
    'cli_main',
    
    # Programmatic interfaces
    'run_petre_from_config',
    'run_petre_from_dict',
    'run_petre_legacy',
    
    # Workflow functions
    'create_petre_orchestrator',
    'run_petre_workflow',
    
    # Utility functions
    'validate_configuration',
    'get_default_configuration',
    
    # Metadata
    '__version__',
    '__author__',
    '__description__'
]