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
config_imports = smart_import('config', ['AppConfig', 'create_app_config'])
interfaces_imports = smart_import('interfaces', ['PETREError', 'ConfigurationError'])
core_imports = smart_import('core', ['PETREComponentFactory'])

# Optional annotation generation support
try:
    annotation_imports = smart_import('annotation_generator', [
        'AnnotationGenerator', 'AnnotationMethod', 'AnnotationConfig'
    ])
    AnnotationGenerator = annotation_imports['AnnotationGenerator']
    AnnotationMethod = annotation_imports['AnnotationMethod']
    AnnotationConfig = annotation_imports['AnnotationConfig']
    _ANNOTATION_AVAILABLE = True
except (ImportError, ModuleNotFoundError, KeyError):
    AnnotationGenerator = None
    AnnotationMethod = None
    AnnotationConfig = None
    _ANNOTATION_AVAILABLE = False

# Extract imports for clean usage
parse_arguments = cli_imports['parse_arguments']
validate_arguments = cli_imports['validate_arguments']
configure_logging = cli_imports['configure_logging']
create_configuration = cli_imports['create_configuration']
print_configuration_summary = cli_imports['print_configuration_summary']
print_results_summary = cli_imports['print_results_summary']
handle_error = cli_imports['handle_error']

# Extract imported functions and classes for use
AppConfig = config_imports['AppConfig']
create_app_config = config_imports['create_app_config']
PETREError = interfaces_imports['PETREError']
ConfigurationError = interfaces_imports['ConfigurationError']
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
            "starting_anonymization": config.starting_annon_name,
            "orchestrator": orchestrator  # Include orchestrator for post-processing
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

        # Generate anonymized texts if requested
        if parsed_args.write_anonymized_texts and not parsed_args.dry_run:
            logger.info("Generating anonymized texts file")
            orchestrator = results.get("orchestrator")
            if orchestrator:
                orchestrator.write_anonymized_texts(
                    config.ks,
                    parsed_args.anonymized_texts_filename
                )

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
def run_petre_from_config(config_path_or_config, **overrides) -> Dict[str, Any]:
    """
    Run PETRE from a configuration file or AppConfig object with optional parameter overrides.

    Args:
        config_path_or_config: Path to JSON configuration file OR AppConfig instance
        **overrides: Optional parameter overrides

    Returns:
        Dictionary containing evaluation results

    Example:
        >>> results = run_petre_from_config('config.json', ks=[2, 3, 5])
        >>> results = run_petre_from_config(config_object)
    """
    # Handle both string path and AppConfig object
    if isinstance(config_path_or_config, str):
        config = create_app_config(config_path_or_config, **overrides)
    elif isinstance(config_path_or_config, AppConfig):
        config = config_path_or_config
        # Apply overrides if any (create new config with updates)
        if overrides:
            config_dict = vars(config).copy()
            config_dict.update(overrides)
            # We'd need to recreate from dict, but this is complex with dataclass
            # For now, just use as-is and log warning
            if overrides:
                logging.warning("Overrides not supported when passing AppConfig object directly")
    else:
        raise ConfigurationError(f"Expected string path or AppConfig object, got {type(config_path_or_config)}")

    # Handle dynamic annotation generation if needed
    config = _handle_annotation_generation(config)

    return run_petre_workflow(config)


def _handle_annotation_generation(config: AppConfig) -> AppConfig:
    """
    Handle dynamic annotation generation if annotation_method is specified.

    Args:
        config: Application configuration

    Returns:
        Updated configuration with generated annotations
    """
    if config.annotation_method and _ANNOTATION_AVAILABLE:
        # Map string method names to enum values
        method_mapping = {
            'spacy_ner3': AnnotationMethod.SPACY_NER3,
            'spacy_ner4': AnnotationMethod.SPACY_NER4,
            'spacy_ner7': AnnotationMethod.SPACY_NER7,
            'presidio': AnnotationMethod.PRESIDIO,
            'combined': AnnotationMethod.COMBINED
        }

        method = method_mapping.get(config.annotation_method)
        if not method:
            raise ConfigurationError(f"Unknown annotation method: {config.annotation_method}")

        # Create annotation generator
        generator = AnnotationGenerator()

        # Configure annotation generation
        annotation_config = AnnotationConfig(
            method=method,
            confidence_threshold=config.annotation_confidence_threshold,
            entity_types=set(config.annotation_entity_types) if config.annotation_entity_types else None,
            merge_overlapping=True,
            min_span_length=2,
            max_span_length=100
        )

        # Generate annotations
        logging.info("Generating annotations using method: %s", config.annotation_method)

        if method == AnnotationMethod.COMBINED:
            # Use multiple methods for combined approach
            methods = [AnnotationMethod.SPACY_NER3, AnnotationMethod.PRESIDIO]
            annotations = generator.generate_combined_annotations(
                data_file=config.data_file_path,
                methods=methods,
                individual_name_column=config.individual_name_column,
                text_column=config.original_text_column,
                config=annotation_config
            )
        else:
            annotations = generator.generate_annotations(
                data_file=config.data_file_path,
                individual_name_column=config.individual_name_column,
                text_column=config.original_text_column,
                config=annotation_config
            )

        # Save generated annotations to the same directory structure as the original
        annotation_file = os.path.join(config.output_base_folder_path, f"Annotations_{config.annotation_method}_generated.json")

        # Ensure output directory exists (but don't create extra nested folders)
        os.makedirs(config.output_base_folder_path, exist_ok=True)
        generator.save_annotations(annotations, annotation_file, config.annotation_method)

        logging.info("Generated %d annotation sets", len(annotations))
        logging.info("Saved annotations to: %s", annotation_file)

        # Create new config with the generated annotation file
        config_dict = {
            **{field.name: getattr(config, field.name) for field in config.__dataclass_fields__.values() if field.init},
            'starting_anonymization_path': annotation_file,
            'annotation_method': None  # Clear this to avoid re-generation
        }

        return AppConfig(**config_dict)

    return config


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