"""
Command-line interface for PETRE application.

Handles argument parsing, validation, logging configuration, and user interaction.
Separated from business logic for clean architecture.
"""

from __future__ import annotations

import sys
import os
import argparse
import logging
from typing import Dict, Any, List, Optional

try:
    from .import_utils import smart_import
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from import_utils import smart_import

# Import dependencies
config_imports = smart_import('config', ['create_app_config', 'ConfigurationError'])
interfaces_imports = smart_import('interfaces', ['PETREError'])

create_app_config = config_imports['create_app_config']
ConfigurationError = config_imports['ConfigurationError']
PETREError = interfaces_imports['PETREError']


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="PETRE: Privacy-preserving Entities Transparency and Re-identification Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s config.json
  %(prog)s config.json --verbose
  %(prog)s config.json --output-dir ./results --mask-text "[REDACTED]"

Configuration file should contain:
  - output_base_folder_path: Base directory for outputs
  - data_file_path: Path to dataset (JSON or CSV)
  - individual_name_column: Column name for individual identifiers
  - original_text_column: Column name for text content
  - starting_anonymization_path: Path to initial annotations
  - tri_pipeline_path: Path to TRI model directory
  - ks: List of k-anonymity values to evaluate
        """
    )

    # Positional arguments
    parser.add_argument(
        'config_file',
        help='Path to JSON configuration file'
    )

    # Optional configuration overrides
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output base folder path'
    )

    parser.add_argument(
        '--mask-text',
        type=str,
        default='',
        help='Text to use for masking (default: empty string)'
    )

    parser.add_argument(
        '--explainability-mode',
        choices=['SHAP', 'Greedy'],
        help='Explainability method to use'
    )

    parser.add_argument(
        '--no-mask-all-instances',
        action='store_true',
        help='Disable masking all instances of disclosive terms'
    )

    parser.add_argument(
        '--no-chunking',
        action='store_true',
        help='Disable noun phrase chunking in text processing'
    )

    parser.add_argument(
        '--ks',
        nargs='+',
        type=int,
        help='List of k-anonymity values to evaluate'
    )

    parser.add_argument(
        '--write-anonymized-texts',
        action='store_true',
        help='Generate anonymized texts file for all k values after completion'
    )

    parser.add_argument(
        '--anonymized-texts-filename',
        type=str,
        default='anonymized_texts.json',
        help='Filename for anonymized texts output (default: anonymized_texts.json)'
    )

    # Logging and output control
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress most output (errors only)'
    )

    parser.add_argument(
        '--log-file',
        type=str,
        help='Path to log file (default: log to console only)'
    )

    # Development and debugging
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration and setup without running anonymization'
    )

    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        default='auto',
        help='Force specific device (default: auto-detect)'
    )

    return parser


def parse_arguments(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        args: Optional list of arguments (defaults to sys.argv)

    Returns:
        Parsed arguments namespace

    Raises:
        SystemExit: If argument parsing fails
    """
    parser = build_parser()
    return parser.parse_args(args)


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate parsed arguments for consistency and file existence.

    Args:
        args: Parsed arguments namespace

    Raises:
        ConfigurationError: If arguments are invalid
    """
    # Check configuration file exists
    if not os.path.isfile(args.config_file):
        raise ConfigurationError(f"Configuration file not found: {args.config_file}")

    # Validate k values if provided
    if args.ks:
        if not all(k > 0 for k in args.ks):
            raise ConfigurationError("All k values must be positive integers")

    # Validate conflicting options
    if args.verbose and args.quiet:
        raise ConfigurationError("Cannot use both --verbose and --quiet options")

    # Validate log file directory if provided
    if args.log_file:
        log_dir = os.path.dirname(os.path.abspath(args.log_file))
        if not os.path.exists(log_dir):
            raise ConfigurationError(f"Log file directory does not exist: {log_dir}")

    # Validate output directory if provided
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            try:
                os.makedirs(args.output_dir, exist_ok=True)
            except OSError as e:
                raise ConfigurationError(f"Cannot create output directory {args.output_dir}: {e}") from e


def configure_logging(args: argparse.Namespace) -> None:
    """
    Configure logging based on command-line arguments.

    Args:
        args: Parsed arguments namespace
    """
    # Determine log level
    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    # Configure log format
    log_format = '%(asctime)s %(levelname)-8s [%(name)s] %(message)s'

    # Configure handlers
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)

    # File handler if specified
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
        force=True  # Override any existing configuration
    )

    # Log configuration
    logger = logging.getLogger(__name__)
    logger.info("Logging configured - Level: %s", logging.getLevelName(log_level))
    if args.log_file:
        logger.info("Logging to file: %s", args.log_file)


def build_config_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build configuration overrides from command-line arguments.

    Args:
        args: Parsed arguments namespace

    Returns:
        Dictionary of configuration overrides
    """
    overrides = {}

    # Direct mappings
    if args.output_dir:
        overrides['output_base_folder_path'] = args.output_dir

    if args.mask_text is not None:
        overrides['mask_text'] = args.mask_text

    if args.explainability_mode:
        overrides['explainability_mode'] = args.explainability_mode

    if args.ks:
        overrides['ks'] = args.ks

    # Boolean flags (note: these are inverted flags)
    if args.no_mask_all_instances:
        overrides['use_mask_all_instances'] = False

    if args.no_chunking:
        overrides['use_chunking'] = False

    return overrides


def print_configuration_summary(config, args: argparse.Namespace) -> None:
    """
    Print a summary of the configuration.

    Args:
        config: Application configuration
        args: Parsed arguments
    """
    if args.quiet:
        return

    print("\n" + "="*60)
    print("PETRE Configuration Summary")
    print("="*60)
    print(f"Configuration file: {args.config_file}")
    print(f"Output directory: {config.output_folder_path}")
    print(f"Data file: {config.data_file_path}")
    print(f"TRI pipeline: {config.tri_pipeline_path}")
    print(f"Starting anonymization: {config.starting_anonymization_path}")
    print(f"K values: {config.ks}")
    print(f"Explainability mode: {config.explainability_mode}")
    print(f"Device: {config.device}")
    print(f"Mask text: '{config.mask_text}'")
    print(f"Mask all instances: {config.use_mask_all_instances}")
    print(f"Use chunking: {config.use_chunking}")
    print("="*60 + "\n")


def print_results_summary(results: Dict[str, Any], args: argparse.Namespace) -> None:
    """
    Print a summary of the results.

    Args:
        results: Results dictionary
        args: Parsed arguments
    """
    if args.quiet:
        return

    print("\n" + "="*60)
    print("PETRE Results Summary")
    print("="*60)

    # Print k-specific results if available
    for k, result in results.items():
        if isinstance(k, int) or k.startswith('k='):
            print(f"K={k}: {result.get('steps', 0)} anonymization steps")

    print("="*60 + "\n")


def handle_error(error: Exception, args: argparse.Namespace) -> int:
    """
    Handle errors with appropriate logging and exit codes.

    Args:
        error: Exception that occurred
        args: Parsed arguments

    Returns:
        Appropriate exit code
    """
    logger = logging.getLogger(__name__)

    if isinstance(error, ConfigurationError):
        if not args.quiet:
            print(f"Configuration Error: {error}", file=sys.stderr)
        logger.error("Configuration error: %s", error)
        return 2

    elif isinstance(error, PETREError):
        if not args.quiet:
            print(f"PETRE Error: {error}", file=sys.stderr)
        logger.error("PETRE error: %s", error)
        return 3

    elif isinstance(error, KeyboardInterrupt):
        if not args.quiet:
            print("\nOperation cancelled by user", file=sys.stderr)
        logger.info("Operation cancelled by user")
        return 130

    else:
        if not args.quiet:
            print(f"Unexpected error: {error}", file=sys.stderr)
        logger.exception("Unexpected error occurred")
        return 1


def create_configuration(args: argparse.Namespace):
    """
    Create application configuration from arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        AppConfig instance

    Raises:
        ConfigurationError: If configuration is invalid
    """
    # Build overrides from command-line arguments
    overrides = build_config_overrides(args)

    # Create configuration
    config = create_app_config(args.config_file, **overrides)

    return config


# CLI validation functions
def validate_config_file_format(file_path: str) -> None:
    """Validate configuration file format."""
    if not file_path.endswith('.json'):
        raise ConfigurationError("Configuration file must be in JSON format")


def validate_k_values(k_values: List[int]) -> None:
    """Validate k-anonymity values."""
    if not k_values:
        raise ConfigurationError("At least one k value must be specified")

    if not all(isinstance(k, int) and k > 0 for k in k_values):
        raise ConfigurationError("All k values must be positive integers")


def validate_device_choice(device: str) -> None:
    """Validate device choice."""
    valid_devices = ['auto', 'cpu', 'cuda', 'mps']
    if device not in valid_devices:
        raise ConfigurationError(f"Device must be one of {valid_devices}")


__all__ = [
    'build_parser',
    'parse_arguments',
    'validate_arguments',
    'configure_logging',
    'build_config_overrides',
    'print_configuration_summary',
    'print_results_summary',
    'handle_error',
    'create_configuration',
    'validate_config_file_format',
    'validate_k_values',
    'validate_device_choice'
]