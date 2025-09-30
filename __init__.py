"""
PETRE: Privacy-preserving Entities Transparency and Re-identification Evaluation

A professional Python package for evaluating privacy risks and implementing
k-anonymity protection for text data containing personally identifiable information.

Public API exports and package initialization.
"""

from __future__ import annotations

# Version information
__version__ = "2.0.0"
__author__ = "PETRE Development Team"
__email__ = "petre@example.com"
__description__ = "Privacy-preserving Entities Transparency and Re-identification Evaluation"
__license__ = "MIT"

# Import smart import utilities first
try:
    from .import_utils import smart_import
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from import_utils import smart_import

# Core interfaces - always available
interfaces_imports = smart_import('interfaces', [
    'DataProcessor', 'ModelManager', 'ExplainabilityMethod',
    'PipelineEvaluator', 'AnonymizationEngine', 'PETREOrchestrator',
    'PETREDataset', 'ComponentFactory',
    'PETREError', 'ConfigurationError', 'DataProcessingError',
    'ModelError', 'EvaluationError', 'AnonymizationError'
])

# Configuration management
config_imports = smart_import('config', [
    'AppConfig', 'create_app_config', 'load_config_from_file',
    'validate_config_data', 'get_config_with_defaults'
])

# Context managers for resource lifecycle
contexts_imports = smart_import('contexts', [
    'TRIPipelineContext', 'SpaCyModelContext', 'SHAPExplainerContext',
    'ModelResourceContext', 'memory_management_context', 'device_context'
])

# Core implementations
core_imports = smart_import('core', [
    'PETREDataProcessor', 'PETREModelManager',
    'SHAPExplainabilityMethod', 'GreedyExplainabilityMethod',
    'PETREPipelineEvaluator', 'PETREDatasetImpl',
    'PETREComponentFactory', 'PETREOrchestratorImpl'
])

# Main application interface
main_imports = smart_import('main', [
    'run_petre_from_config', 'run_petre_from_dict', 'validate_configuration',
    'get_default_configuration', 'create_petre_orchestrator', 'run_petre_workflow'
])

# CLI functionality (optional - may not be needed for library usage)
try:
    cli_imports = smart_import('cli', [
        'build_parser', 'parse_arguments', 'configure_logging'
    ])
    _CLI_AVAILABLE = True
except ImportError:
    _CLI_AVAILABLE = False


# === PUBLIC API EXPORTS ===

# Core Interfaces
DataProcessor = interfaces_imports['DataProcessor']
ModelManager = interfaces_imports['ModelManager']
ExplainabilityMethod = interfaces_imports['ExplainabilityMethod']
PipelineEvaluator = interfaces_imports['PipelineEvaluator']
AnonymizationEngine = interfaces_imports['AnonymizationEngine']
PETREOrchestrator = interfaces_imports['PETREOrchestrator']
PETREDataset = interfaces_imports['PETREDataset']
ComponentFactory = interfaces_imports['ComponentFactory']

# Exception Types
PETREError = interfaces_imports['PETREError']
ConfigurationError = interfaces_imports['ConfigurationError']
DataProcessingError = interfaces_imports['DataProcessingError']
ModelError = interfaces_imports['ModelError']
EvaluationError = interfaces_imports['EvaluationError']
AnonymizationError = interfaces_imports['AnonymizationError']

# Configuration Management
AppConfig = config_imports['AppConfig']
create_app_config = config_imports['create_app_config']
load_config_from_file = config_imports['load_config_from_file']
validate_config_data = config_imports['validate_config_data']

# Context Managers
TRIPipelineContext = contexts_imports['TRIPipelineContext']
SpaCyModelContext = contexts_imports['SpaCyModelContext']
SHAPExplainerContext = contexts_imports['SHAPExplainerContext']
ModelResourceContext = contexts_imports['ModelResourceContext']

# Concrete Implementations
PETREDataProcessor = core_imports['PETREDataProcessor']
PETREModelManager = core_imports['PETREModelManager']
SHAPExplainabilityMethod = core_imports['SHAPExplainabilityMethod']
GreedyExplainabilityMethod = core_imports['GreedyExplainabilityMethod']
PETREPipelineEvaluator = core_imports['PETREPipelineEvaluator']
PETREDatasetImpl = core_imports['PETREDatasetImpl']
PETREComponentFactory = core_imports['PETREComponentFactory']
PETREOrchestratorImpl = core_imports['PETREOrchestratorImpl']

# Main Application Functions
run_petre_from_config = main_imports['run_petre_from_config']
validate_configuration = main_imports['validate_configuration']
get_default_configuration = main_imports['get_default_configuration']
create_petre_orchestrator = main_imports['create_petre_orchestrator']
run_petre_workflow = main_imports['run_petre_workflow']

# Annotation Generation (if available)
try:
    annotation_imports = smart_import('annotation_generator', [
        'AnnotationGenerator', 'AnnotationMethod', 'AnnotationConfig',
        'create_spacy_ner3_annotations', 'create_presidio_annotations', 'create_combined_annotations'
    ])

    AnnotationGenerator = annotation_imports['AnnotationGenerator']
    AnnotationMethod = annotation_imports['AnnotationMethod']
    AnnotationConfig = annotation_imports['AnnotationConfig']
    create_spacy_ner3_annotations = annotation_imports['create_spacy_ner3_annotations']
    create_presidio_annotations = annotation_imports['create_presidio_annotations']
    create_combined_annotations = annotation_imports['create_combined_annotations']
    _ANNOTATION_AVAILABLE = True
except (ImportError, ModuleNotFoundError, KeyError):
    # Annotation generation not available - graceful fallback
    AnnotationGenerator = None
    AnnotationMethod = None
    AnnotationConfig = None
    create_spacy_ner3_annotations = None
    create_presidio_annotations = None
    create_combined_annotations = None
    _ANNOTATION_AVAILABLE = False

# CLI Functions (if available)
if _CLI_AVAILABLE:
    build_parser = cli_imports['build_parser']
    parse_arguments = cli_imports['parse_arguments']
    configure_logging = cli_imports['configure_logging']


# === CONVENIENCE FUNCTIONS ===

def create_default_orchestrator(config_path: str, **overrides):
    """
    Create a PETRE orchestrator with default factory.

    Args:
        config_path: Path to JSON configuration file
        **overrides: Configuration overrides

    Returns:
        Configured PETRE orchestrator
    """
    config = create_app_config(config_path, **overrides)
    return create_petre_orchestrator(config)


def quick_evaluation(config_path: str, **overrides) -> dict:
    """
    Quick evaluation using default settings.

    Args:
        config_path: Path to JSON configuration file
        **overrides: Configuration overrides

    Returns:
        Evaluation results
    """
    return run_petre_from_config(config_path, **overrides)


# === PACKAGE METADATA ===

# Package information
__all__ = [
    # Version and metadata
    '__version__', '__author__', '__description__',

    # Core interfaces
    'DataProcessor', 'ModelManager', 'ExplainabilityMethod',
    'PipelineEvaluator', 'AnonymizationEngine', 'PETREOrchestrator',
    'PETREDataset', 'ComponentFactory',

    # Exception types
    'PETREError', 'ConfigurationError', 'DataProcessingError',
    'ModelError', 'EvaluationError', 'AnonymizationError',

    # Configuration
    'AppConfig', 'create_app_config', 'load_config_from_file',
    'validate_config_data',

    # Context managers
    'TRIPipelineContext', 'SpaCyModelContext', 'SHAPExplainerContext',
    'ModelResourceContext',

    # Implementations
    'PETREDataProcessor', 'PETREModelManager',
    'SHAPExplainabilityMethod', 'GreedyExplainabilityMethod',
    'PETREPipelineEvaluator', 'PETREDatasetImpl',
    'PETREComponentFactory', 'PETREOrchestratorImpl',

    # Main functions
    'run_petre_from_config', 'validate_configuration',
    'get_default_configuration', 'create_petre_orchestrator',
    'run_petre_workflow',

    # Convenience functions
    'create_default_orchestrator', 'quick_evaluation'
]

# Add CLI functions to __all__ if available
if _CLI_AVAILABLE:
    __all__.extend(['build_parser', 'parse_arguments', 'configure_logging'])

# Add annotation generation functions to __all__ if available
if _ANNOTATION_AVAILABLE:
    __all__.extend([
        'AnnotationGenerator', 'AnnotationMethod', 'AnnotationConfig',
        'create_spacy_ner3_annotations', 'create_presidio_annotations',
        'create_combined_annotations'
    ])


# Package initialization logging
import logging
_logger = logging.getLogger(__name__)
_logger.debug("PETRE package initialized - version %s", __version__)
_logger.debug("CLI functionality available: %s", _CLI_AVAILABLE)