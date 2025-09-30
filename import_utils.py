"""
Smart import utilities for dual execution support.

Handles automatic fallback between relative and absolute imports,
enabling both direct script execution and module execution patterns.
"""

from __future__ import annotations

import sys
import os
import importlib
from typing import Dict, Any, List


def smart_import(module_name: str, items: List[str]) -> Dict[str, Any]:
    """
    Import items with automatic fallback between relative and absolute imports.

    This function detects the execution context and uses the appropriate import
    strategy to support both:
    - Direct execution: python3 main.py
    - Module execution: python3 -m package.main

    Args:
        module_name: Name of the module to import from
        items: List of items to import from the module

    Returns:
        Dictionary mapping item names to imported objects

    Raises:
        ImportError: If the module or items cannot be imported
    """
    # Determine if we're running as a module or script
    calling_frame = sys._getframe(1)
    calling_module = calling_frame.f_globals.get('__name__', '')
    calling_file = calling_frame.f_globals.get('__file__', '')

    # Check if we're running as a package (__main__ in package context)
    is_package_execution = (
        calling_module.startswith('petre.') or
        calling_module == 'petre' or
        (calling_module == '__main__' and calling_file and 'petre' in calling_file)
    )

    imported_items = {}

    try:
        # First attempt: relative import (for package execution)
        if is_package_execution:
            try:
                module = importlib.import_module(f'.{module_name}', package='petre')
            except (ImportError, ValueError):
                # Fallback to absolute import
                module = importlib.import_module(module_name)
        else:
            # Direct script execution: use absolute import
            # Add current directory to path if needed
            current_dir = os.path.dirname(os.path.abspath(calling_file)) if calling_file else os.getcwd()
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)

            module = importlib.import_module(module_name)

    except ImportError:
        # Final fallback: try the other approach
        try:
            if is_package_execution:
                # Try absolute import
                current_dir = os.path.dirname(os.path.abspath(calling_file)) if calling_file else os.getcwd()
                if current_dir not in sys.path:
                    sys.path.insert(0, current_dir)
                module = importlib.import_module(module_name)
            else:
                # Try relative import
                package_name = _determine_package_name(calling_file)
                if package_name:
                    module = importlib.import_module(f'.{module_name}', package=package_name)
                else:
                    raise ImportError(f"Cannot import {module_name}")
        except ImportError as e:
            raise ImportError(f"Failed to import {module_name} using both relative and absolute imports: {e}")

    # Extract requested items from the module
    for item in items:
        if hasattr(module, item):
            imported_items[item] = getattr(module, item)
        else:
            raise ImportError(f"Cannot import name '{item}' from '{module_name}'")

    return imported_items


def _determine_package_name(file_path: str) -> str | None:
    """
    Determine the package name from the file path.

    Args:
        file_path: Path to the calling file

    Returns:
        Package name if determinable, None otherwise
    """
    if not file_path:
        return None

    # Look for package indicators
    path_parts = os.path.normpath(file_path).split(os.sep)

    # Find 'petre' in the path
    for i, part in enumerate(path_parts):
        if part == 'petre':
            # Check if there's an __init__.py in the petre directory
            petre_dir = os.sep.join(path_parts[:i+1])
            if os.path.exists(os.path.join(petre_dir, '__init__.py')):
                return 'petre'

    return None


def ensure_module_path() -> None:
    """
    Ensure the current module directory is in sys.path.

    This is useful for direct script execution to ensure
    all local modules can be imported.
    """
    # Get the directory of the calling file
    calling_frame = sys._getframe(1)
    calling_file = calling_frame.f_globals.get('__file__')

    if calling_file:
        module_dir = os.path.dirname(os.path.abspath(calling_file))
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)


def smart_import_single(module_name: str, item_name: str) -> Any:
    """
    Import a single item using smart import strategy.

    Convenience function for importing a single item.

    Args:
        module_name: Name of the module to import from
        item_name: Name of the item to import

    Returns:
        The imported item
    """
    result = smart_import(module_name, [item_name])
    return result[item_name]


# Convenience functions for common patterns
def import_config(items: List[str]) -> Dict[str, Any]:
    """Import items from config module."""
    return smart_import('config', items)


def import_interfaces(items: List[str]) -> Dict[str, Any]:
    """Import items from interfaces module."""
    return smart_import('interfaces', items)


def import_core(items: List[str]) -> Dict[str, Any]:
    """Import items from core module."""
    return smart_import('core', items)


def import_contexts(items: List[str]) -> Dict[str, Any]:
    """Import items from contexts module."""
    return smart_import('contexts', items)


def import_cli(items: List[str]) -> Dict[str, Any]:
    """Import items from cli module."""
    return smart_import('cli', items)


# Re-export for convenience
__all__ = [
    'smart_import',
    'smart_import_single',
    'ensure_module_path',
    'import_config',
    'import_interfaces',
    'import_core',
    'import_contexts',
    'import_cli'
]