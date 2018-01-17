"""This package contains classes/functions to serve as utils for the
potential requirment of modeling accelerator with FLAME code.
"""
from .core import *
from .io import *
from .misc import *

__all__ = [
    'BeamState', 'ModelFlame', 'collect_data', 'configure',
    'convert_results', 'generate_latfile', 'get_all_names',
    'get_all_types', 'get_element', 'get_index_by_name',
    'get_index_by_type', 'get_names_by_pattern', 'inspect_lattice',
    'propagate', 'machine_setter', 'flatten', 'get_intersection',
]

__version__ = "0.2.1"
