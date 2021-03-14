"""
This submodule contains helper functions to help with quick prototyping
using pymunk together with pyglet.

Intended to help with debugging and prototyping, not for actual production use
in a full application. The methods contained in this module is opinionated
about your coordinate system and not very optimized (they use batched
drawing, but there is probably room for optimizations still).
"""

__docformat__ = "reStructuredText"
from .draw_options import DrawOptions
