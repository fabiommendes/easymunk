"""
A clone of the math module, but trigonometric functions are based on degrees rather
than radians.
"""
from math import *

# __all__ = ["cos", "sin", "tan", "acos", "asin", "atan", "atan2", "pi", "degrees",
#            "radians", "sqrt", "log", "log10", "log2"]
_cos, _sin, _tan = cos, sin, tan
_acos, _asin, _atan, _atan2 = acos, asin, atan, atan2

cos = lambda x: _cos(radians(x))
sin = lambda x: _sin(radians(x))
tan = lambda x: _tan(radians(x))

acos = lambda x: degrees(_acos(x))
asin = lambda x: degrees(_asin(x))
atan = lambda x: degrees(_atan(x))
atan2 = lambda x, y: degrees(_atan2(x, y))
