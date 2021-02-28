import hypothesis.strategies as st
from ..vec2d import Vec2d
from .. import shapes


nice_float = lambda **kwargs: st.floats(allow_nan=False, allow_infinity=False, **kwargs)

#
# Vectors and matrices
#
vec2d_vecs = lambda: st.builds(Vec2d, nice_float(), nice_float())
vec2d_tuples = lambda: st.builds(lambda x, y: (x, y), nice_float(), nice_float())
vecs2d = lambda: st.one_of(vec2d_vecs(), vec2d_tuples())

#
# Shapes
#
