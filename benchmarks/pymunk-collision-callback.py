import timeit

s = """
import pymunk
print("easymunk.version", easymunk.version)
s = easymunk.Space()
s.add(easymunk.Circle(s.static_body, 5))
b = easymunk.Body(1,10)
c = easymunk.Circle(b, 5)
s.add(b, c)
h = s.add_default_collision_handler()
def f(arb, s, data):
    return False
h.pre_solve = f
"""

print(min(timeit.repeat("s.step(0.01)", setup=s, repeat=10)))
