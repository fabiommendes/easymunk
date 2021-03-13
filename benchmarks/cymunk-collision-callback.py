import timeit

s = """
import cymunk as pymunk
#print("easymunk.version", easymunk.version)
s = easymunk.Space()
s.add(easymunk.Circle(s.static_body, 5))
b = easymunk.Body(1,10)
c = easymunk.Circle(b, 5)
s.add(b, c)
def f(arb):
    return False
s.set_default_collision_handler(pre_solve=f)
"""
print(min(timeit.repeat("s.step(0.01)", setup=s, repeat=10)))
