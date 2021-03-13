import timeit

s = """
import pymunk
print("easymunk.version", easymunk.version)
b = easymunk.Body(1, 10)
b.position = 1.0, 2.0
b.angle = 3.0
t = 0
"""
print(
    min(
        timeit.repeat(
            """
t += b.position.x + b.position.y + b.angle
""",
            setup=s,
            repeat=10,
        )
    )
)
