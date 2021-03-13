import time

import easymunk
from easymunk.vec2d import Vec2d


class PyramidDemo:
    def __init__(self, threads=1):

        # Init pymunk and create space
        if threads == 0:
            self.space = easymunk.Space(threaded=False)
        else:
            self.space = easymunk.Space(threaded=True)
        self.space.gravity = (0.0, -900.0)
        self.space.threads = threads

        # ground
        shape = easymunk.Segment((5, 100), (595, 100), 1.0, self.space.static_body)
        shape.friction = 1.0
        self.space.add(shape)

        # pyramid
        x = Vec2d(-270, 7.5) + (300, 100)
        y = Vec2d(0, 0)
        deltaX = Vec2d(0.5625, 1.1) * 20
        deltaY = Vec2d(1.125, 0.0) * 20

        for i in range(25):
            y = Vec2d(*x)
            for j in range(i, 25):
                size = 10
                points = [(-size, -size), (-size, size), (size, size), (size, -size)]
                mass = 1.0
                moment = easymunk.moment_for_poly(mass, points, (0, 0))
                body = easymunk.Body(mass, moment)
                body.position = y
                shape = easymunk.Poly(points, body=body)
                shape.friction = 1
                self.space.add(body, shape)

                y += deltaY

            x += deltaX

    def step(self, n=1):
        for x in range(n):
            dt = 1.0 / 150.0
            self.space.step(dt)


if __name__ == "__main__":

    for num_threads in [0, 1, 2, 3, 4, 6, 8]:

        demo = PyramidDemo(threads=num_threads)
        start = time.time()
        demo.step(10000)
        end = time.time()
        print("Threads {}, time {}".format(num_threads, end - start))
