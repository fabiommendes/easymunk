"""
Remake of the pyramid demo from the box2d testbed.
"""

from random import uniform

import pygame

import easymunk as mk
import easymunk.pygame as util
from easymunk import Vec2d


class Tangram:
    def __init__(self, fps=30):
        self.running = True
        self.fps = fps
        self.width, self.height = shape = 800, 600
        self.screen = pygame.display.set_mode(shape)
        self.clock = pygame.time.Clock()
        self.space = mk.Space()

        # Ground
        a = Vec2d(20, 30)
        b = Vec2d(self.width - 20, 70)
        body = self.space.static_body
        line = lambda u, v: body.create_segment(u, v, radius=10.0, friction=1.0)
        line(a, b), line(a, a + (-10, 200)), line(b, b + (10, 200))

        # Actions
        self.step = 0
        self.next_action = iter(self.actions()).__next__

        # Draw options for drawing
        self.draw_options = mk.pygame.DrawOptions(self.screen, flip_y=True)

    def actions(self):
        L = 100
        pos = Vec2d(self.width // 2, self.height // 2 + 100)
        add = self.space.add
        shape = mk.Poly(
            vertices=[(0, 0), (L, 0), (0, L)],
            body=mk.Body(position=pos + (L, 0)),
            density=1,
        )

        yield add(t1 := shape.body)
        yield add(sqr := t1.copy().move(L, L).rotate(180).fuse_with(t1).move(-L, 0))
        yield add(t2 := t1.copy().rotate(90).move(-L, 0))
        yield add(rhombus := t2.copy().rotate(180).move(0, L).fuse_with(t2).move(-L, 0))
        yield add(t3 := t1.copy().rotate(90).fuse_with(t1).move(-L, L))
        yield add(t4 := t1.copy().rotate(180).scale(2).move(-L, 0))
        yield add(t5 := t4.copy().rotate(90))

        v = 300
        for shape in self.space.shapes:
            shape.elasticity = 0.5
            shape.friction = 0.5
        for body in self.space.bodies:
            if body.body_type == body.DYNAMIC:
                body.angular_velocity = uniform(-500, 500)
        self.space.gravity = (0.0, -900.0)

        while True:
            yield

    def run(self):
        while self.running:
            if self.step % 5 == 0:
                self.next_action()
            self.loop()
            self.step += 1

    def loop(self):
        for event in pygame.event.get():
            if (
                event.type == pygame.QUIT
                or event.type == pygame.KEYDOWN
                and event.key == pygame.K_ESCAPE
            ):
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                pygame.image.save(self.screen, "tangram.png")

        dt = 1.0 / self.fps
        self.space.step(dt)
        self.draw()

        # Tick clock and update fps in title
        self.clock.tick(self.fps)
        pygame.display.set_caption("fps: " + str(self.clock.get_fps()))

    def draw(self):
        self.screen.fill(pygame.Color("white"))
        self.space.debug_draw(self.draw_options)
        pygame.display.flip()


if __name__ == "__main__":
    game = Tangram()
    game.run()
