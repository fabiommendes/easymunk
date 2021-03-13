"""A very basic flipper game.
"""
__version__ = "$Id:$"
__docformat__ = "reStructuredText"

import random

import pygame

import easymunk
import easymunk.pygame
from easymunk import Vec2d

pygame.init()
screen = pygame.display.set_mode((600, 600))
clock = pygame.time.Clock()
running = True

# Physics stuff
space = easymunk.Space()
space.gravity = (0.0, 900.0)
draw_options = easymunk.pygame.DrawOptions(screen)

# Balls
balls = []

# walls
static_lines = [
    easymunk.Segment((150, 500), (50, 50), 1.0, space.static_body),
    easymunk.Segment((450, 500), (550, 50), 1.0, space.static_body),
    easymunk.Segment((50, 50), (300, 0), 1.0, space.static_body),
    easymunk.Segment((300, 0), (550, 50), 1.0, space.static_body),
    easymunk.Segment((300, 180), (400, 200), 1.0, space.static_body),
]
for line in static_lines:
    line.elasticity = 0.7
    line.group = 1
space.add(*static_lines)

fp = [(20, -20), (-120, 0), (20, 20)]
mass = 100
moment = easymunk.moment_for_poly(mass, fp)

# right flipper
r_flipper_body = easymunk.Body(mass, moment)
r_flipper_body.position = 450, 500
r_flipper_shape = easymunk.Poly(fp, body=r_flipper_body)
space.add(r_flipper_body, r_flipper_shape)

r_flipper_joint_body = easymunk.Body(body_type=easymunk.Body.KINEMATIC)
r_flipper_joint_body.position = r_flipper_body.position
j = easymunk.PinJoint(r_flipper_body, r_flipper_joint_body, (0, 0), (0, 0))
# todo: tweak values of spring better
s = easymunk.DampedRotarySpring(
    r_flipper_body, r_flipper_joint_body, 0.15, 20000000, 900000
)
space.add(j, s)

# left flipper
l_flipper_body = easymunk.Body(mass, moment)
l_flipper_body.position = 150, 500
l_flipper_shape = easymunk.Poly([(-x, y) for x, y in fp], body=l_flipper_body)
space.add(l_flipper_body, l_flipper_shape)

l_flipper_joint_body = easymunk.Body(body_type=easymunk.Body.KINEMATIC)
l_flipper_joint_body.position = l_flipper_body.position
j = easymunk.PinJoint(l_flipper_body, l_flipper_joint_body, (0, 0), (0, 0))
s = easymunk.DampedRotarySpring(
    l_flipper_body, l_flipper_joint_body, -0.15, 20000000, 900000
)
space.add(j, s)

r_flipper_shape.group = l_flipper_shape.group = 1
r_flipper_shape.elasticity = l_flipper_shape.elasticity = 0.4

# "bumpers"
for p in [(240, 100), (360, 100)]:
    body = easymunk.Body(body_type=easymunk.Body.KINEMATIC)
    body.position = p
    shape = easymunk.Circle(10, body=body)
    shape.elasticity = 1.5
    space.add(body, shape)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
            pygame.image.save(screen, "flipper.png")

        elif event.type == pygame.KEYDOWN and event.key == pygame.K_j:
            r_flipper_body.apply_impulse_at_local_point(
                Vec2d.unit() * -40000, (-100, 0)
            )
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:
            l_flipper_body.apply_impulse_at_local_point(Vec2d.unit() * 40000, (-100, 0))
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_b:
            mass = 1
            radius = 25
            inertia = easymunk.moment_for_circle(mass, 0, radius, (0, 0))
            body = easymunk.Body(mass, inertia)
            x = random.randint(115, 350)
            body.position = x, 200
            shape = easymunk.Circle(radius, (0, 0), body)
            shape.elasticity = 0.95
            space.add(body, shape)
            balls.append(shape)

    # Clear screen
    screen.fill(pygame.Color("white"))

    # Draw stuff
    space.debug_draw(draw_options)

    r_flipper_body.position = 450, 500
    l_flipper_body.position = 150, 500
    r_flipper_body.velocity = l_flipper_body.velocity = 0, 0

    # Remove any balls outside
    to_remove = []
    for ball in balls:
        if ball.body.position.distance((300, 300)) > 1000:
            to_remove.append(ball)

    for ball in to_remove:
        space.remove(ball.body, ball)
        balls.remove(ball)

    # Update physics
    dt = 1.0 / 60.0 / 5.0
    for x in range(5):
        space.step(dt)

    # Flip screen
    pygame.display.flip()
    clock.tick(50)
    pygame.display.set_caption("fps: " + str(clock.get_fps()))
