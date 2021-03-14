import pyxel
import easymunk.pyxel as phys

# A very simple example. Maybe add more things later
pyxel.init(120, 80)
pyxel.mouse(True)
space = phys.space(gravity=(0, -50), flip_y=True)
circ = phys.circ(60, 40, 5, velocity=(50, 50), friction=0.5)
tri = phys.tri(5, 5, 20, 20, 5, 20, velocity=(100, 100), friction=0.5)
margins = phys.margin(friction=1.0, elasticity=0.95)
space.run(wireframe=True)
