Easymunk
========

.. image::  https://raw.githubusercontent.com/fabiommendes/easymunk/master/docs/src/_static/easymunk_logo_animation.gif

Easymunk is a easy-to-use pythonic 2d physics library that can be used whenever
you need 2d rigid body physics from Python. Perfect when you need 2d physics
in your game, demo or other application! It is built on top of the very 
capable 2d physics library `Chipmunk <http://chipmunk-physics.net>`_.


Easymunk is a fork of the excellent Easymunk project, but it allows itself to deviate
more from the original C-library. The goal is to explore a more Pythonic interface
and tends to be easier to use.

The first version was released in 2021, based on Pymunk 6.0. It owns greatly from Pymunk's
maturity and 10 years of active development! Easymunk is a laboratory and we hope to
give back code to Pymunk upstream and collaborate with its development.

Pymunk: 2007 - 2020, Victor Blomqvist - vb@viblo.se, MIT License
Easymunk: 2021, Fábio Macêdo Mendes - fabiomacedomendese@gmail.com, MIT License


Installation
------------

In the normal case Easymunk can be installed from PyPI with pip::

    > pip install easymunk

It has one direct dependency, CFFI.

Easymunk can also be installed with conda, from the conda-forge channel::

    > conda install -c conda-forge easymunk


Example
-------

Quick code example::
    
    import easymunk as mk       # Import easymunk.

    space = mk.Space(           # Create a Space which contain the simulation
        gravity=(0, -10),       # setting its gravity
    )

    body = space.add_box(       # Create a Body with mass, moment,
        mass=10,                # position and shape.
        moment=150,
        position=(50,100),
        width=10,
        height=20,
    )

    while True:                 # Infinite loop simulation
        space.step(0.02)        # Step the simulation one step forward
        space.debug_draw()      # Print the state of the simulation
    
For more detailed and advanced examples, take a look at the included demos 
(in examples/).

Examples are not included if you install with `pip install easymunk`. Instead you
need to download the source archive (easymunk-x.y.z.zip). Download available from
https://pypi.org/project/easymunk/#files


Documentation
-------------

The source distribution of Easymunk ships with a number of demos of different
simulations in the examples directory, and it also contains the full 
documentation including API reference.

You can also find the full documentation including examples and API reference 
on the Easymunk homepage, http://fabiommendes.github.io/easymunk.


The Easymunk Vision
-------------------

    "*Make 2d physics easy to include in your game*"

It is (or is striving to be):

* **Easy to use** - It should be easy to use, no complicated code should be 
  needed to add physics to your game or program.
* **"Pythonic"** - It should not be visible that a c-library (Chipmunk) is in 
  the bottom, it should feel like a Python library (no strange naming, OO, 
  no memory handling and more)
* **Simple to build & install** - You shouldn't need to have a zillion of 
  libraries installed to make it install, or do a lot of command line tricks.
* **Multi-platform** - Should work on both Windows, \*nix and OSX.
* **Non-intrusive** - It should not put restrictions on how you structure 
  your program and not force you to use a special game loop, it should be 
  possible to use with other libraries like Pygame and Pyglet. 

  
Contact & Support
-----------------
.. _contact-support:

**Homepage**
    http://fabiommendes.github.io/easymunk

**Stackoverflow**
    You can ask questions/browse old ones at Stackoverflow, just look for 
    the Easymunk tag. http://stackoverflow.com/questions/tagged/easymunk

**Issue Tracker**
    Please use the issue tracker at github to report any issues you find:
    https://github.com/fabiommendes/easymunk/issues
    
Regardless of the method you use I will try to answer your questions as soon 
as I see them. (And if you ask on SO other people might help as well!)


Dependencies / Requirements
---------------------------

Basically Easymunk have been made to be as easy to install and distribute as
possible, usually `pip install` will take care of everything for you.

- Python (Runs on CPython 3.7 and later and Pypy3)
- Chipmunk (Compiled library already included on common platforms)
- CFFI (will be installed automatically by Pip)
- Setuptools (should be included with Pip)

* GCC and friends (optional, you need it to compile Easymunk from source. On
  windows Visual Studio is required to compile)
* Pygame (optional, you need it to run the Pygame based demos)
* Pyglet (optional, you need it to run the Pyglet based demos)
* Pyxel (optional, you need it to run the Pyxel based demos)
* Matplotlib & Jupyter Notebook (optional, you need it to run the Matplotlib 
  based demos)
* Sphinx & aafigure & sphinx_autodoc_typehints (optional, you need it to build 
  documentation)


Install from source / Chipmunk Compilation
------------------------------------------

This section is only required in case you do not install easymunk from the
prebuild binary wheels (normally if you do not use `pip install` or you are 
on a uncommon platform).

Easymunk is built on top of the c library Chipmunk. It uses CFFI to interface
with the Chipmunk library file. Because of this Chipmunk has to be compiled
together with Easymunk as an extension module.

There are basically two options, either building it automatically as part of 
installation using for example Pip::

    > pip install easymunk-source-dist.zip

Or if you have the source unpacked / you got Easymunk by cloning its git repo,
you can explicitly tell Easymunk to compile it inplace::

    > python setup.py build_ext --inplace
