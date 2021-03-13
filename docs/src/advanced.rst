Advanced 
========

In this section different "Advanced" topics are covered, things you normally 
dont need to worry about when you use Easymunk but might be of interest if you
want a better understanding of Easymunk for example to extend it.

First off, Easymunk is a pythonic wrapper around the C-library Chipmunk.

To wrap Chipmunk Easymunk uses CFFI in API mode. On top of the CFFI wrapping is
a handmade pythonic layer to make it nice to use from Python code.

Why CFFI?
---------

This is a straight copy from the github issue tracking the CFFI upgrade. 
https://github.com/viblo/pymunk/issues/99

CFFI have a number of advantages but also a downsides.

Advantages (compared to ctypes):

* Its an active project. The developers and users are active, there are new 
  releases being made and its possible to ask and get answers within a day on 
  the CFFI mailing list.
* Its said to be the way forward for Pypy, with promise of better performance 
  compares to ctypes.
* A little easier than ctypes to wrap things since you can just copy-paste the 
  c headers.

Disadvatages (compared to ctypes):

* ctypes is part of the CPython standard library, CFFI is not. That means that 
  it will be more difficult to install Easymunk if it uses CFFI, since a
  copy-paste install is no longer possible in an easy way.

For me I see the 1st advantage as the main point. I have had great difficulties 
with strange segfaults with 64bit pythons on windows, and also sometimes on 
32bit python, and support for 64bit python on both windows and linux is 
something I really want. Hopefully those problems will be easier to handle with 
CFFI since it has an active community.

Then comes the 3rd advantage, that its a bit easier to wrap the c code. For 
ctypes I have a automatic wrapping script that does most of the low level 
wrapping, but its not supported, very difficult to set up (I only managed 
inside a VM with linux) and quite annoying. CFFI would be a clear improvement.

For the disadvantage of ctypes I think it will be acceptable, even if not 
ideal. Many python packages have to be installed in some way (like pygame), 
and nowadays with pip its very easy to do. So I hope that it will be ok.


Code Layout
-----------

Most of Easymunk should be quite straight forward.

Except for the documented API Easymunk has a couple of interesting parts. Low
level bindings to Chipmunk, a custom documentation generation extension and a
customized setup.py file to allow compilation of Chipmunk.

The low level chipmunk bindings are located in the file 
pymunk_extension_build.py. 

docs/src/ext/autoexample.py
    A Sphinx extension that scans a directory and extracts the toplevel 
    docstring. Used to autogenerate the examples documentation.

easymunk/_chipmunk_cffi.py
    This file only contains a call to _chipmunk_cffi_abi.py, and exists mostly
    as a wrapper to be able to switch between abi and api mode of Cffi. This 
    is currently not in use in the relased code, but is used during 
    experimentation.
    
easymunk/_chipmkunk_cffi_abi.py
    This file contains the pure Cffi wrapping definitons. Bascially a giant 
    string created by copy-paster from the relevant header files of Chipmunk.  

setup.py
    Except for the standard setup stuff this file also contain the custom 
    build commands to build Chipmunk from source, using a build_ext extension.

easymunk/tests/*
    Collection of (unit) tests. Does not cover all cases, but most core 
    things are there. The tests require a working chipmunk library file.
    
tools/*
    Collection of helper scripts that can be used to various development tasks
    such as generating documentation.


Tests
-----

There are a number of unit tests included in the easymunk.tests package
(easymunk/tests). Not exactly all the code is tested, but most of it (at the time
of writing its about 85% of the core parts). 

The tests can be run by calling the module ::

    > python -m pymunk.tests

Its possible to control which tests to run, by specifying a filtering 
argument. The matching is as broad as possible, so `UnitTest` matches all the 
unit tests, `test_arbiter` all tests in `test_arbiter.py` and 
`testResetitution` matches the exact `testRestitution` test case ::

    > python -m pymunk.tests -f testRestitution

To see all options to the tests command use -h ::

    > python -m pymunk.tests -h

Since the tests cover even the optional parts, you either have to make sure 
all the optional dependencies are installed, or filter out those tests.
    
    
Working with non-wrapped parts of Chipmunk
------------------------------------------

In case you need to use something that exist in Chipmunk but currently is not 
included in easymunk the easiest method is to add it manually.

For example, lets assume that the is_sleeping property of a body was not 
wrapped by easymunk. The Chipmunk method to get this property is named
cpBodyIsSleeping.

First we need to check if its included in the cdef definition in 
pymunk_extension_build.py. If its not just add it.
    
    `cpBool cpBodyIsSleeping(const cpBody *body);`
    
Then to make it easy to use we want to create a python method that looks nice::

    def is_sleeping(body):
        return cp.cpBodyIsSleeping(body._body)

Now we are ready with the mapping and ready to use our new method.
    

Weak References and free Methods
-----------------------------------

Internally Easymunk allocates structs from Chipmunk (the c library). For example a
Body struct is created from inside the constructor method when a easymunk.Body is
created. Because of this its important that the corresponding c side memory is 
deallocated properly when not needed anymore, usually when the Python side 
object is garbage collected. Most Easymunk objects use `ffi.gc` with a custom
free function to do this. Note that the order of freeing is very important to 
avoid errors.
