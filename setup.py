from setuptools import setup  # type: ignore

# todo: add/remove/think about this list
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Games/Entertainment",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: pygame",
    "Programming Language :: Python :: 3",
]

with (open("README.rst")) as f:
    long_description = f.read()

setup(
    name="easymunk",
    url="http://fabiommendes.github.io/easymunk/",
    author="Fábio Mendes",
    author_email="fabiomacedomendes@gmail.com",
    version="1.0.0",
    description="Easymunk is a easy-to-use pythonic 2d physics library",
    long_description=long_description,
    packages=["easymunk", "easymunk.hypothesis"],
    include_package_data=True,
    license="MIT License",
    classifiers=classifiers,
    command_options={
        "build_sphinx": {
            "build_dir": ("setup.py", "docs"),
            "source_dir": ("setup.py", "docs/src"),
        }
    },
    python_requires=">=3.7",
    # Require >1.14.0 since that (and older) has problem with returning structs
    # from functions.
    setup_requires=["cffi > 1.14.0"],
    install_requires=["cffi > 1.14.0", "sidekick"],
    cffi_modules=["easymunk/pymunk_extension_build.py:ffibuilder"],
    extras_require={
        "dev": ["pyglet", "pygame", "sphinx", "aafigure", "wheel", "matplotlib"]
    },
    test_suite="tests",
)
