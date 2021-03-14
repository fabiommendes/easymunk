"""
Draw module that flips y coordinates to make it consistent with mathematical
convention.
"""

import pyxel


def _flip_y(y):
    return pyxel.height - y - 1


class _FlipY:
    @staticmethod
    def pset(x: int, y: int, col: int):
        return pyxel.pset(x, _flip_y(y), col)

    @staticmethod
    def circ(x: int, y: int, r: int, col: int):
        return pyxel.circ(x, _flip_y(y), r, col)

    @staticmethod
    def circb(x: int, y: int, r: int, col: int):
        return pyxel.circb(x, _flip_y(y), r, col)

    @staticmethod
    def line(x1: int, y1: int, x2: int, y2: int, col: int):
        return pyxel.line(x1, _flip_y(y1), x2, _flip_y(y2), col)

    @staticmethod
    def tri(x1: int, y1: int, x2: int, y2: int, x3: int, y3: int, col: int):
        return pyxel.tri(x1, _flip_y(y1), x2, _flip_y(y2), x3, _flip_y(y3), col)

    @staticmethod
    def trib(x1: int, y1: int, x2: int, y2: int, x3: int, y3: int, col: int):
        return pyxel.trib(x1, _flip_y(y1), x2, _flip_y(y2), x3, _flip_y(y3), col)

    @staticmethod
    def rect(x: int, y: int, w: int, h: int, col: int):
        return pyxel.rect(x, _flip_y(y), w, h, col)

    @staticmethod
    def rectb(x: int, y: int, w: int, h: int, col: int):
        return pyxel.rectb(x, _flip_y(y), w, h, col)


flip_y = _FlipY()
