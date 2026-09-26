"""The few C++ standard-library behaviours the port reproduces on purpose.

The held-out split draws its ties from std::mt19937_64, so the port carries that generator (the
published algorithm; numpy has only the 32-bit twister). std::sort and std::unique are here as
names so the driver reads like the C++; every comparator the port sorts with is a total order,
so Python's sort gives the same arrangement. The rounding helpers name the C casts they stand
for. Nothing here binds a C library: the arithmetic comes from math and numpy, and the port
agrees with the binary to a tolerance, not bit for bit.
"""
from __future__ import annotations

from functools import cmp_to_key
import math

fabs = math.fabs
sqrt = math.sqrt
isfinite = math.isfinite


def clamp(x, lo, hi):
    """std::clamp: returns lo if x < lo, hi if hi < x, else x (NaN passes through as x)."""
    if x < lo:
        return lo
    if hi < x:
        return hi
    return x


def to_int(x: float) -> int:
    """static_cast<int>(double): truncation toward zero."""
    return int(x)


class MT19937_64:
    """std::mt19937_64, seeded as std::mt19937_64(seed)."""
    MASK = 0xFFFFFFFFFFFFFFFF

    def __init__(self, seed: int = 5489):
        self.mt = [0] * 312
        self.mt[0] = seed & self.MASK
        for i in range(1, 312):
            self.mt[i] = (6364136223846793005 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 62)) + i) & self.MASK
        self.index = 312

    def _twist(self):
        mt = self.mt
        for i in range(312):
            y = (mt[i] & 0xFFFFFFFF80000000) | (mt[(i + 1) % 312] & 0x7FFFFFFF)
            v = mt[(i + 156) % 312] ^ (y >> 1)
            if y & 1:
                v ^= 0xB5026F5AA96619E9
            mt[i] = v
        self.index = 0

    def __call__(self) -> int:
        if self.index >= 312:
            self._twist()
        y = self.mt[self.index]
        self.index += 1
        y ^= (y >> 29) & 0x5555555555555555
        y ^= (y << 17) & 0x71D67FFFEDA60000
        y ^= (y << 37) & 0xFFF7EEE000000000
        y ^= y >> 43
        return y & self.MASK


def std_sort(a: list, less) -> None:
    """Sort in place under the C++ comparator `less(a, b)` (a total order everywhere it is used)."""
    a.sort(key=cmp_to_key(lambda x, y: -1 if less(x, y) else (1 if less(y, x) else 0)))


def std_unique(a: list) -> list:
    """std::unique followed by erase: adjacent equal elements collapsed (identity equality)."""
    out = []
    for x in a:
        if not out or out[-1] is not x:
            out.append(x)
    return out


def g10(x) -> str:
    """An ostream with setprecision(10) in the default float field: printf("%.10g")."""
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, int):
        return str(x)
    if x != x:
        return "nan" if math.copysign(1.0, x) > 0 else "-nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return "%.10g" % x
