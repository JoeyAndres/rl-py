import struct

from rl.utility import concatArray, floatArrayToBitset


def _bits(f):
    (i,) = struct.unpack("<Q", struct.pack("<d", f))
    return format(i, "064b")


def test_float_array_to_bitset():
    floatArray = [0.1, 0.2, 0.3]
    expected = "".join(_bits(f) for f in floatArray)
    assert floatArrayToBitset(floatArray).to_string() == expected


def test_concat_array():
    assert concatArray([1, 2, 3], [4, 5]) == [1, 2, 3, 4, 5]
