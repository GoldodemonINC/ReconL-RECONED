"""Minimal PNG reader plus the comparisons that say *how* two frames differ.

No PIL on this machine, so this decodes RGBA8 PNGs directly: parse the chunks,
inflate the IDAT stream, then undo the per-row filters. Enough for the benches'
own output.
"""

import struct
import sys
import zlib


def read_png(path):
    data = open(path, 'rb').read()
    assert data[:8] == b'\x89PNG\r\n\x1a\n', 'not a png'
    pos = 8
    idat = b''
    width = height = bit_depth = color_type = 0
    while pos < len(data):
        (length,) = struct.unpack('>I', data[pos:pos + 4])
        kind = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if kind == b'IHDR':
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack('>IIBBBBB', body)
            assert bit_depth == 8 and color_type == 6 and interlace == 0, 'needs RGBA8, non-interlaced'
        elif kind == b'IDAT':
            idat += body
        elif kind == b'IEND':
            break
    raw = zlib.decompress(idat)
    stride = width * 4
    rows = []
    prev = bytearray(stride)
    at = 0
    for _ in range(height):
        filt = raw[at]
        line = bytearray(raw[at + 1:at + 1 + stride])
        at += 1 + stride
        if filt == 1:
            for i in range(4, stride):
                line[i] = (line[i] + line[i - 4]) & 0xFF
        elif filt == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif filt == 3:
            for i in range(stride):
                left = line[i - 4] if i >= 4 else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif filt == 4:
            for i in range(stride):
                a = line[i - 4] if i >= 4 else 0
                b = prev[i]
                c = prev[i - 4] if i >= 4 else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pred) & 0xFF
        rows.append([tuple(line[x * 4:x * 4 + 4]) for x in range(width)])
        prev = line
    return width, height, rows


def flatten(rows):
    return [px for row in rows for px in row]


def mean(px):
    n = len(px)
    return tuple(round(sum(p[i] for p in px) / n, 1) for i in range(4))


def compare(a, b):
    differing = sum(1 for x, y in zip(a, b) if x != y)
    worst = max((max(abs(x[i] - y[i]) for i in range(4)) for x, y in zip(a, b)), default=0)
    return differing, len(a), worst


def main():
    golden_path, candidate_path = sys.argv[1], sys.argv[2]
    gw, gh, grow = read_png(golden_path)
    cw, ch, crow = read_png(candidate_path)
    print(f'golden    {gw}x{gh}  mean {mean(flatten(grow))}')
    print(f'candidate {cw}x{ch}  mean {mean(flatten(crow))}')
    if (gw, gh) != (cw, ch):
        print('size mismatch')
        return

    g = flatten(grow)
    c = flatten(crow)
    d, n, w = compare(g, c)
    print(f'as-is      : {d}/{n} differ, worst channel delta {w}')

    flipped_rows = list(reversed(grow))
    f = flatten(flipped_rows)
    d, n, w = compare(f, c)
    print(f'v-flip     : {d}/{n} differ, worst channel delta {w}')

    mirrored = [list(reversed(row)) for row in grow]
    m = flatten(mirrored)
    d, n, w = compare(m, c)
    print(f'h-mirror   : {d}/{n} differ, worst channel delta {w}')

    print('\nrow   golden(R,G,B)      candidate(R,G,B)')
    step = max(1, gh // 8)
    for y in range(0, gh, step):
        ga = tuple(round(sum(grow[y][x][i] for x in range(gw)) / gw) for i in range(3))
        ca = tuple(round(sum(crow[y][x][i] for x in range(cw)) / cw) for i in range(3))
        print(f'{y:4d}  {str(ga):18s}  {str(ca)}')


if __name__ == '__main__':
    main()
