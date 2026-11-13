"""Read-only COL1 extraction for entrance hypotheses; never operates the game.

Layout follows reVC-src/src/core/FileLoader.cpp::LoadCollisionModel.
Only the two identity-rotated nbeachw instances are converted to world space.
This is geometry evidence, not a substitute for native loaded-world queries.
"""
from pathlib import Path
import hashlib
import json
import struct


ROOT = Path(__file__).resolve().parent
GAME = Path('C:/VC')
COL = GAME / 'data/maps/nbeachw/nbeachw.col'
IPL = GAME / 'data/maps/nbeachw/nbeachw.ipl'
NAMES = {'buildingsite2': 3830, 'bldngst2mesh': 3908}


def parse_model(data):
    bounds = struct.unpack_from('<10f', data)
    spheres = []
    count = struct.unpack_from('<h', data, 40)[0]
    cursor = 44
    for _ in range(count):
        spheres.append(struct.unpack_from('<4f4B', data, cursor))
        cursor += 20
    line_count = struct.unpack_from('<h', data, cursor)[0]
    cursor += 4 + line_count * 24
    count = struct.unpack_from('<h', data, cursor)[0]
    cursor += 4
    boxes = []
    for _ in range(count):
        boxes.append(struct.unpack_from('<6f4B', data, cursor))
        cursor += 28
    count = struct.unpack_from('<h', data, cursor)[0]
    cursor += 4
    vertices = []
    for _ in range(count):
        vertices.append(struct.unpack_from('<3f', data, cursor))
        cursor += 12
    count = struct.unpack_from('<h', data, cursor)[0]
    cursor += 4
    triangles = []
    for _ in range(count):
        triangles.append(struct.unpack_from('<3i4B', data, cursor))
        cursor += 16
    assert cursor == len(data), (cursor, len(data))
    assert all(0 <= index < len(vertices) for t in triangles for index in t[:3])
    return dict(bounds=bounds, spheres=spheres, ignored_engine_lines=line_count,
                boxes=boxes, vertices=vertices, triangles=triangles)


def main():
    raw = COL.read_bytes()
    evidence = dict(source=str(COL), sha256=hashlib.sha256(raw).hexdigest(),
                    loader='FileLoader.cpp::LoadCollisionModel (COLL format)',
                    native_validation_required=True, models={})
    for line_number, line in enumerate(IPL.read_text().splitlines(), 1):
        fields = [p.strip() for p in line.split(',')]
        if len(fields) == 13 and fields[1] in NAMES:
            assert int(fields[0]) == NAMES[fields[1]]
            assert tuple(map(float, fields[6:])) == (1., 1., 1., 0., 0., 0., 1.)
            evidence['models'][fields[1]] = dict(id=int(fields[0]), ipl_line=line_number,
                                                translation=tuple(map(float, fields[3:6])))
    cursor = 0
    while cursor < len(raw):
        magic, size = struct.unpack_from('<4sI', raw, cursor)
        assert magic == b'COLL'
        name = raw[cursor+8:cursor+32].split(b'\0')[0].decode()
        if name in NAMES:
            parsed = parse_model(raw[cursor+32:cursor+8+size])
            item = evidence['models'][name]
            item.update(parsed, col_offset=cursor, col_record_size=size + 8)
            shift = item['translation']
            item['world_vertices'] = [[v[i] + shift[i] for i in range(3)] for v in item['vertices']]
            item['world_boxes'] = [[b[i] + shift[i % 3] for i in range(6)] + list(b[6:])
                                   for b in item['boxes']]
        cursor += size + 8
    assert cursor == len(raw)
    index = (GAME / 'models/gta3.dir').read_bytes()
    for offset in range(0, len(index), 32):
        sector, sectors, name = struct.unpack_from('<II24s', index, offset)
        if name.split(b'\0')[0].lower() == b'nbeachw.col':
            with (GAME / 'models/gta3.img').open('rb') as stream:
                stream.seek(sector * 2048)
                stored = stream.read(sectors * 2048)
            assert stored[:len(raw)] == raw
            evidence['archive'] = dict(sector=sector, sectors=sectors, identical_prefix=True)
            break
    assert 'archive' in evidence
    evidence['candidate_first_floor_route'] = [
        [175., -380., 12.], [117.7, -380., 12.], [117.7, -367., 12.],
        [117.7, -354., 19.5], [117.7, -343.5, 20.35],
        [156.55, -343.5, 20.35], [156.55, -348.03, 20.35]]
    evidence['limitations'] = [
        'Candidate only: every segment must pass current native full-radius query.',
        'Other IPL models, mission objects, current collision residency and vehicle dynamics are not modeled here.',
        'Raw COL floats retained; engine may quantize triangle vertices to 1/128m under COMPRESSED_COL_VECTORS.',
        'An open floor aperture contains a sloped stair and central divider; it is not a free vertical shaft.']
    path = ROOT / 'building-collision-evidence.json'
    path.write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(path)
    print(json.dumps({name: dict(vertices=len(m['vertices']), triangles=len(m['triangles']),
                                boxes=len(m['boxes'])) for name, m in evidence['models'].items()}))


if __name__ == '__main__':
    main()
