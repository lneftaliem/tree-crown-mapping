#!/usr/bin/env python3
"""
rebuild_polygons.py -- rebuilds <city>_ripley_polygons_with_income.geojson (the crown polygons with every attribute, as read
by run_analysis.py) from the outline-only polygons in data/crown_polygons/ and the points in data/tree_points/.

The published polygon files carry only tree_id and the crown outline (coordinates rounded to 7 decimal degrees, about 1 cm);
every other attribute is identical to the point of the same tree_id, so it is joined back from the points.

  python rebuild_polygons.py <dataset dir>

<dataset dir> must hold the decompressed <city>_ripley_points_with_income.geojson files; the polygon files are written next
to them. Austin needs about 4 GB of memory (attributes of 2.6 million points).
"""
import os, sys, json, gzip, glob
import pyogrio

HERE = os.path.dirname(os.path.abspath(__file__))
POLY_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'data', 'crown_polygons')
CITIES = ('austin', 'bloomington', 'cupertino', 'surrey')


class Parts:
    """Reads <name>.part00, .part01, ... as one stream (or <name> itself when it is not split)."""
    def __init__(self, stem):
        self.files = sorted(glob.glob(stem + '.part*')) or [stem]
        self.i, self.f = 0, open(self.files[0], 'rb')

    def read(self, n=-1):
        out = b''
        while self.i < len(self.files) and (n < 0 or len(out) < n):
            b = self.f.read(-1 if n < 0 else n - len(out))
            if b:
                out += b
                continue
            self.f.close(); self.i += 1
            if self.i == len(self.files):
                break
            self.f = open(self.files[self.i], 'rb')
        return out


def main(ds):
    for city in CITIES:
        pts = os.path.join(ds, f'{city}_ripley_points_with_income.geojson')
        df = pyogrio.read_dataframe(pts, read_geometry=False)
        fields = list(df.columns)
        attrs = dict(zip(df.tree_id, df.to_dict('records')))
        del df
        out = os.path.join(ds, f'{city}_ripley_polygons_with_income.geojson')
        n = 0
        with gzip.open(Parts(os.path.join(POLY_DIR, f'{city}_crown_polygons.geojson.gz')), 'rt') as f, open(out, 'w') as o:
            o.write('{\n"type": "FeatureCollection",\n"name": "%s_ripley_polygons_with_income",\n'
                    '"crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },\n"features": [\n' % city)
            for line in f:
                s = line.strip().rstrip(',')
                if not s.startswith('{"type":"Feature"'):
                    continue
                ft = json.loads(s)
                a = attrs.pop(ft['properties']['tree_id'])
                props = {k: (None if a[k] != a[k] else a[k]) for k in fields}  # NaN -> null
                o.write((',\n' if n else '') + json.dumps({'type': 'Feature', 'properties': props, 'geometry': ft['geometry']},
                                                          default=lambda v: v.item()))
                n += 1
            o.write('\n]\n}\n')
        assert not attrs, f'{city}: {len(attrs)} points have no polygon'
        print(f'{city}: {n:,} polygons -> {out}')


if __name__ == '__main__':
    main(sys.argv[1])
