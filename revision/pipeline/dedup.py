"""
dedup.py -- cross-tile centroid deduplication (manuscript method).

Manuscript: "we extracted geographic centroids for every delineated crown and indexed them
using a KD-Tree. For each pair of detections within 1.0 m of each other, we retained only one
detection if the two points originated from different tiles, preserving closely spaced trees
within a single tile."

Implementation (deterministic greedy maximal independent set on the cross-tile conflict graph):
  1. KDTree.query_pairs(r) on projected (UTM, metres) centroids.
  2. Keep only pairs whose source tiles differ; same-tile pairs never interact.
  3. Rank every detection by a priority key (default: distance of the centroid from the edge of
     its own source tile, larger = better, because crowns near a tile edge are truncated; ties
     broken by tile id then label, so the result never depends on input order).
  4. Visit detections from highest to lowest priority. A detection is removed only if a
     cross-tile neighbour within r has already been KEPT; otherwise it is kept.

Guarantees (tested in test_dedup.py):
  * no two retained detections from different tiles lie within r of each other;
  * every removed detection has a retained cross-tile partner within r (nothing is dropped
    without a surviving duplicate, so chains cannot cascade);
  * same-tile detections are never removed because of each other;
  * the output is invariant to row order.
"""
import numpy as np
from scipy.spatial import cKDTree


def cross_tile_pairs(xy, tile, r):
    pairs = cKDTree(xy).query_pairs(r, output_type='ndarray')
    if len(pairs) == 0:
        return pairs
    tile = np.asarray(tile)
    return pairs[tile[pairs[:, 0]] != tile[pairs[:, 1]]]


def dedup_cross_tile(xy, tile, priority, tiebreak, r=1.0):
    """Return a boolean keep-mask.

    xy        (n, 2) projected coordinates in metres
    tile      (n,) source tile id (any hashable / str)
    priority  (n,) float, larger = preferred (e.g. distance to own tile edge in metres)
    tiebreak  (n,) str, unique per detection (e.g. f"{tile}|{label:09d}"); resolves equal priority
    """
    xy = np.asarray(xy, dtype=float)
    n = len(xy)
    keep = np.ones(n, dtype=bool)
    pairs = cross_tile_pairs(xy, tile, r)
    if len(pairs) == 0:
        return keep
    involved = np.unique(pairs)
    # adjacency (CSR) over involved nodes only
    a = np.concatenate([pairs[:, 0], pairs[:, 1]])
    b = np.concatenate([pairs[:, 1], pairs[:, 0]])
    o = np.argsort(a, kind='stable')
    a, b = a[o], b[o]
    starts = np.searchsorted(a, involved, 'left')
    ends = np.searchsorted(a, involved, 'right')
    nbrs = {int(v): b[s:e] for v, s, e in zip(involved, starts, ends)}
    # order: priority desc, then tiebreak asc (fully deterministic)
    tb = np.asarray(tiebreak)[involved]
    pr = np.asarray(priority, dtype=float)[involved]
    order = np.lexsort((tb, -pr))
    kept = np.zeros(n, dtype=bool)
    for v in involved[order]:
        if kept[nbrs[int(v)]].any():
            keep[v] = False
        else:
            kept[v] = True
    return keep


def previous_loop(xy, tile, r=1.0, sort=True):
    """The loop used by the previous attempt (sort=True) / prepare_ripley_data.py (sort=False)."""
    pairs = cKDTree(np.asarray(xy, float)).query_pairs(r)
    it = sorted(pairs) if sort else pairs
    to_remove = set()
    for i, j in it:
        if tile[i] != tile[j]:
            if i not in to_remove:
                to_remove.add(j)
    keep = np.ones(len(xy), dtype=bool)
    keep[list(to_remove)] = False
    return keep
