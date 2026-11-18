"""Directed road graph of one trip with bounded shortest paths (Dijkstra with a cutoff).

Transitions in the HMM need the network distance between two candidate positions
a few metres apart, thousands of times per trip. A full single-source Dijkstra on a
300k-node network per query would be far too slow, so searches stop at a distance
cutoff (only a few hundred metres are ever relevant at 1 Hz) and results are cached
per source node.
"""
from __future__ import annotations

import heapq
import math

import numpy as np
import pandas as pd


class RoadGraph:
    def __init__(self, nodes: pd.DataFrame, arcs: pd.DataFrame):
        nx, ny = nodes["x"].to_numpy(), nodes["y"].to_numpy()
        self.u = arcs["u"].to_numpy()
        self.v = arcs["v"].to_numpy()
        self.length = np.hypot(nx[self.v] - nx[self.u], ny[self.v] - ny[self.u])
        n = len(nodes)
        order = np.argsort(self.u, kind="stable")              # CSR of outgoing arcs per node
        self._out_arcs = order
        self._start = np.searchsorted(self.u[order], np.arange(n + 1))
        self._cache: dict[int, tuple[float, dict, dict]] = {}

    def _search(self, src: int, cutoff: float) -> tuple[dict, dict]:
        """dist[node] and the last arc used to reach it, for nodes within `cutoff` of src."""
        hit = self._cache.get(src)
        if hit is not None and hit[0] >= cutoff:
            return hit[1], hit[2]
        dist = {src: 0.0}
        via = {src: -1}
        heap = [(0.0, src)]
        out, start, v, length = self._out_arcs, self._start, self.v, self.length
        while heap:
            d, node = heapq.heappop(heap)
            if d > dist.get(node, math.inf):
                continue
            for k in range(start[node], start[node + 1]):
                a = out[k]
                nd = d + length[a]
                if nd <= cutoff and nd < dist.get(v[a], math.inf):
                    dist[v[a]] = nd
                    via[v[a]] = a
                    heapq.heappush(heap, (nd, v[a]))
        self._cache[src] = (cutoff, dist, via)
        return dist, via

    def node_distance(self, src: int, dst: int, cutoff: float) -> float:
        dist, _ = self._search(src, cutoff)
        return dist.get(dst, math.inf)

    def route_distance(self, a: int, ta: float, b: int, tb: float, cutoff: float,
                       backtrack_tol: float = 5.0) -> float:
        """Driving distance from fraction ta along arc a to fraction tb along arc b
        (inf if longer than cutoff). Small backwards moves on the same arc (GPS jitter,
        < backtrack_tol metres) count as their absolute length."""
        if a == b:
            d = (tb - ta) * self.length[a]
            if d >= -backtrack_tol:
                return abs(d)
        head = (1.0 - ta) * self.length[a]
        tail = tb * self.length[b]
        budget = cutoff - head - tail
        if budget < 0:
            return math.inf
        return head + self.node_distance(int(self.v[a]), int(self.u[b]), budget) + tail

    def path_arcs(self, a: int, b: int, cutoff: float) -> list[int] | None:
        """Arcs strictly between arc a and arc b on the shortest path (None if no path)."""
        if a == b or self.v[a] == self.u[b]:
            return []
        dist, via = self._search(int(self.v[a]), cutoff)
        node = int(self.u[b])
        if node not in dist:
            return None
        arcs = []
        while via[node] != -1:
            arcs.append(int(via[node]))
            node = int(self.u[via[node]])
        return arcs[::-1]
