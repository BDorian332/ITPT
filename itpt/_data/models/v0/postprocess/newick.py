#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
newick_clean.py

Construit un Newick (avec longueurs de branches RELATIVES) à partir de 3 listes de points :
- leaves    : [(x,y), ...]
- internals : [(x,y), ...]
- corners   : [(x,y), ...]

Hypothèses (comme ton ancien script) :
- arbre vertical
- y sert uniquement à regrouper les enfants via des "intervals" définis par les corners
- x sert à calculer les longueurs de branches (horizontales)

Longueurs :
- length(child->parent) = abs(x_child - x_parent)
  (tu peux ensuite multiplier toutes les longueurs par une constante pour récupérer la temporalité réelle)

Sortie :
- Newick terminé par ';'
"""

from typing import List, Tuple, Optional


def build_newick(
    leaves: List[Tuple[float, float]],
    internals: List[Tuple[float, float]],
    corners: List[Tuple[float, float]],
    *,
    x_tol: float = 18.0,
    y_tol: float = 14.0,
    leaf_names: Optional[List[str]] = None,
    decimals: int = 9
) -> Optional[str]:
    """
    Args:
        leaves, internals, corners: list of (x,y)
        x_tol: tolérance en x pour associer 2 corners à un internal
        y_tol: marge autour de l'intervalle [y_corner1, y_corner2] pour capturer les clusters
        leaf_names: noms des feuilles (sinon A,B,C...)
        decimals: nombre de décimales dans les longueurs

    Returns:
        Newick string avec longueurs, ou None si impossible.
    """
    if not leaves or not internals or not corners:
        return None

    def fmt_len(L: float) -> str:
        return f"{L:.{decimals}f}"

    def branch_len(x_child: float, x_parent: float) -> float:
        return abs(float(x_child) - float(x_parent))

    # --------------------------------------------------------
    # 1) Ordonner et nommer les feuilles
    # --------------------------------------------------------
    leaves_sorted = sorted([(float(x), float(y)) for (x, y) in leaves], key=lambda p: p[1], reverse=True)

    if leaf_names is None:
        leaf_names = []
        for i in range(len(leaves_sorted)):
            leaf_names.append(chr(ord("A") + i) if i < 26 else f"L{i}")

    if len(leaf_names) != len(leaves_sorted):
        raise ValueError("leaf_names length must match number of leaves")

    xs = sorted([x for (x, _y) in leaves_sorted])
    leaf_tip_x = xs[len(xs) // 2]  # médiane

    # clusters = sous-arbres actifs
    # - y : position (pour sélectionner par intervalle vertical)
    # - x : x "du nœud représentant" ce cluster (pour les longueurs)
    # - nw: newick du cluster SANS ';'
    clusters = []
    for name, (_x, y) in zip(leaf_names, leaves_sorted):
        clusters.append({"y": float(y), "x": float(leaf_tip_x), "nw": name})

    # --------------------------------------------------------
    # 2) Préparer corners
    # --------------------------------------------------------
    corners_f = [(float(x), float(y)) for (x, y) in corners]

    def find_two_corners_for_node(xn: float, yn: float):
        cand = [(x, y) for (x, y) in corners_f if abs(x - xn) <= x_tol]
        if len(cand) < 2:
            return None

        above = [c for c in cand if c[1] <= yn]
        below = [c for c in cand if c[1] >= yn]

        if above and below:
            c1 = min(above, key=lambda c: abs(c[1] - yn))
            c2 = min(below, key=lambda c: abs(c[1] - yn))
            if c1 != c2:
                return c1, c2

        cand.sort(key=lambda c: abs(c[1] - yn))
        return cand[0], cand[1]

    # --------------------------------------------------------
    # 3) Fusion bottom-up
    # --------------------------------------------------------
    internals_sorted = sorted([(float(x), float(y)) for (x, y) in internals], key=lambda p: p[0], reverse=True)

    for xn, yn in internals_sorted:
        found = find_two_corners_for_node(xn, yn)
        if found is None:
            continue

        (x1, y1), (x2, y2) = found
        y_low, y_high = (y1, y2) if y1 <= y2 else (y2, y1)

        low = y_low - y_tol
        high = y_high + y_tol

        inside_idx = [i for i, c in enumerate(clusters) if low <= c["y"] <= high]
        if len(inside_idx) < 2:
            continue

        inside = [clusters[i] for i in inside_idx]
        inside.sort(key=lambda c: c["y"], reverse=True)

        parts = []
        for c in inside:
            L = branch_len(c["x"], xn)
            parts.append(f"{c['nw']}:{fmt_len(L)}")

        merged_nw = "(" + ",".join(parts) + ")"

        merged_cluster = {"y": float(yn), "x": float(xn), "nw": merged_nw}

        clusters = [c for j, c in enumerate(clusters) if j not in inside_idx]
        clusters.append(merged_cluster)
        clusters.sort(key=lambda c: c["y"], reverse=True)

    # --------------------------------------------------------
    # 4) Finalisation
    # --------------------------------------------------------
    if len(clusters) == 1:
        return clusters[0]["nw"] + ";"

    root_x = min([x for (x, _y) in internals_sorted] + [x for (x, _y) in corners_f])

    clusters.sort(key=lambda c: c["y"], reverse=True)
    parts = []
    for c in clusters:
        L = branch_len(c["x"], root_x)
        parts.append(f"{c['nw']}:{fmt_len(L)}")

    return "(" + ",".join(parts) + ");"


# ============================================================
# MAIN (TEST)
# ============================================================
def main():
    import json
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in-json", help="Input JSON file (list of dicts)", default="../heatmap/train_dataset/points_dataset.json")
    ap.add_argument("--out-txt", help="Output TXT (1 JSON per line)", default="./newick.txt")
    ap.add_argument("--x-tol", type=float, default=18.0)
    ap.add_argument("--y-tol", type=float, default=14.0)
    ap.add_argument("--decimals", type=int, default=6)
    args = ap.parse_args()

    with open(args.in_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list")

    with open(args.out_txt, "w", encoding="utf-8") as fout:
        for item in data:
            image = item.get("image", None)

            leaves = [(float(x), float(y)) for x, y in item.get("leaf", [])]
            internals = [(float(x), float(y)) for x, y in item.get("node", [])]
            corners = [(float(x), float(y)) for x, y in item.get("corner", [])]

            newick = build_newick(
                leaves,
                internals,
                corners,
                x_tol=args.x_tol,
                y_tol=args.y_tol,
                decimals=args.decimals,
            )

            # one JSON per line
            line = {
                "image": image,
                "newick": newick,
            }

            fout.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(data)} lines to {args.out_txt}")

if __name__ == "__main__":
    main()
