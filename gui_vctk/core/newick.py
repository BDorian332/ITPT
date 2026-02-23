from __future__ import annotations

from typing import List, Dict, Any

from gui_vctk.core.models import Point as GuiPoint, PointType as GuiPointType

# ITPT newick
from itpt.core import Point as ItptPoint, build_newick


def _to_itpt_nodes_and_corners(gui_points: List[GuiPoint]) -> tuple[List[ItptPoint], float]:
    """
    Convertit GUI -> ITPT points ("node"/"corner") et calcule x_leave (max X).
    """
    out: List[ItptPoint] = []
    xs: List[float] = []

    for p in gui_points:
        if p.ptype == GuiPointType.CORNER:
            out.append(ItptPoint(float(p.x), float(p.y), "corner"))
            xs.append(float(p.x))
        elif p.ptype in (GuiPointType.NODE, GuiPointType.ROOT):
            out.append(ItptPoint(float(p.x), float(p.y), "node"))
            xs.append(float(p.x))

    x_leave = max(xs) if xs else 0.0
    return out, x_leave


def _tips_to_texts_aligned(gui_points: List[GuiPoint], x_leave: float) -> List[Dict[str, Any]]:
    """
    IMPORTANT: ITPT associe les labels via get_nearest_label(x_leave, y).
    Donc on place les bbox sur x_leave et on encode le y du tip.
    """
    texts: List[Dict[str, Any]] = []

    for tip in gui_points:
        if tip.ptype != GuiPointType.TIP:
            continue

        label = (tip.label or "").strip()
        if not label:
            # Si vide, on met un nom par défaut propre
            label = "leaf"

        y = float(tip.y)
        texts.append({
            "text": label,
            "bbox": [float(x_leave), y - 1.0, float(x_leave), y + 1.0],
        })

    return texts


def compute_newick(gui_points: List[GuiPoint]) -> str:
    if not gui_points:
        return "();"

    pts_itpt, x_leave = _to_itpt_nodes_and_corners(gui_points)
    if not pts_itpt:
        return "();"

    texts = _tips_to_texts_aligned(gui_points, x_leave)

    # Paramètres en pixels (tu peux ajuster si besoin)
    margin_px = 12.0
    max_distance_px = 80.0  # ↑ un peu plus grand pour être safe

    newick_obj = build_newick(
        pts_itpt,
        margin=margin_px,
        texts=texts,
        max_distance=max_distance_px,
        scale_width=1.0,
        scale_height=1.0,
        verbose=False,
    )

    return newick_obj.to_string() if newick_obj else "();"
