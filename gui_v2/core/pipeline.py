from __future__ import annotations

from typing import Any, List

from gui_v2.core.models import Point, PointType

_model = None


def _safe_get(obj: Any, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _guess_scale(points: list[Any], W: int, H: int) -> tuple[float, float]:
    """
    Devine l'échelle des coordonnées renvoyées par la V1 et calcule (sx, sy)
    pour convertir vers pixels de l'image originale (W,H).
    Cas gérés :
      - coords normalisées [0..1]  -> sx=W,   sy=H
      - coords en 512 (cleaned)    -> sx=W/512, sy=H/512
      - coords en 1500 (preproc)   -> sx=W/1500, sy=H/1500
    """
    if not points:
        return (1.0, 1.0)

    xs = [float(_safe_get(p, "x", 0.0)) for p in points]
    ys = [float(_safe_get(p, "y", 0.0)) for p in points]
    max_x = max(xs) if xs else 0.0
    max_y = max(ys) if ys else 0.0
    m = max(max_x, max_y)

    # 1) Normalisé 0..1
    if m <= 2.0:
        return (float(W), float(H))

    # 2) Image cleaned 512
    if m <= 520.0:
        return (float(W) / 512.0, float(H) / 512.0)

    # 3) Image preproc 1500
    if m <= 1550.0:
        return (float(W) / 1500.0, float(H) / 1500.0)

    # fallback (déjà en pixels)
    return (1.0, 1.0)

def _guess_bbox_scale(texts: list[Any], W: int, H: int) -> tuple[float, float]:
    """
    Devine l'échelle des bbox OCR et retourne (sx, sy) pour convertir vers pixels (W,H).
    bbox peut être :
        - normalisée [0..1]
        - en 512
        - en 1500
        - déjà en pixels
    """
    if not texts:
        return (1.0, 1.0)

    vals: list[float] = []
    for t in texts:
        bbox = _safe_get(t, "bbox", None)
        if bbox and len(bbox) >= 4:
            vals.extend([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])

    if not vals:
        return (1.0, 1.0)

    m = max(vals)

    if m <= 2.0:        # normalisé
        return (float(W), float(H))
    if m <= 520.0:      # 512-space
        return (float(W) / 512.0, float(H) / 512.0)
    if m <= 1550.0:     # 1500-space
        return (float(W) / 1500.0, float(H) / 1500.0)

    return (1.0, 1.0)   # déjà en pixels




def run_pipeline(image_path: str) -> List[Point]:
    global _model

    if _model is None:
        from itpt._data.models.v1.model import v1 as V1Model
        _model = V1Model()
        _model.load()

    # 1) Load & preprocess
    img_rgb_resized, _img_tensor, (H, W) = _model.load_and_preprocess(image_path)

    # 2) Crop -> Clean -> Nodes
    cropped = _model.extract_tree([img_rgb_resized])
    cleaned = _model.clean_tree(cropped)
    nodes_by_image = _model.detect_nodes(cleaned)

    # 3) OCR texts (souvent 0 chez toi, mais on garde)
    texts_by_image = _model.detect_texts([img_rgb_resized])

    raw_pts = nodes_by_image[0] if nodes_by_image else []
    sx, sy = _guess_scale(raw_pts, W, H)

    out: List[Point] = []

    # --- Nodes / corners
    for p in raw_pts:
        ptype = _safe_get(p, "type", "node")
        x = float(_safe_get(p, "x", 0.0)) * sx
        y = float(_safe_get(p, "y", 0.0)) * sy

        if ptype == "corner":
            out.append(Point(x, y, PointType.CORNER, None))
        elif ptype == "root":
            out.append(Point(x, y, PointType.ROOT, None))
        else:
            out.append(Point(x, y, PointType.NODE, None))

        # --- Texts -> tips
    texts = texts_by_image[0] if texts_by_image else []
    tsx, tsy = _guess_bbox_scale(texts, W, H)

    for t in texts:
        bbox = _safe_get(t, "bbox", None)
        txt = (_safe_get(t, "text", "") or "").strip()

        if not bbox or len(bbox) < 4:
            continue

        x1, y1, x2, y2 = map(float, bbox[:4])

        cx = 0.5 * (x1 + x2) * tsx
        cy = 0.5 * (y1 + y2) * tsy

        out.append(Point(cx, cy, PointType.TIP, txt if txt else None))

    return out
