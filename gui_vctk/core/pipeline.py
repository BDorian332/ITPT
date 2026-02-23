from __future__ import annotations

from typing import Any, List

from gui_vctk.core.models import Point, PointType
from gui_vctk.core.settings_state import SETTINGS

_model = None
_model_version = None


def _safe_get(obj: Any, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _guess_scale(points: list[Any], W: int, H: int) -> tuple[float, float]:
    if not points:
        return (1.0, 1.0)

    xs = [float(_safe_get(p, "x", 0.0)) for p in points]
    ys = [float(_safe_get(p, "y", 0.0)) for p in points]
    m = max(max(xs) if xs else 0.0, max(ys) if ys else 0.0)

    if m <= 2.0:        # normalisé
        return (float(W), float(H))
    if m <= 520.0:      # 512
        return (float(W) / 512.0, float(H) / 512.0)
    if m <= 1550.0:     # 1500
        return (float(W) / 1500.0, float(H) / 1500.0)
    return (1.0, 1.0)


def _guess_bbox_scale(texts: list[Any], W: int, H: int) -> tuple[float, float]:
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
    if m <= 520.0:      # 512
        return (float(W) / 512.0, float(H) / 512.0)
    if m <= 1550.0:     # 1500
        return (float(W) / 1500.0, float(H) / 1500.0)
    return (1.0, 1.0)


def _ensure_model_loaded():
    global _model, _model_version

    if _model is not None and _model_version == SETTINGS.version:
        return

    # (re)load model if version changed
    if SETTINGS.version == "v1":
        from itpt._data.models.v1.model import v1 as Model
    else:
        # placeholder: v0 pas encore branché
        from itpt._data.models.v1.model import v1 as Model

    _model = Model()
    _model.load()
    _model_version = SETTINGS.version


def run_pipeline(image_path: str) -> List[Point]:
    _ensure_model_loaded()

    # 1) Load & preprocess
    img_rgb_resized, _img_tensor, (H, W) = _model.load_and_preprocess(image_path)

    # 2) Pre-processing configurable
    data = [img_rgb_resized]

    if SETTINGS.cropping:
        data = _model.extract_tree(data)

    if SETTINGS.denoising:
        data = _model.clean_tree(data)

    # 3) Detect nodes
    nodes_by_image = _model.detect_nodes(data)

    # 4) Detect texts (OCR)
    texts_by_image = _model.detect_texts([img_rgb_resized])

    # 5) Convert nodes/corners -> points GUI
    raw_pts = nodes_by_image[0] if nodes_by_image else []
    sx, sy = _guess_scale(raw_pts, W, H)

    out: List[Point] = []

    for p in raw_pts:
        ptype = _safe_get(p, "type", "node")
        x = float(_safe_get(p, "x", 0.0)) * sx
        y = float(_safe_get(p, "y", 0.0)) * sy

        if ptype == "corner":
            out.append(Point(x, y, PointType.CORNER, None))
        else:
            # root -> node (tu n'utilises plus ROOT côté édition)
            out.append(Point(x, y, PointType.NODE, None))

    # 6) Convert texts -> tips
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

    # 7) Post-processing placeholder (pas encore implémenté)
    if SETTINGS.post_clean:
        pass
    if SETTINGS.post_merge:
        pass

    return out
