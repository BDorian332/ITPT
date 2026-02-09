import numpy as np
from typing import List, Optional, Tuple
from numpy.typing import NDArray
from .newick import Point, get_nearest_point, reset_points

Segment = Tuple[Tuple[float, float], Tuple[float, float]]

def process_no_root_node(
    node: Point,
    points: List[Point],
    direction: str,
    x_leave: float,
    margin: float,
    depth: int = 0
) -> List[Segment]:
    if node.processed:
        print("    " * depth + f"Skipping node: {node.to_string()} (already processed)")
        return []
    node.processed = True
    print("    " * depth + f"Processing node: {node.to_string()} in direction {direction}")

    results: List[Segment] = []

    if node.type != "corner":
        next_pt = get_nearest_point(node.x, node.y, points, direction, margin)
        if next_pt:
            print("    " * (depth + 1) + f"Found next point {next_pt.to_string()} in direction {direction}")
            sub = process_no_root_node(next_pt, points, direction, x_leave, margin, depth=depth + 1)
            if sub:
                results.append(((node.x, node.y), (next_pt.x, next_pt.y)))
                results.extend(sub)

    right_pt = get_nearest_point(node.x, node.y, points, "right", margin)
    if right_pt:
        next_len = abs(right_pt.x - node.x)
        print("    " * (depth + 1) + f"Propagating right to {right_pt.to_string()} with length {next_len}")
        root_sub = process_root_node(right_pt, points, x_leave, margin, next_len, depth=depth + 1)
        if len(root_sub) == 0:
            pass
        elif len(root_sub) == 1:
            results.append(((node.x, node.y), (x_leave, node.y)))
        else:
            results.append(((node.x, node.y), (right_pt.x, right_pt.y)))
            results.extend(root_sub)
    else:
        print("    " * (depth + 1) + f"No point to the right, creating leaf at x={x_leave}")
        results.append(((node.x, node.y), (x_leave, node.y)))

    return results

def process_root_node(
    node: Point,
    points: List[Point],
    x_leave: float,
    margin: float,
    incoming_length: float = 0.0,
    depth: int = 0
) -> List[Segment]:
    if node.processed:
        print("    " * depth + f"Skipping node: {node.to_string()} (already processed)")
        return []
    node.processed = True
    print("    " * depth + f"Processing root node: {node.to_string()} with incoming length {incoming_length}")

    if abs(node.x - x_leave) <= margin:
        print("    " * depth + f"Node near x_leave, creating leaf: {node.to_string()} with length {incoming_length}")
        return [((0, 0), (0, 0))]

    down_pt = get_nearest_point(node.x, node.y, points, "down", margin)
    up_pt = get_nearest_point(node.x, node.y, points, "up", margin)

    down_tree = process_no_root_node(down_pt, points, "down", x_leave, margin, depth=depth + 1) if down_pt else []
    up_tree = process_no_root_node(up_pt, points, "up", x_leave, margin, depth=depth + 1) if up_pt else []

    results: List[Segment] = []

    if not down_tree and not up_tree:
        print("    " * (depth + 1) + f"No up/down branches, creating leaf with length {incoming_length + abs(x_leave - node.x)}")
        results.append(((0, 0), (0, 0)))
        return results

    if bool(down_tree) ^ bool(up_tree):
        kept = down_tree if down_tree else up_tree
        results.extend(kept)

        kept_pt = down_pt if down_tree else up_pt
        dy = kept_pt.y - node.y
        sym_y = node.y - dy
        print("    " * (depth + 1) + f"Missing one corner, creating symmetric corner at y={sym_y}")
        sym_tree = process_no_root_node(
            Point(kept_pt.x, sym_y, "corner"),
            points,
            "up" if kept_pt == down_pt else "down",
            x_leave,
            margin,
            depth=depth + 1
        )
        results.append(((node.x, node.y), (kept_pt.x, kept_pt.y)))
        results.append(((node.x, node.y), (kept_pt.x, sym_y)))
        results.extend(sym_tree)
    else:
        results.append(((node.x, node.y), (down_pt.x, down_pt.y)))
        results.append(((node.x, node.y), (up_pt.x, up_pt.y)))
        results.extend(down_tree)
        results.extend(up_tree)

    right_pt = get_nearest_point(node.x, node.y, points, "right", margin)
    if right_pt:
        next_len = abs(right_pt.x - node.x)
        print("    " * (depth + 1) + f"Propagating right to {right_pt.to_string()} with length {next_len}")
        root_sub = process_root_node(right_pt, points, x_leave, margin, next_len, depth=depth + 1)
        if len(root_sub) == 0:
            pass
        elif len(root_sub) == 1:
            results.append(((node.x, node.y), (x_leave, node.y)))
        else:
            results.append(((node.x, node.y), (right_pt.x, right_pt.y)))
            results.extend(root_sub)

    return results

def build_segments_from_points(points: List[Point], margin: float = 0.5) -> Optional[List[Segment]]:
    print("Resetting points...")
    reset_points(points)

    if not points:
        print("No points provided.")
        return None

    min_x = min(p.x for p in points)
    print(f"Minimal x found: {min_x}")

    start_candidates = [p for p in points if abs(p.x - min_x) <= margin]
    print(f"Found {len(start_candidates)} start candidates within margin {margin}")

    start_point = min(start_candidates, key=lambda p: p.y)
    start_point.type = "node"
    print(f"Start point chosen (lowest y among min_x): {start_point.to_string()}")

    x_leave = max(p.x for p in points)
    print(f"x_leave set to: {x_leave}")

    segments = process_no_root_node(start_point, points, "up", x_leave, margin)
    print("Segments built.")

    return segments

def build_segments(
    nodes: List[Tuple],
    corners: List[Tuple] = [],
    scale: float = 1500.0,
    margin: float = 0.5,
) -> Optional[List[Segment]]:
    points: List[Point] = []

    for t in nodes:
        if len(t) < 2:
            continue
        x, y = float(t[0]) * scale, float(t[1]) * scale
        points.append(Point(x, y))

    for t in corners:
        if len(t) < 2:
            continue
        x, y = float(t[0]) * scale, float(t[1]) * scale
        points.append(Point(x, y, "corner"))

    return build_segments_from_points(points, margin=margin)

def normalize_point(p: Point, width: float, height: float) -> Point:
    return Point(p.x / width, p.y / height, p.type)

def normalize_segments(
    segments: List[Segment],
    width: float,
    height: float
) -> List[Segment]:
    normalized: List[Segment] = []

    for (x1, y1), (x2, y2) in segments:
        p1 = normalize_point(Point(x1, y1), width, height)
        p2 = normalize_point(Point(x2, y2), width, height)
        normalized.append(((p1.x, p1.y), (p2.x, p2.y)))

    return normalized

def draw_segment(
    heatmap: NDArray[np.floating],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    intensity: float,
    thickness: int
):
    length = int(max(abs(x2 - x1), abs(y2 - y1))) + 1

    xs = np.linspace(x1, x2, length).astype(int)
    ys = np.linspace(y1, y2, length).astype(int)

    h, w = heatmap.shape
    half_thick = thickness // 2

    for x, y in zip(xs, ys):
        for dx in range(-half_thick, half_thick + 1):
            for dy in range(-half_thick, half_thick + 1):
                xi = x + dx
                yi = y + dy
                if 0 <= xi < w and 0 <= yi < h:
                    heatmap[yi, xi] = min(1.0, heatmap[yi, xi] + intensity)

def gaussian_blur(
    img: NDArray[np.floating],
    sigma: float
):
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    img = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, img)
    img = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 1, img)

    return img

def segments_to_heatmap(
    segments: List[Segment],
    scale_x: int = 1500,
    scale_y: int = 1500,
    draw_intensity: float = 1.0,
    draw_thickness: int = 1,
    blur_sigma: float = 2.0
):
    heatmap = np.zeros((scale_y, scale_x), dtype=np.float32)

    for (x1, y1), (x2, y2) in segments:
        px1 = int(x1 * scale_x)
        py1 = int(y1 * scale_y)
        px2 = int(x2 * scale_x)
        py2 = int(y2 * scale_y)

        draw_segment(heatmap, px1, py1, px2, py2, draw_intensity, draw_thickness)

    heatmap = gaussian_blur(heatmap, blur_sigma)

    if heatmap.max() > 1:
        heatmap /= heatmap.max()

    return heatmap
