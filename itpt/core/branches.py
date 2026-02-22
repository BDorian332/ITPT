import numpy as np
from typing import List, Optional, Tuple
from numpy.typing import NDArray
from itpt.core.newick import Point, get_nearest_point, reset_points, scale_points

Segment = Tuple[Tuple[float, float], Tuple[float, float]]

def process_no_root_node(
    node: Point,
    points: List[Point],
    direction: str,
    x_leave: float,
    margin: float,
    depth: int = 0,
    verbose: bool = False
) -> List[Segment]:
    if node.processed:
        if verbose:
            print("    " * depth + f"Skipping node: {node.to_string()} (already processed)")
        return []

    node.processed = True
    if verbose:
        print("    " * depth + f"Processing node: {node.to_string()} in direction {direction}")

    results: List[Segment] = []

    if node.type != "corner":
        next_pt = get_nearest_point(node.x, node.y, points, direction, margin)
        if next_pt:
            if verbose:
                print("    " * (depth + 1) + f"Found next point {next_pt.to_string()} in direction {direction}")

            sub = process_no_root_node(
                next_pt, points, direction, x_leave, margin,
                depth=depth + 1, verbose=verbose
            )

            if sub:
                results.append(((node.x, node.y), (next_pt.x, next_pt.y)))
                results.extend(sub)

    right_pt = get_nearest_point(node.x, node.y, points, "right", margin)

    if right_pt:
        if verbose:
            print("    " * (depth + 1) + f"Propagating right to {right_pt.to_string()}")

        root_sub = process_root_node(right_pt, points, x_leave, margin, depth=depth + 1, verbose=verbose)

        if len(root_sub) == 1:
            results.append(((node.x, node.y), (x_leave, node.y)))
        elif len(root_sub) > 1:
            results.append(((node.x, node.y), (right_pt.x, right_pt.y)))
            results.extend(root_sub)

    else:
        if verbose:
            print("    " * (depth + 1) + f"No point to the right, creating leaf at x={x_leave}, y={node.y}")

        results.append(((node.x, node.y), (x_leave, node.y)))

    return results

def process_root_node(
    node: Point,
    points: List[Point],
    x_leave: float,
    margin: float,
    depth: int = 0,
    verbose: bool = False
) -> List[Segment]:

    if node.processed:
        if verbose:
            print("    " * depth + f"Skipping node: {node.to_string()} (already processed)")
        return []

    node.processed = True

    if verbose:
        print("    " * depth + f"Processing root node: {node.to_string()}")

    if abs(node.x - x_leave) <= margin:
        if verbose:
            print("    " * depth + f"Node near x_leave, creating leaf")

        return [((0, 0), (0, 0))]

    down_pt = get_nearest_point(node.x, node.y, points, "down", margin)
    up_pt = get_nearest_point(node.x, node.y, points, "up", margin)

    down_tree = process_no_root_node(
        down_pt, points, "down", x_leave, margin,
        depth=depth + 1, verbose=verbose
    ) if down_pt else []

    up_tree = process_no_root_node(
        up_pt, points, "up", x_leave, margin,
        depth=depth + 1, verbose=verbose
    ) if up_pt else []

    results: List[Segment] = []

    if not down_tree and not up_tree:
        if verbose:
            print("    " * (depth + 1) + "No up/down branches, creating leaf")

        results.append(((0, 0), (0, 0)))
        return results

    if bool(down_tree) ^ bool(up_tree):
        kept = down_tree if down_tree else up_tree
        results.extend(kept)

        kept_pt = down_pt if down_tree else up_pt
        dy = kept_pt.y - node.y
        sym_y = node.y - dy

        if verbose:
            print("    " * (depth + 1) + f"Missing one corner, creating symmetric corner at y={sym_y}")

        sym_tree = process_no_root_node(
            Point(kept_pt.x, sym_y, "corner"),
            points,
            "up" if kept_pt == down_pt else "down",
            x_leave,
            margin,
            depth=depth + 1,
            verbose=verbose
        )

        if not sym_tree:

            if verbose:
                print("    " * (depth + 1) + f"Symmetric branch is empty, creating leaf at y={sym_y}")

            results.append(((node.x, sym_y), (x_leave, sym_y)))
        else:
            results.extend(sym_tree)

        results.append(((node.x, node.y), (kept_pt.x, kept_pt.y)))
        results.append(((node.x, node.y), (kept_pt.x, sym_y)))
    else:
        results.append(((node.x, node.y), (down_pt.x, down_pt.y)))
        results.append(((node.x, node.y), (up_pt.x, up_pt.y)))
        results.extend(down_tree)
        results.extend(up_tree)

    right_pt = get_nearest_point(node.x, node.y, points, "right", margin)
    if right_pt:
        if verbose:
            print("    " * (depth + 1) + f"Propagating right to {right_pt.to_string()}")

        root_sub = process_root_node(right_pt, points, x_leave, margin, depth=depth + 1, verbose=verbose)
        if len(root_sub) == 1:
            results.append(((node.x, node.y), (x_leave, node.y)))
        elif len(root_sub) > 1:
            results.append(((node.x, node.y), (right_pt.x, right_pt.y)))
            results.extend(root_sub)

    return results

def build_segments(
    points: List[Point],
    margin: float = 5,
    scale_width: float = 1500,
    scale_height: float = 1500,
    verbose: bool = False
) -> Optional[List[Segment]]:
    if verbose:
        print("Scaling points...")
    scaled_points = scale_points(points, scale_width, scale_height)

    if not points:
        if verbose:
            print("No points provided.")
        return None

    if verbose:
        print("Resetting points...")
    reset_points(scaled_points)

    min_x = min(p.x for p in scaled_points)
    start_candidates = [p for p in scaled_points if abs(p.x - min_x) <= margin]

    if verbose:
        print(f"Minimal x found: {min_x}")
        print(f"Found {len(start_candidates)} start candidates within margin {margin}")

    start_point = min(start_candidates, key=lambda p: p.y)
    start_point.type = "node"

    if verbose:
        print(f"Start point chosen: {start_point.to_string()}")

    x_leave = max(p.x for p in scaled_points)
    if verbose:
        print(f"x_leave set to: {x_leave}")

    segments = process_no_root_node(
        start_point,
        scaled_points,
        "up",
        x_leave,
        margin,
        verbose=verbose
    )

    segments = scale_segments(segments, scale_width=1.0/scale_width, scale_height=1.0/scale_height)

    if verbose:
        print("Segments built.")

    return segments

def scale_segment(segment: Segment, scale_width: float = 1.0, scale_height: float = 1.0) -> Segment:
    (x1, y1), (x2, y2) = segment
    return (x1 * scale_width, y1 * scale_height), (x2 * scale_width, y2 * scale_height)

def scale_segments(segments: List[Segment], scale_width: float = 1.0, scale_height: float = 1.0) -> List[Segment]:
    return [scale_segment(s, scale_width, scale_height) for s in segments]

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
