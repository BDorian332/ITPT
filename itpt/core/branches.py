from typing import List, Optional
from .utils import Point, get_nearest_point, reset_points, scale_points, Segment, scale_segments

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
