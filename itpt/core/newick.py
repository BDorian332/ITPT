from typing import List, Tuple, Optional

class Point:
    def __init__(self, x: float, y: float, point_type: str = "node"):
        self.x = x
        self.y = y
        self.type = point_type # "node" or "corner"
        self.processed = False

    def to_string(self) -> str:
        return f"Point({self.x}, {self.y}, {self.type}, processed={self.processed})"

class NewickInternal:
    def __init__(self, name: Optional[str] = None, length: float = 0.0, children: List["NewickInternal"] = []):
        self.name = name
        self.length = length
        self.children = children

    def to_string(self) -> str:
        if not self.children:
            return f"{self.name}:{self.length:.6f}"
        inner = ",".join(child.to_string() for child in self.children)
        return f"({inner}):{self.length:.6f}"

    def max_path_length(self, acc: float = 0.0) -> float:
        total = acc + self.length
        if not self.children:
            return total
        return max(child.max_path_length(total) for child in self.children)

    def scale_lengths(self, factor: float):
        self.length *= factor
        for child in self.children:
            child.scale_lengths(factor)

class Newick:
    def __init__(self, internals: List[NewickInternal] = []):
        self.internals = internals

    def to_string(self) -> str:
        if not self.internals:
            return "();"
        inner = ",".join(internal.to_string() for internal in self.internals)
        return f"({inner});"

    def max_path_length(self, acc: float = 0.0) -> float:
        return max(root.max_path_length() for root in self.internals)

    def scale_lengths(self, factor: float):
        for root in self.internals:
            root.scale_lengths(factor)

    def normalize(self):
        if not self.internals:
            return

        max_path_length = self.max_path_length()

        if max_path_length == 0:
            return

        factor = 1.0 / max_path_length
        self.scale_lengths(factor)

def get_nearest_label(x: float, y: float, texts: List[dict], max_distance: float) -> str:
    nearest_label = "leaf"
    min_distance = max_distance

    for entry in texts:
        bbox = entry.get("bbox")
        text = entry.get("text")

        x_left = bbox[0]
        y_center = (bbox[1] + bbox[3]) / 2
        distance = ((x - x_left) ** 2 + (y - y_center) ** 2) ** 0.5

        if distance <= min_distance:
            min_distance = distance
            nearest_label = text

    return nearest_label

def get_nearest_point(x: float, y: float, points: List[Point], direction: str, margin: float) -> Optional[Point]:
    nearest_point = None
    min_distance = float('inf')

    for p in points:
        if direction == "up" and p.y > y and abs(p.x - x) <= margin:
            distance = p.y - y
            if distance < min_distance:
                min_distance = distance
                nearest_point = p
        elif direction == "down" and p.y < y and abs(p.x - x) <= margin:
            distance = y - p.y
            if distance < min_distance:
                min_distance = distance
                nearest_point = p
        elif direction == "right" and p.x > x and abs(p.y - y) <= margin:
            distance = p.x - x
            if distance < min_distance:
                min_distance = distance
                nearest_point = p
        elif direction == "left" and p.x < x and abs(p.y - y) <= margin:
            distance = x - p.x
            if distance < min_distance:
                min_distance = distance
                nearest_point = p
    return nearest_point

def process_no_root_node(
    node: Point,
    points: List[Point],
    direction: str,
    x_leave: float,
    margin: float,
    texts: List[dict],
    max_distance: float,
    depth: int = 0,
    verbose: bool = False
) -> List[NewickInternal]:

    if node.processed:
        if verbose:
            print("    " * depth + f"Skipping node: {node.to_string()} (already processed)")
        return []

    node.processed = True

    if verbose:
        print("    " * depth + f"Processing node: {node.to_string()} in direction {direction}")

    results: List[NewickInternal] = []

    if node.type != "corner":
        next_pt = get_nearest_point(node.x, node.y, points, direction, margin)
        if next_pt:
            if verbose:
                print("    " * (depth + 1) + f"Found next point {next_pt.to_string()} in direction {direction}")

            sub = process_no_root_node(
                next_pt,
                points,
                direction,
                x_leave,
                margin,
                texts,
                max_distance,
                depth=depth + 1,
                verbose=verbose
            )
            results.extend(sub)

    right_pt = get_nearest_point(node.x, node.y, points, "right", margin)

    if right_pt:
        next_len = abs(right_pt.x - node.x)

        if verbose:
            print("    " * (depth + 1) + f"Propagating right to {right_pt.to_string()} with length {next_len}")

        root_sub = process_root_node(
            right_pt,
            points,
            x_leave,
            margin,
            texts,
            max_distance,
            incoming_length=next_len,
            depth=depth + 1,
            verbose=verbose
        )

        if len(root_sub) == 1:
            results.append(root_sub[0])
        elif len(root_sub) > 1:
            results.append(NewickInternal(length=next_len, children=root_sub))

    else:
        if verbose:
            print("    " * (depth + 1) + f"No point to the right, creating leaf at x={x_leave}")

        results.append(NewickInternal(get_nearest_label(x_leave, node.y, texts, max_distance), length=abs(x_leave - node.x)))

    return results

def process_root_node(
    node: Point,
    points: List[Point],
    x_leave: float,
    margin: float,
    texts: List[dict],
    max_distance: float,
    incoming_length: float = 0.0,
    depth: int = 0,
    verbose: bool = False
) -> List[NewickInternal]:

    if node.processed:
        if verbose:
            print("    " * depth + f"Skipping node: {node.to_string()} (already processed)")
        return []

    node.processed = True

    if verbose:
        print("    " * depth + f"Processing root node: {node.to_string()} with incoming length {incoming_length}")

    if abs(node.x - x_leave) <= margin:
        if verbose:
            print("    " * depth + f"Node near x_leave, creating leaf: {node.to_string()} with length {incoming_length}")

        return [NewickInternal(get_nearest_label(x_leave, node.y, texts, max_distance), length=incoming_length)]

    down_pt = get_nearest_point(node.x, node.y, points, "down", margin)
    up_pt = get_nearest_point(node.x, node.y, points, "up", margin)

    down_tree = process_no_root_node(
        down_pt, points, "down", x_leave, margin,
        texts, max_distance,
        depth=depth + 1,
        verbose=verbose
    ) if down_pt else []

    up_tree = process_no_root_node(
        up_pt, points, "up", x_leave, margin,
        texts, max_distance,
        depth=depth + 1,
        verbose=verbose
    ) if up_pt else []

    results: List[NewickInternal] = []

    if not down_tree and not up_tree:
        if verbose:
            print("    " * (depth + 1) + "No up/down branches, creating leaf")

        results.append(NewickInternal(get_nearest_label(x_leave, node.y, texts, max_distance), length=incoming_length + abs(x_leave - node.x)))
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
            texts,
            max_distance,
            depth=depth + 1,
            verbose=verbose
        )

        results.extend(sym_tree)

    else:
        results.extend(down_tree)
        results.extend(up_tree)

    right_pt = get_nearest_point(node.x, node.y, points, "right", margin)

    if right_pt:
        next_len = abs(right_pt.x - node.x)

        if verbose:
            print("    " * (depth + 1) + f"Propagating right to {right_pt.to_string()} with length {next_len}")

        root_sub = process_root_node(
            right_pt,
            points,
            x_leave,
            margin,
            texts,
            max_distance,
            incoming_length=next_len,
            depth=depth + 1,
            verbose=verbose
        )

        if len(root_sub) == 1:
            results.append(root_sub[0])
        elif len(root_sub) > 1:
            results.append(NewickInternal(length=next_len, children=root_sub))

    return results

def reset_points(points: List[Point]) -> None:
    for p in points:
        p.processed = False

def build_newick(
    points: List[Point],
    margin: float = 0.5,
    texts: List[dict] = [],
    max_distance: float = 1,
    scale_width: float = 1.0,
    scale_height: float = 1.0,
    verbose: bool = False
) -> Optional[Newick]:
    """
    Build a Newick tree from a list of Points and optional texts, applying separate width/height scaling.

    points : list of Point objects
    margin : margin for determining start point
    texts : list of text dicts with "bbox" key
    max_distance : maximum distance to associate a text to a point
    scale_width : factor to scale x coordinates
    scale_height : factor to scale y coordinates
    verbose : if True, print debug info
    return : Newick object or None if no points
    """
    scaled_points = scale_points(points, scale_width, scale_height)
    scaled_texts = scale_texts(texts, scale_width, scale_height)

    if verbose:
        print("Resetting points...")
    reset_points(scaled_points)

    if not scaled_points:
        if verbose:
            print("No points provided.")
        return None

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

    newick_internals = process_no_root_node(
        start_point,
        scaled_points,
        "up",
        x_leave,
        margin,
        scaled_texts,
        max_distance,
        verbose=verbose
    )

    if verbose:
        print("Newick built.")

    return Newick(newick_internals)

def scale_points(points: List[Point], scale_width: float = 1.0, scale_height: float = 1.0) -> List[Point]:
    """
    Scale a list of Point objects by separate width and height scales.

    points : list of Point objects
    scale_width : scale factor for x coordinate
    scale_height : scale factor for y coordinate
    return : new list of scaled Point objects
    """
    return [Point(p.x * scale_width, p.y * scale_height, p.type) for p in points]

def scale_texts(texts: List[dict], scale_width: float = 1.0, scale_height: float = 1.0) -> List[dict]:
    """
    Scale text bounding boxes by separate width and height scales.

    texts : list of dicts with "bbox": [x1, y1, x2, y2]
    scale_width : scale factor for x coordinates
    scale_height : scale factor for y coordinates
    return : new list of scaled text dicts
    """
    scaled_texts = []
    for entry in texts:
        x1, y1, x2, y2 = entry["bbox"]
        scaled_bbox = [
            x1 * scale_width,
            y1 * scale_height,
            x2 * scale_width,
            y2 * scale_height
        ]
        scaled_texts.append({**entry, "bbox": scaled_bbox})
    return scaled_texts

def points_to_tuples(points: List[Point], scale_width: float = 1.0, scale_height: float = 1.0):
    nodes_list = []
    corners_list = []

    for p in points:
        x = p.x * scale_width
        y = p.y * scale_height
        if p.type == "node":
            nodes_list.append((x, y))
        elif p.type == "corner":
            corners_list.append((x, y))

    return nodes_list, corners_list

def tuples_to_points(nodes_tuples: List[tuple], corners_tuples: List[tuple], scale_width: float = 1.0, scale_height: float = 1.0):
    points = []

    for x, y in nodes_tuples:
        points.append(Point(x * width, y * height, "node"))

    for x, y in corners_tuples:
        points.append(Point(x * width, y * height, "corner"))

    return points
