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

class Newick:
    def __init__(self, internals: List[NewickInternal] = []):
        self.internals = internals

    def to_string(self) -> str:
        if not self.internals:
            return "();"
        inner = ",".join(internal.to_string() for internal in self.internals)
        return f"({inner});"

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
    depth: int = 0
) -> List[NewickInternal]:
    if node.processed:
        print("    " * depth + f"Skipping node: {node.to_string()} (already processed)")
        return []
    node.processed = True
    print("    " * depth + f"Processing node: {node.to_string()} in direction {direction}")

    results: List[NewickInternal] = []

    if node.type != "corner":
        next_pt = get_nearest_point(node.x, node.y, points, direction, margin)
        if next_pt:
            print("    " * (depth + 1) + f"Found next point {next_pt.to_string()} in direction {direction}")
            sub = process_no_root_node(next_pt, points, direction, x_leave, margin, texts, max_distance, depth=depth + 1)
            results.extend(sub)

    right_pt = get_nearest_point(node.x, node.y, points, "right", margin)
    if right_pt:
        next_len = abs(right_pt.x - node.x)
        print("    " * (depth + 1) + f"Propagating right to {right_pt.to_string()} with length {next_len}")
        root_sub = process_root_node(right_pt, points, x_leave, margin, texts, max_distance, incoming_length=next_len, depth=depth + 1)
        if len(root_sub) == 0:
            pass
        elif len(root_sub) == 1:
            results.append(root_sub[0])
        else:
            results.append(NewickInternal(length=next_len, children=root_sub))
    else:
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
    depth: int = 0
) -> List[NewickInternal]:
    if node.processed:
        print("    " * depth + f"Skipping node: {node.to_string()} (already processed)")
        return []
    node.processed = True
    print("    " * depth + f"Processing root node: {node.to_string()} with incoming length {incoming_length}")

    if abs(node.x - x_leave) <= margin:
        print("    " * depth + f"Node near x_leave, creating leaf: {node.to_string()} with length {incoming_length}")
        return [NewickInternal(get_nearest_label(x_leave, node.y, texts, max_distance), length=incoming_length)]

    down_pt = get_nearest_point(node.x, node.y, points, "down", margin)
    up_pt = get_nearest_point(node.x, node.y, points, "up", margin)

    down_tree = process_no_root_node(down_pt, points, "down", x_leave, margin, texts, max_distance, depth=depth + 1) if down_pt else []
    up_tree = process_no_root_node(up_pt, points, "up", x_leave, margin, texts, max_distance, depth=depth + 1) if up_pt else []

    results: List[NewickInternal] = []

    if not down_tree and not up_tree:
        print("    " * (depth + 1) + f"No up/down branches, creating leaf with length {incoming_length + abs(x_leave - node.x)}")
        results.append(NewickInternal(get_nearest_label(x_leave, node.y, texts, max_distance), length=incoming_length + abs(x_leave - node.x)))
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
            texts,
            max_distance,
            depth=depth + 1
        )
        results.extend(sym_tree)
    else:
        results.extend(down_tree)
        results.extend(up_tree)

    right_pt = get_nearest_point(node.x, node.y, points, "right", margin)
    if right_pt:
        next_len = abs(right_pt.x - node.x)
        print("    " * (depth + 1) + f"Propagating right to {right_pt.to_string()} with length {next_len}")
        root_sub = process_root_node(right_pt, points, x_leave, margin, texts, max_distance, incoming_length=next_len, depth=depth + 1)
        if len(root_sub) == 0:
            pass
        elif len(root_sub) == 1:
            results.append(root_sub[0])
        else:
            results.append(NewickInternal(length=next_len, children=root_sub))

    return results

def reset_points(points: List[Point]) -> None:
    for p in points:
        p.processed = False

def build_newick_from_points(points: List[Point], margin: float = 0.5, texts: List[dict] = [], max_distance: float = 1) -> Optional[Newick]:
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

    newick_internals = process_no_root_node(start_point, points, "up", x_leave, margin, texts, max_distance)
    print("Newick built.")

    return Newick(newick_internals)

def build_newick(
    nodes: List[Tuple],
    corners: List[Tuple] = [],
    scale: float = 1500.0,
    margin: float = 0.5,
    texts: List[dict] = [],
    max_distance: float = 20
) -> Optional["Newick"]:
    """
    Build a Newick tree from nodes and corners.

    nodes : list of tuples, each tuple with at least 2 elements (x, y, ...)
    corners : list of tuples, each tuple with at least 2 elements (x, y, ...)
    margin : margin for comparison
    scale : scale factor to apply to coordinates
    texts : text labels with their bounding box
    max_distance : maximum distance allowed to associate a text label to a point
    return : Newick object
    """
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

    scaled_texts = []
    for entry in texts:
        x1, y1, x2, y2 = entry["bbox"]
        scaled_bbox = [
            x1 * scale,
            y1 * scale,
            x2 * scale,
            y2 * scale,
        ]

        scaled_texts.append({
            **entry,
            "bbox": scaled_bbox
        })

    return build_newick_from_points(points, margin=margin, texts=scaled_texts, max_distance=max_distance)
