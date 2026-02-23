from typing import List, Tuple, Optional
from .utils import Point, get_nearest_point, get_nearest_label, reset_points, scale_points, scale_texts, align_points_x

class NewickInternal:
    def __init__(self, name: Optional[str] = None, length: float = 0.0, children: List["NewickInternal"] = []):
        self.name = name
        self.length = length
        self.children = children

    def to_string(self) -> str:
        if not self.children:
            return f"{self.name}:{self.length}"
        inner = ",".join(child.to_string() for child in self.children)
        return f"({inner}):{self.length}"

    def max_path_length(self, acc: float = 0.0) -> float:
        total = acc + self.length
        if not self.children:
            return total
        return max(child.max_path_length(total) for child in self.children)

    def scale_lengths(self, factor: float):
        self.length *= factor
        for child in self.children:
            child.scale_lengths(factor)

    def force_total_length(self, target: float = 1.0, current_acc: float = 0.0):
        if not self.children:
            self.length = max(0.0, target - current_acc)
            return

        for child in self.children:
            child.force_total_length(target, current_acc + self.length)

    def get_all_path_lengths(self, acc: float = 0.0) -> List[Tuple[str, float]]:
        current_path = acc + self.length

        if not self.children:
            return [(self.name, current_path)]

        results = []
        for child in self.children:
            results.extend(child.get_all_path_lengths(current_path))
        return results

class Newick:
    def __init__(self, internals: List[NewickInternal] = []):
        self.internals = internals

    def to_string(self) -> str:
        if not self.internals:
            return "();"
        inner = ",".join(internal.to_string() for internal in self.internals)
        return f"({inner});"

    def max_path_length(self) -> float:
        return max(root.max_path_length() for root in self.internals)

    def scale_lengths(self, factor: float):
        for root in self.internals:
            root.scale_lengths(factor)

    def force_total_length(self, target: float = 1.0):
        for root in self.internals:
            root.force_total_length(target=1.0)

    def get_all_path_lengths(self) -> List[Tuple[str, float]]:
        results = []
        for root in self.internals:
            results.extend(root.get_all_path_lengths())
        return results

    def check_leaf_alignment(self):
        data = self.get_all_path_lengths()

        if not data:
            print("Empty tree.")
            return

        lengths = [d[1] for d in data]

        print(f"\n--- Checking Newick Alignment ({len(data)} leaves) ---")
        for i, (name, length) in enumerate(data):
            print(f"Leaf {i+1} [{name}] : {length}")

        print(f"Min Length : {min(lengths)}")
        print(f"Max Length : {max(lengths)}")
        print(f"Difference : {max(lengths) - min(lengths):.2e}")

    def normalize(self):
        if not self.internals:
            return

        max_len = self.max_path_length()
        if max_len == 0: return

        self.scale_lengths(1.0 / max_len)

        self.force_total_length(target=1.0)

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

        results.append(NewickInternal(get_nearest_label(x_leave, node.y, texts, max_distance, depth, verbose), length=abs(x_leave - node.x)))

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

        return [NewickInternal(get_nearest_label(x_leave, node.y, texts, max_distance, depth, verbose), length=incoming_length)]

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

        results.append(NewickInternal(get_nearest_label(x_leave, node.y, texts, max_distance, depth, verbose), length=incoming_length + abs(x_leave - node.x)))
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

        if not sym_tree:

            if verbose:
                print("    " * (depth + 1) + f"Symmetric branch is empty, creating leaf at x={x_leave}")

            results.append(NewickInternal(name=get_nearest_label(x_leave, sym_y, texts, max_distance, depth, verbose), length=abs(x_leave - node.x)))
        else:
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

def build_newick(
    points: List[Point],
    margin: float = 5,
    texts: List[dict] = [],
    max_distance: float = 20,
    scale_width: float = 1500,
    scale_height: float = 1500,
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
    if verbose:
        print("Scaling points...")
    scaled_points = scale_points(points, scale_width, scale_height)
    if verbose:
        print("Aligning points...")
    scaled_aligned_points = align_points_x(scaled_points, margin)

    print("Scaling texts...")
    scaled_texts = scale_texts(texts, scale_width, scale_height)

    if not scaled_aligned_points:
        if verbose:
            print("No points provided.")
        return None

    if verbose:
        print("Resetting points...")
    reset_points(scaled_aligned_points)

    min_x = min(p.x for p in scaled_aligned_points)
    start_candidates = [p for p in scaled_aligned_points if abs(p.x - min_x) <= margin]

    if verbose:
        print(f"Minimal x found: {min_x}")
        print(f"Found {len(start_candidates)} start candidates within margin {margin}")

    start_point = min(start_candidates, key=lambda p: p.y)
    start_point.type = "node"

    if verbose:
        print(f"Start point chosen: {start_point.to_string()}")

    x_leave = max(p.x for p in scaled_aligned_points)
    if verbose:
        print(f"x_leave set to: {x_leave}")

    newick_internals = process_no_root_node(
        start_point,
        scaled_aligned_points,
        "up",
        x_leave,
        margin,
        scaled_texts,
        max_distance,
        verbose=verbose
    )

    newick = Newick(newick_internals)
    newick.normalize()

    if verbose:
        print("Newick built.")

    return newick
