import re
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from .utils import Point, get_nearest_point, get_nearest_label, reset_points, scale_points, scale_texts, align_points_x, calculate_similarity

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

    def estimate_yule_lambda(self) -> float:
        n = len(self.get_all_path_lengths())
        max_l = self.max_path_length()
        if n < 2 or max_l <= 0: return 0.0
        return float(np.log(n) / max_l)

    def estimate_birth_death(self, eps: float = 0.5) -> Tuple[float, float, float]:
        n = len(self.get_all_path_lengths())
        max_l = self.max_path_length()
        if n < 2 or max_l <= 0: return 0.0, 0.0, 0.0

        lam = (np.log(n) - np.log(1 - eps)) / max_l
        mu = lam * eps
        r = lam - mu
        return float(lam), float(mu), float(r)

    def compute_similarity_by_path(self, other: "Newick") -> float:
        paths1 = self.get_all_path_lengths()
        paths2 = other.get_all_path_lengths()

        v1_raw = sorted(p[1] for p in self.get_all_path_lengths())
        v2_raw = sorted(p[1] for p in other.get_all_path_lengths())

        max_v1_raw = max(v1_raw)
        max_v2_raw = max(v2_raw)
        v1 = [x / max_v1_raw for x in v1_raw]
        v2 = [x / max_v2_raw for x in v2_raw]

        if not v1 and not v2: return 100.0
        if not v2 or not v2: return 0.0

        # Interpolation
        v1_interp = np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(v1)), v1)
        v2_interp = np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(v2)), v2)

        similarities = [
            calculate_similarity(p1, p2)
            for p1, p2 in zip(v1_interp, v2_interp)
        ]
        size_penalty = min(len(v1_raw), len(v2_raw)) / max(len(v1_raw), len(v2_raw))
        return float(np.mean(similarities) * size_penalty)

    def get_depth_profile(self):
        depths = []
        def traverse(node, d):
            depths.append(d)
            for child in node.children:
                traverse(child, d + 1)
        for root in self.internals:
            traverse(root, 0)
        return sorted(depths)

    def compute_similarity_by_depth(self, other: "Newick") -> float:
        v1 = self.get_depth_profile()
        v2 = other.get_depth_profile()

        if not v1 and not v2: return 100.0
        if not v1 or not v2: return 0.0

        size_penalty = min(len(v1), len(v2)) / max(len(v1), len(v2))

        # Interpolation
        v1_interp = np.interp(np.linspace(0, 1, 50), np.linspace(0, 1, len(v1)), v1)
        v2_interp = np.interp(np.linspace(0, 1, 50), np.linspace(0, 1, len(v2)), v2)

        similarities = [
            calculate_similarity(p1, p2)
            for p1, p2 in zip(v1_interp, v2_interp)
        ]
        return float(np.mean(similarities) * size_penalty)

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

def parse_newick_string(newick_str: str) -> Newick:
    newick_str = newick_str.strip()
    if newick_str.endswith(";"):
        newick_str = newick_str[:-1]

    def parse_node(s: str) -> NewickInternal:
        # "name:length" ou "(...):length"

        # last ':' that is not inside "(...)"
        depth = 0
        colon_idx = -1
        for i in range(len(s) - 1, -1, -1):
            if s[i] == ')': depth += 1
            elif s[i] == '(': depth -= 1
            elif s[i] == ':' and depth == 0:
                colon_idx = i
                break

        name_part = s
        length = 0.0

        if colon_idx != -1:
            name_part = s[:colon_idx]
            try:
                length = float(s[colon_idx + 1:])
            except ValueError:
                length = 0.0

        if name_part.startswith('(') and name_part.endswith(')'):
            # Internal node

            children_str = name_part[1:-1]
            children = []

            # split by ',' at first level
            start = 0
            depth = 0
            for i, char in enumerate(children_str):
                if char == '(': depth += 1
                elif char == ')': depth -= 1
                elif char == ',' and depth == 0:
                    children.append(parse_node(children_str[start:i]))
                    start = i + 1
            children.append(parse_node(children_str[start:]))

            return NewickInternal(name=None, length=length, children=children)
        else:
            # Leaf
            return NewickInternal(name=name_part, length=length, children=[])

    if newick_str.startswith('(') and newick_str.endswith(')'):
        root_node = parse_node(newick_str)
        return Newick(internals=root_node.children if root_node.children else [root_node])
    else:
        # Newick is just a leaf
        return Newick(internals=[parse_node(newick_str)])

def compare_newick_phylogeny(ref: Newick, other: Newick, target_max_path_length: float = 1.0, birth_death_eps: float = 0.5) -> Dict[str, Any]:
    for tree in [ref, other]:
        tree.normalize()
        tree.scale_lengths(target_max_path_length)

    # Pure Birth (Yule)
    y1 = ref.estimate_yule_lambda()
    y2 = other.estimate_yule_lambda()

    # Birth Death
    lam1, mu1, r1 = ref.estimate_birth_death(eps=birth_death_eps)
    lam2, mu2, r2 = other.estimate_birth_death(eps=birth_death_eps)

    return {
        "pure_birth_yule": {
            "lambda_ref": y1,
            "lambda_other": y2,
            "similarity_percent": calculate_similarity(y1, y2),
        },
        "birth_death": {
            "assumed_eps_mu_over_lambda": birth_death_eps,
            "lambda_ref": lam1,
            "lambda_other": lam2,
            "lambda_similarity_percent": calculate_similarity(lam1, lam2),
            "mu_ref": mu1,
            "mu_other": mu2,
            "mu_similarity_percent": calculate_similarity(mu1, mu2),
            "net_div_r_ref": r1,
            "net_div_r_other": r2,
            "net_div_r_similarity_percent": calculate_similarity(r1, r2)
        }
    }

def compare_newick_topology(ref: Newick, other: Newick) -> Dict[str, Any]:
    path_sim = ref.compute_similarity_by_path(other)
    depth_sim = ref.compute_similarity_by_depth(other)
    global_sim = (path_sim + depth_sim) / 2

    return {
        "similarity_by_path": path_sim,
        "similarity_by_depth": depth_sim,
        "average_similarity": global_sim
    }
