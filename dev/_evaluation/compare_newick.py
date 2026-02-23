import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

@dataclass
class Node:
    __hash__ = object.__hash__
    name: str = ""
    length: float = 0.0
    parent: Optional["Node"] = None
    children: List["Node"] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

def parse_newick(newick: str) -> Node:
    s = newick.strip()
    if not s.endswith(";"):
        raise ValueError("Newick must end with ';'")
    s = s[:-1]

    stack: List[Node] = []
    cur = Node(name="__root__")
    token = ""
    i = 0

    def split_name_length(tok: str) -> Tuple[str, float]:
        if ":" in tok:
            parts = tok.split(":")
            name = parts[0].strip()
            try:
                length = float(parts[1]) if parts[1].strip() else 0.0
            except ValueError:
                raise ValueError(f"Invalid branch length in token: {tok}")
            return name, length
        return tok.strip(), 0.0

    def flush_token_as_leaf(tok: str):
        tok = tok.strip()
        if not tok:
            return
        name, length = split_name_length(tok)
        leaf = Node(name=name, length=length, parent=cur)
        cur.children.append(leaf)

    while i < len(s):
        c = s[i]
        if c == "(":
            internal = Node(name="", parent=cur)
            cur.children.append(internal)
            stack.append(cur)
            cur = internal
            token = ""
            i += 1
        elif c == ",":
            flush_token_as_leaf(token)
            token = ""
            i += 1
        elif c == ")":
            flush_token_as_leaf(token)
            token = ""
            parent = stack.pop() if stack else None
            cur_closed = cur
            cur = parent if parent is not None else cur

            i += 1
            meta = ""
            while i < len(s) and s[i] not in ",()":
                meta += s[i]
                i += 1
            meta = meta.strip()
            if meta:
                name, length = split_name_length(meta)
                cur_closed.name = name
                cur_closed.length = length
        else:
            token += c
            i += 1

    if len(cur.children) == 1 and cur.name == "__root__":
        root = cur.children[0]
        root.parent = None
        return root

    cur.parent = None
    return cur

def get_leaves(root: Node) -> List[Node]:
    leaves = []
    stack = [root]
    while stack:
        n = stack.pop()
        if not n.children:
            leaves.append(n)
        else:
            stack.extend(n.children)
    return leaves

def compute_root_distances(root: Node) -> Dict[Node, float]:
    dist = {root: 0.0}
    stack = [root]
    while stack:
        n = stack.pop()
        for ch in n.children:
            dist[ch] = dist[n] + ch.length
            stack.append(ch)
    return dist

def rescale_tree(root: Node, factor: float) -> None:
    stack = [root]
    while stack:
        n = stack.pop()
        for ch in n.children:
            ch.length *= factor
            stack.append(ch)

def lca(a: Node, b: Node) -> Node:
    seen = set()
    x = a
    while x is not None:
        seen.add(id(x))
        x = x.parent
    y = b
    while y is not None:
        if id(y) in seen:
            return y
        y = y.parent
    raise RuntimeError("LCA not found")

def tree_height(root: Node) -> float:
    dist = compute_root_distances(root)
    leaves = get_leaves(root)
    return max(dist[lf] for lf in leaves) if leaves else 0.0

def estimate_yule_lambda(n_tips: int, T: float) -> float:
    if T <= 0.0 or n_tips <= 1:
        return 0.0
    return math.log(float(n_tips)) / T

def estimate_bd_lambda_mu(n_tips: int, T: float, eps: float) -> Tuple[float, float, float, float]:
    if T <= 0.0 or n_tips <= 1:
        return 0.0, 0.0, 0.0, eps
    eps = min(max(eps, 0.0), 0.999999)
    r = math.log(float(n_tips) * (1.0 - eps) + eps) / T
    lam = r / (1.0 - eps)
    mu = eps * lam
    return lam, mu, r, eps

def similarity_percent_scalar(a: float, b: float) -> float:
    mx = max(abs(a), abs(b))
    if mx <= 1e-12:
        return 100.0
    return max(0.0, 1.0 - abs(a - b) / mx) * 100.0

def compare_newicks(newick_ref: str, newick_to_scale: str, eps: float = 0.5) -> Dict[str, Dict[str, float]]:
    t1 = parse_newick(newick_ref)
    t2 = parse_newick(newick_to_scale)

    leaves1 = get_leaves(t1)
    leaves2 = get_leaves(t2)

    set1 = {lf.name for lf in leaves1}
    set2 = {lf.name for lf in leaves2}
    common = sorted(list(set1 & set2))
    if len(common) < 2:
        raise ValueError("Need at least 2 common leaves between trees.")

    dist1 = compute_root_distances(t1)
    dist2 = compute_root_distances(t2)
    by_name1 = {lf.name: lf for lf in leaves1}
    by_name2 = {lf.name: lf for lf in leaves2}
    mean1 = sum(dist1[by_name1[n]] for n in common) / len(common)
    mean2 = sum(dist2[by_name2[n]] for n in common) / len(common)
    factor = (mean1 / mean2) if mean2 > 1e-12 else 1.0
    rescale_tree(t2, factor)

    n1 = len(leaves1)
    n2 = len(leaves2)
    T1 = tree_height(t1)
    T2 = tree_height(t2)

    y1 = estimate_yule_lambda(n1, T1)
    y2 = estimate_yule_lambda(n2, T2)
    ysim = similarity_percent_scalar(y1, y2)

    lam1, mu1, r1, _ = estimate_bd_lambda_mu(n1, T1, eps)
    lam2, mu2, r2, _ = estimate_bd_lambda_mu(n2, T2, eps)
    lsim = similarity_percent_scalar(lam1, lam2)
    msim = similarity_percent_scalar(mu1, mu2)
    rsim = similarity_percent_scalar(r1, r2)

    return {
        "scaling": {
            "mean_root_to_tip_ref": mean1,
            "mean_root_to_tip_other_before": mean2,
            "scale_factor_applied_to_other": factor,
        },
        "pure_birth_yule": {
            "lambda_ref": y1,
            "lambda_other": y2,
            "similarity_percent": ysim,
        },
        "birth_death": {
            "assumed_eps_mu_over_lambda": eps,
            "lambda_ref": lam1,
            "lambda_other": lam2,
            "lambda_similarity_percent": lsim,
            "mu_ref": mu1,
            "mu_other": mu2,
            "mu_similarity_percent": msim,
            "net_div_r_ref": r1,
            "net_div_r_other": r2,
            "net_div_r_similarity_percent": rsim,
        },
    }

def pretty_print(result: Dict[str, Dict[str, float]]) -> None:
    sc = result["scaling"]
    print("=== Scaling ===")
    print(f"mean(root->tip) ref = {sc['mean_root_to_tip_ref']:.6g}")
    print(f"mean(root->tip) other (before) = {sc['mean_root_to_tip_other_before']:.6g}")
    print(f"scale factor applied to other = {sc['scale_factor_applied_to_other']:.6g}")

    yu = result["pure_birth_yule"]
    print("\n=== Pure-birth (Yule) ===")
    print(f"λ (speciation, μ=0) | ref={yu['lambda_ref']:.6g} | other={yu['lambda_other']:.6g} | sim={yu['similarity_percent']:.2f}%")

    bd = result["birth_death"]
    print("\n=== Birth–death (approx) ===")
    print(f"ε = μ/λ (fixed)     | {bd['assumed_eps_mu_over_lambda']:.3g}")
    print(f"λ (speciation)      | ref={bd['lambda_ref']:.6g} | other={bd['lambda_other']:.6g} | sim={bd['lambda_similarity_percent']:.2f}%")
    print(f"μ (extinction)      | ref={bd['mu_ref']:.6g} | other={bd['mu_other']:.6g} | sim={bd['mu_similarity_percent']:.2f}%")
    print(f"r = λ−μ (net div.)  | ref={bd['net_div_r_ref']:.6g} | other={bd['net_div_r_other']:.6g} | sim={bd['net_div_r_similarity_percent']:.2f}%")
