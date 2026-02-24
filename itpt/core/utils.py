import numpy as np
from typing import List, Tuple, Optional
from numpy.typing import NDArray

class Point:
    def __init__(self, x: float, y: float, point_type: str = "node"):
        self.x = x
        self.y = y
        self.type = point_type # "node" or "corner"
        self.processed = False

    def to_string(self) -> str:
        return f"Point({self.x}, {self.y}, {self.type}, processed={self.processed})"

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

def get_nearest_label(x: float, y: float, texts: List[dict], max_distance: float, depth: int, verbose: bool) -> str:
    nearest_label = "leaf"
    min_distance = max_distance

    if verbose:
        print("    " * depth + f"Detecting label near point ({x}, {y})")

    for entry in texts:
        bbox = entry.get("bbox")
        text = entry.get("text")

        x_left = bbox[0]
        y_center = (bbox[1] + bbox[3]) / 2
        distance = ((x - x_left) ** 2 + (y - y_center) ** 2) ** 0.5

        if verbose:
            print("    " * depth + f"    - Testing '{text}' at ({x_left:.1f}, {y_center:.1f})")
            print("    " * depth + f"      Distance: {distance:.2f}")

        if distance <= min_distance:
            min_distance = distance
            nearest_label = text
            if verbose: print("    " * depth + f"      => MATCH (New nearest)")

    return nearest_label

def reset_points(points: List[Point]) -> None:
    for p in points:
        p.processed = False

def scale_point(point: Point, scale_width: float = 1.0, scale_height: float = 1.0) -> Point:
    return Point(point.x * scale_width, point.y * scale_height, point.type)

def scale_points(points: List[Point], scale_width: float = 1.0, scale_height: float = 1.0) -> List[Point]:
    """
    Scale a list of Point objects by separate width and height scales.

    points : list of Point objects
    scale_width : scale factor for x coordinate
    scale_height : scale factor for y coordinate
    return : new list of scaled Point objects
    """
    return [scale_point(p, scale_width, scale_height) for p in points]

def scale_text(text: List[dict], scale_width: float = 1.0, scale_height: float = 1.0) -> dict:
    x1, y1, x2, y2 = text["bbox"]
    scaled_bbox = [
        x1 * scale_width,
        y1 * scale_height,
        x2 * scale_width,
        y2 * scale_height
    ]
    return {**text, "bbox": scaled_bbox}

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
        scaled_texts.append(scale_text(entry, scale_width, scale_height))
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
        points.append(Point(x * scale_width, y * scale_height, "node"))

    for x, y in corners_tuples:
        points.append(Point(x * scale_width, y * scale_height, "corner"))

    return points

def align_points_x(points: List[Point], margin: float) -> List[Point]:
    if not points:
        return []

    sorted_pts = sorted(points, key=lambda p: p.x)

    groups = []
    if sorted_pts:
        current_group = [sorted_pts[0]]
        for i in range(1, len(sorted_pts)):
            if sorted_pts[i].x - current_group[-1].x <= margin:
                current_group.append(sorted_pts[i])
            else:
                groups.append(current_group)
                current_group = [sorted_pts[i]]
        groups.append(current_group)

    aligned_points = []
    for group in groups:
        avg_x = sum(p.x for p in group) / len(group)
        for p in group:
            new_p = Point(avg_x, p.y, p.type)
            new_p.processed = p.processed
            aligned_points.append(new_p)

    return aligned_points

Segment = Tuple[Tuple[float, float], Tuple[float, float]]

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

def calculate_similarity(v1: float, v2: float) -> float:
    denominator = max(v1, v2, 1e-9)
    similarity = (1 - abs(v1 - v2) / denominator) * 100
    return max(0.0, similarity)
