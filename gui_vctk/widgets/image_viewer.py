import math
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk

from gui_vctk.core.models import Point, PointType, POINT_COLORS

# Segments preview (lignes vertes)
from itpt.core.newick import Point as ItptPoint, scale_points
from itpt.core.branches import build_segments, scale_segments


class ImageViewer(ctk.CTkFrame):
    """
    Image viewer optimisé:
    - rendu rapide: crop + resize uniquement de la zone visible (évite le lag au zoom)
    - pan (clic milieu / shift+clic / clic droit drag), zoom (molette)
    - ajout/déplacement/suppression de points
    - labels tip
    - prévisualisation de l'arbre: segments verts
    """

    def __init__(self, master):
        super().__init__(master)

        self.canvas = tk.Canvas(self, bg="#111111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.configure(takefocus=True)

        self._img_pil: Image.Image | None = None

        # zoom/pan
        self.scale = 1.0
        self.min_scale = 0.1
        self.max_scale = 8.0

        # pan offsets (en pixels canvas)
        self.pan_x = 0.0
        self.pan_y = 0.0

        # top-left de l'image rendue (coords canvas) (recalculé)
        self.img_x0 = 0.0
        self.img_y0 = 0.0

        # points
        self.points: list[Point] = []
        self.point_radius = 5
        self._draw_ids: list[int] = []

        # segments (prévisualisation)
        self.segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

        # interaction
        self.mode = "move"  # "move" | "add"
        self.selected_index: int | None = None
        self._drag_point_index: int | None = None

        self._panning = False
        self._pan_last = (0, 0)

        # home page hook (pour leaves panel)
        self._home_page = None

        # --- Optimisation rendu: slice visible ---
        self._slice_tk: ImageTk.PhotoImage | None = None
        self._slice_id: int | None = None
        self._slice_bbox: tuple[int, int, int, int] | None = None  # (x0,y0,x1,y1) coords image

        self._bind_events()

    # -------------------------
    # External hooks / API
    # -------------------------
    def set_home_page(self, home_page):
        self._home_page = home_page

    def set_mode(self, mode: str):
        self.mode = mode

    def load_image(self, path: str):
        self._img_pil = Image.open(path).convert("RGBA")
        self.scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.selected_index = None

        # invalidate slice cache
        self._slice_tk = None
        self._slice_bbox = None

        # force layout sizes updated
        self.update_idletasks()
        self.canvas.focus_set()
        self._render_image_and_points()

    def set_points(self, points: list[Point]):
        self.points = list(points)
        self.selected_index = None
        self._update_segments()
        self._redraw_points()

    def get_points(self) -> list[Point]:
        return list(self.points)

    def clear_points(self):
        self.points = []
        self.segments = []
        self.selected_index = None
        self._redraw_points()
        if self._home_page is not None:
            self._home_page.refresh_leaf_panel()

    def delete_selected_point(self):
        if self.selected_index is None:
            return
        if 0 <= self.selected_index < len(self.points):
            self.points.pop(self.selected_index)
        self.selected_index = None
        self._update_segments()
        self._redraw_points()
        if self._home_page is not None:
            self._home_page.refresh_leaf_panel()

    def set_point_label(self, index: int, label: str):
        if 0 <= index < len(self.points):
            self.points[index].label = label
            self._redraw_points()

    # -------------------------
    # Events
    # -------------------------
    def _bind_events(self):
        self.canvas.bind("<Configure>", lambda e: self._render_image_and_points())

        self.canvas.bind("<Button-1>", lambda e: (self.canvas.focus_set(), self._on_left_down(e)))
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_up)

        self.canvas.bind("<Button-2>", lambda e: (self.canvas.focus_set(), self._on_middle_down(e)))
        self.canvas.bind("<B2-Motion>", self._on_middle_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_middle_up)

        self.canvas.bind("<Shift-Button-1>", self._on_middle_down)
        self.canvas.bind("<Shift-B1-Motion>", self._on_middle_drag)
        self.canvas.bind("<Shift-ButtonRelease-1>", self._on_middle_up)

        self.canvas.bind("<MouseWheel>", self._on_mousewheel)      # Windows
        self.canvas.bind("<Button-4>", self._on_mousewheel_linux)  # Linux
        self.canvas.bind("<Button-5>", self._on_mousewheel_linux)

        # Flèches = pan
        self.canvas.bind("<Left>", self._on_arrow_key)
        self.canvas.bind("<Right>", self._on_arrow_key)
        self.canvas.bind("<Up>", self._on_arrow_key)
        self.canvas.bind("<Down>", self._on_arrow_key)

        # Pan: right click drag
        self.canvas.bind("<Button-3>", self._on_right_down)
        self.canvas.bind("<B3-Motion>", self._on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_right_up)

        # Delete selection
        self.canvas.bind_all("<Delete>", lambda e: self.delete_selected_point())

    # -------------------------
    # Coordinate mapping
    # -------------------------
    def _compute_img_top_left(self):
        if not self._img_pil:
            self.img_x0 = 0.0
            self.img_y0 = 0.0
            return

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        sw = self._img_pil.width * self.scale
        sh = self._img_pil.height * self.scale

        # center + pan
        self.img_x0 = (cw - sw) / 2.0 + self.pan_x
        self.img_y0 = (ch - sh) / 2.0 + self.pan_y

    def image_to_canvas(self, x: float, y: float):
        return (self.img_x0 + x * self.scale, self.img_y0 + y * self.scale)

    def canvas_to_image(self, x: float, y: float):
        return ((x - self.img_x0) / self.scale, (y - self.img_y0) / self.scale)

    # -------------------------
    # Rendering
    # -------------------------
    def _render_image_and_points(self):
        self._render_image()
        self._redraw_points()

    def _render_image(self):
        if not self._img_pil:
            self.canvas.delete("IMG")
            return

        self._compute_img_top_left()

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        # viewport canvas -> coords image
        ix0, iy0 = self.canvas_to_image(0, 0)
        ix1, iy1 = self.canvas_to_image(cw, ch)

        pad = 3
        x0 = max(0, int(min(ix0, ix1)) - pad)
        y0 = max(0, int(min(iy0, iy1)) - pad)
        x1 = min(self._img_pil.width, int(max(ix0, ix1)) + pad)
        y1 = min(self._img_pil.height, int(max(iy0, iy1)) + pad)

        if x1 <= x0 or y1 <= y0:
            self.canvas.delete("IMG")
            self._slice_tk = None
            self._slice_bbox = None
            return

        bbox = (x0, y0, x1, y1)

        # Only rebuild slice if bbox changed or cache empty
        if self._slice_tk is None or self._slice_bbox != bbox:
            visible = self._img_pil.crop(bbox)

            slice_w = max(1, int((x1 - x0) * self.scale))
            slice_h = max(1, int((y1 - y0) * self.scale))

            resized = visible.resize((slice_w, slice_h), Image.Resampling.BILINEAR)

            self._slice_tk = ImageTk.PhotoImage(resized)
            self._slice_bbox = bbox

        canvas_x, canvas_y = self.image_to_canvas(x0, y0)

        self.canvas.delete("IMG")
        self._slice_id = self.canvas.create_image(
            canvas_x, canvas_y, image=self._slice_tk, anchor="nw", tags=("IMG",)
        )

    def _redraw_points(self):
        for did in self._draw_ids:
            self.canvas.delete(did)
        self._draw_ids = []

        if not self._img_pil:
            return

        # --- Segments (prévisualisation arbre) derrière les points ---
        for (x1, y1), (x2, y2) in self.segments:
            sx1, sy1 = self.image_to_canvas(x1, y1)
            sx2, sy2 = self.image_to_canvas(x2, y2)
            lid = self.canvas.create_line(sx1, sy1, sx2, sy2, fill="green", width=3)
            self._draw_ids.append(lid)

        # points
        for i, p in enumerate(self.points):
            cx, cy = self.image_to_canvas(p.x, p.y)

            # rayon qui suit le zoom (et descend au dézoom)
            base = self.point_radius * (self.scale ** 0.85)
            r = max(0.8, min(18.0, base)) + (2 if i == self.selected_index else 0)

            color = POINT_COLORS[p.ptype]

            oid = self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="black", width=1, fill=color)
            self._draw_ids.append(oid)

            if p.label:
                tid = self.canvas.create_text(cx + 10, cy - 10, text=p.label, fill="black", font=("Segoe UI", 10),
                                              anchor="nw")
                self._draw_ids.append(tid)

    # -------------------------
    # Segments building
    # -------------------------
    def _update_segments(self):
        if not self._img_pil or not self.points:
            self.segments = []
            return

        img_w = float(self._img_pil.width)
        img_h = float(self._img_pil.height)

        pts_itpt: list[ItptPoint] = []
        for p in self.points:
            if p.ptype == PointType.TIP:
                continue
            t = "corner" if p.ptype == PointType.CORNER else "node"
            pts_itpt.append(ItptPoint(float(p.x), float(p.y), t))

        if not pts_itpt:
            self.segments = []
            return

        pts_norm = scale_points(pts_itpt, scale_width=1.0 / img_w, scale_height=1.0 / img_h)

        segs = build_segments(pts_norm)
        if not segs:
            self.segments = []
            return

        self.segments = scale_segments(segs, scale_width=img_w, scale_height=img_h)

    # -------------------------
    # Interaction
    # -------------------------
    def _on_left_down(self, event):
        if not self._img_pil:
            return

        if self.mode == "add":
            ix, iy = self.canvas_to_image(event.x, event.y)
            if 0 <= ix <= self._img_pil.width and 0 <= iy <= self._img_pil.height:
                ptype = PointType.NODE
                if self._home_page is not None:
                    ptype = PointType(self._home_page.ptype_var.get())

                label = None
                if ptype == PointType.TIP:
                    label = f"tip{sum(1 for p in self.points if p.ptype == PointType.TIP) + 1}"

                self.points.append(Point(ix, iy, ptype, label))
                self.selected_index = len(self.points) - 1
                self._update_segments()
                self._redraw_points()
                if self._home_page is not None:
                    self._home_page.refresh_leaf_panel()
            return

        th = max(6.0, 10.0 * self.scale)
        idx = self._hit_test(event.x, event.y, threshold=th)
        self.selected_index = idx
        self._drag_point_index = idx
        self._redraw_points()

    def _on_left_drag(self, event):
        if not self._img_pil:
            return
        if self._drag_point_index is None:
            return

        ix, iy = self.canvas_to_image(event.x, event.y)
        ix = max(0, min(ix, self._img_pil.width))
        iy = max(0, min(iy, self._img_pil.height))

        p = self.points[self._drag_point_index]
        p.x, p.y = ix, iy
        self._update_segments()
        self._redraw_points()

    def _on_left_up(self, event):
        self._drag_point_index = None

    def _on_middle_down(self, event):
        self._panning = True
        self._pan_last = (event.x, event.y)

    def _on_middle_drag(self, event):
        if not self._panning:
            return
        dx = event.x - self._pan_last[0]
        dy = event.y - self._pan_last[1]
        self._pan_last = (event.x, event.y)

        self.pan_x += dx
        self.pan_y += dy

        # invalidate slice bbox (pan changes viewport)
        self._slice_bbox = None
        self._render_image_and_points()

    def _on_middle_up(self, event):
        self._panning = False

    def _on_mousewheel(self, event):
        if not self._img_pil:
            return
        factor = 1.1 if event.delta > 0 else 1 / 1.1
        self._zoom_at(event.x, event.y, factor)

    def _on_mousewheel_linux(self, event):
        if not self._img_pil:
            return
        factor = 1.1 if event.num == 4 else 1 / 1.1
        self._zoom_at(event.x, event.y, factor)

    def _zoom_at(self, canvas_x: float, canvas_y: float, factor: float):
        old_scale = self.scale
        new_scale = max(self.min_scale, min(self.max_scale, old_scale * factor))
        if math.isclose(new_scale, old_scale):
            return

        self._compute_img_top_left()
        ix, iy = self.canvas_to_image(canvas_x, canvas_y)

        self.scale = new_scale

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        sw = self._img_pil.width * self.scale
        sh = self._img_pil.height * self.scale
        center_x0_new = (cw - sw) / 2.0
        center_y0_new = (ch - sh) / 2.0

        self.pan_x = canvas_x - center_x0_new - ix * self.scale
        self.pan_y = canvas_y - center_y0_new - iy * self.scale

        # invalidate slice cache on zoom
        self._slice_tk = None
        self._slice_bbox = None
        self._render_image_and_points()

    def _hit_test(self, canvas_x: float, canvas_y: float, threshold: float = 10.0):
        best_i = None
        best_d = 1e18
        for i, p in enumerate(self.points):
            px, py = self.image_to_canvas(p.x, p.y)
            d = (px - canvas_x) ** 2 + (py - canvas_y) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        if best_i is None:
            return None
        if best_d <= threshold ** 2:
            return best_i
        return None

    def _on_arrow_key(self, event):
        if not self._img_pil:
            return

        step = 50
        if event.state & 0x0001:  # Shift
            step = 60
        if event.state & 0x0004:  # Ctrl
            step = 5

        if event.keysym == "Right":
            self.pan_x -= step
        elif event.keysym == "Left":
            self.pan_x += step
        elif event.keysym == "Down":
            self.pan_y -= step
        elif event.keysym == "Up":
            self.pan_y += step

        self._slice_bbox = None
        self._render_image_and_points()

    def _on_right_down(self, event):
        self.canvas.focus_set()
        self._panning = True
        self._pan_last = (event.x, event.y)

    def _on_right_drag(self, event):
        if not self._panning:
            return
        dx = event.x - self._pan_last[0]
        dy = event.y - self._pan_last[1]
        self._pan_last = (event.x, event.y)

        self.pan_x += dx
        self.pan_y += dy

        self._slice_bbox = None
        self._render_image_and_points()

    def _on_right_up(self, event):
        self._panning = False
