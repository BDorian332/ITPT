import os
import tkinter as tk
import threading
import importlib
import numpy as np
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk
from itpt.models import get_list, get_model
from itpt.core.newick import Point, scale_points
from itpt.core.branches import build_segments, scale_segments

class Step:
    def __init__(self, name, default_enabled=True):
        self.name = name
        self.default_enabled = default_enabled
        self.enabled = default_enabled

class ITPTGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ITPTGUI")
        self.root.geometry("1000x750")

        self.model_names = get_list()

        # ---- Preview state ----

        self.preview_image = None
        self.zoomed_image = None
        self.tk_image = None
        self.zoom = 1.0
        self.last_zoom = self.zoom
        self.pan_x = 0
        self.pan_y = 0
        self.last_pan = (self.pan_x, self.pan_y)
        self.drag_start = None
        self.drag_type = None # "left" or "right"

        # ---- Points, Texts and Segments ----

        self.points = [] # list of Point objects
        self.selected_point = None # for drag

        self.texts = []
        self.selected_text_id = None

        self.add_mode = None # "node", "corner", "text" ou None

        self.segments = []

        # ---- UI ----

        self.build_ui()
        self.bind_events()

    # ---------- UI ----------

    def build_ui(self):
        root = self.root

        # Input
        ttk.Label(root, text="Input image:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.input_entry = ttk.Entry(root)
        self.input_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.input_entry.bind("<KeyRelease>", lambda e: self.update_preview())
        ttk.Button(root, text="Browse", command=self.browse_input).grid(row=0, column=2, padx=5, pady=5)

        # Output
        ttk.Label(root, text="Output file:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.output_entry = ttk.Entry(root)
        self.output_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(root, text="Browse", command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)

        # Model selection
        ttk.Label(root, text="Select model:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        if not self.model_names:
            self.model_name_var = tk.StringVar(value="No models")
            self.model_combobox = ttk.Combobox(root, textvariable=self.model_name_var, state="disabled")
        else:
            self.model_name_var = tk.StringVar(value=self.model_names[0])
            self.model_combobox = ttk.Combobox(root, textvariable=self.model_name_var, values=self.model_names, state="readonly")
        self.model_combobox.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        self.settings_btn = ttk.Button(root, text="Settings", command=self.open_model_settings, state="disabled" if not self.model_names else "enabled")
        self.settings_btn.grid(row=2, column=2, padx=5, pady=5, sticky="ew")
        self.weights_overrides = {}

        # Canvas
        ttk.Label(root, text="Image preview:").grid(row=3, column=0, columnspan=3, sticky="w", padx=5, pady=5)
        self.preview_canvas = tk.Canvas(root, bg="white")
        self.preview_canvas.grid(row=4, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        self.preview_canvas.bind("<Configure>", lambda e: self.redraw_preview(force=True))

        # Point buttons (toggle exclusive)
        edit_buttons_frame = ttk.Frame(root)
        edit_buttons_frame.grid(row=5, column=0, columnspan=3, sticky="w")
        self.add_node_btn = ttk.Button(edit_buttons_frame, text="Add Node", command=lambda: self.toggle_mode("node"), state="disabled")
        self.add_corner_btn = ttk.Button(edit_buttons_frame, text="Add Corner", command=lambda: self.toggle_mode("corner"), state="disabled")
        self.clear_points_btn = ttk.Button(edit_buttons_frame, text="Clear All Points (right click on one to delete it)", command=self.clear_points, state="disabled")
        self.add_text_btn = ttk.Button(edit_buttons_frame, text="Add Text", command=lambda: self.toggle_mode("text"), state="disabled")
        self.clear_texts_btn = ttk.Button(edit_buttons_frame, text="Clear All Texts (right click on one to delete it)", command=self.clear_texts, state="disabled")
        self.add_node_btn.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.add_corner_btn.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.clear_points_btn.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.add_text_btn.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        self.clear_texts_btn.grid(row=0, column=4, padx=5, pady=5, sticky="w")

        # Output text
        ttk.Label(root, text="Generated Newick:").grid(row=6, column=0, columnspan=3, sticky="w", padx=5, pady=5)
        output_frame = ttk.Frame(root)
        output_frame.grid(row=7, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        self.output_text = tk.Text(output_frame, height=5, wrap="word")
        self.output_text.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(output_frame, orient="vertical", command=self.output_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.output_text.configure(yscrollcommand=scrollbar.set)

        # Steps options
        self.steps_frame = ttk.LabelFrame(self.root, text="Steps options")
        self.steps_frame.grid(row=8, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        self.step_vars = {}

        # Convert button + progress
        self.convert_button = ttk.Button(root, text="Convert", command=self.convert)
        self.convert_button.grid(row=9, column=0, columnspan=3, padx=5, pady=5)
        self.progress = ttk.Progressbar(root, mode="indeterminate")
        self.progress.grid(row=10, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        self.progress.grid_remove()

        # Grid expand
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(4, weight=1)

    def update_steps_ui(self):

        def display_no_steps():
            ttk.Label(self.steps_frame, text="No optionnal steps available for the current model").grid(row=0, column=0, padx=5, pady=5)

        for widget in self.steps_frame.winfo_children():
            widget.destroy()

        model_name = self.model_name_var.get()
        if not model_name:
            return

        model_name = model_name.replace(" ", "_")
        try:
            model_module = importlib.import_module(f"gui_v1.models.{model_name}")
        except ModuleNotFoundError:
            display_no_steps()
            self.current_model_module = None
            self.current_model_steps = []
            return

        steps = getattr(model_module, "STEPS", [])
        self.step_vars.clear()
        if steps:
            for i, step in enumerate(steps):
                var = tk.BooleanVar(value=step.default_enabled)
                self.step_vars[step.name] = var
                cb = ttk.Checkbutton(self.steps_frame, text=step.name, variable=var)
                cb.grid(row=0, column=i, sticky="w", padx=5, pady=5)
        else:
            display_no_steps()

        self.current_model_module = model_module
        self.current_model_steps = steps

    def open_model_settings(self):
        model_name = self.model_name_var.get()
        if not model_name or model_name == "No models":
            return

        model = get_model(model_name)
        metadata = model.get_metadata()
        weights_urls = metadata.get("weights_urls", {})

        popup = tk.Toplevel(self.root)
        popup.title(f"Settings - {model_name}")
        popup.geometry("600x450")
        popup.grab_set()

        entries = {}

        main_container = ttk.Frame(popup)
        main_container.pack(fill="both", expand=True)

        weights_group = ttk.LabelFrame(main_container)
        weights_group.pack(fill="x", padx=5, pady=5)
        bold_label = tk.Label(weights_group, text=" Weights Configuration ", font=("Arial", 10, "bold"))
        weights_group.configure(labelwidget=bold_label)

        for i, (key, default_url) in enumerate(weights_urls.items()):
            frame = ttk.LabelFrame(weights_group, text=key)
            frame.pack(fill="x", pady=5, padx=5)

            def reset_ent(ent, default_val):
                ent.delete(0, tk.END)
                ent.insert(0, default_val)

            saved_url = self.weights_overrides.get(f"{key}_url", default_url)
            ttk.Label(frame, text="URL:").grid(row=0, column=0, sticky="w")
            url_ent = ttk.Entry(frame)
            url_ent.insert(0, saved_url)
            url_ent.grid(row=0, column=1, sticky="ew", padx=5)
            ttk.Button(frame, text="Default", width=7, command=lambda ent=url_ent, d=default_url: reset_ent(ent, d)).grid(row=0, column=2, sticky="w", padx=5)

            saved_path = self.weights_overrides.get(f"{key}_path", "")
            ttk.Label(frame, text="Local Path:").grid(row=1, column=0, sticky="w")
            path_ent = ttk.Entry(frame)
            path_ent.insert(0, saved_path)
            path_ent.grid(row=1, column=1, sticky="ew", padx=5)

            def browse_weights():
                p = filedialog.askopenfilename(filetypes=[("Weights", "*.pth *.bin")])
                if p:
                    path_ent.delete(0, tk.END)
                    path_ent.insert(0, p)

            path_btns_frame = ttk.Frame(frame)
            path_btns_frame.grid(row=1, column=2, sticky="w")

            ttk.Button(path_btns_frame, text="Default", width=7, command=lambda ent=path_ent: reset_ent(ent, "")).pack(side="left", padx=5)
            ttk.Button(path_btns_frame, text="...", width=3, command=browse_weights).pack(side="left", padx=5)

            frame.columnconfigure(1, weight=1)
            entries[key] = {"url_widget": url_ent, "path_widget": path_ent}

        def save():
            for key, widgets in entries.items():
                current_url = widgets["url_widget"].get().strip()
                current_path = widgets["path_widget"].get().strip()

                self.weights_overrides[f"{key}_url"] = current_url
                self.weights_overrides[f"{key}_path"] = current_path
                self.weights_overrides[key] = current_path if current_path else current_url
            popup.destroy()

        ttk.Button(popup, text="Save Configuration", command=save).pack(padx=5, pady=5)

    # ---------- Events ----------

    def bind_events(self):
        c = self.preview_canvas

        # Zoom
        c.bind("<MouseWheel>", self.on_zoom)
        c.bind("<Button-4>", self.on_zoom)
        c.bind("<Button-5>", self.on_zoom)

        # Interations
        c.bind("<ButtonPress-1>", self.start_interaction)
        c.bind("<B1-Motion>", self.do_interaction)
        c.bind("<ButtonRelease-1>", self.end_interaction)

        c.bind("<ButtonPress-3>", self.start_interaction)
        c.bind("<B3-Motion>", self.do_interaction)
        c.bind("<ButtonRelease-3>", self.end_interaction)

        # Hover cursor
        c.bind("<Motion>", self.update_cursor)

        # Model change
        self.model_combobox.bind("<<ComboboxSelected>>", self.on_model_change)
        self.on_model_change()

        # Bind edit text
        self.preview_canvas.tag_bind("text", "<Double-1>", self.edit_text)

    def on_model_change(self, event=None):
        self.update_steps_ui()
        self.weights_overrides = {}

    # ---------- Toggle modes ----------

    def toggle_mode(self, mode):
        if self.add_mode == mode:
            self.add_mode = None
        else:
            self.add_mode = mode

        # Ensure mutual exclusivity
        self.add_node_btn.state(["pressed"] if self.add_mode=="node" else ["!pressed"])
        self.add_corner_btn.state(["pressed"] if self.add_mode=="corner" else ["!pressed"])
        self.add_text_btn.state(["pressed"] if self.add_mode=="text" else ["!pressed"])

    # ---------- File ----------

    def browse_input(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.gif *.pgm *.ppm")])
        if path:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, path)
            self.update_preview()

    def browse_output(self):
        path = filedialog.asksaveasfilename(defaultextension=".nwk", filetypes=[("Newick files", "*.nwk")])
        if path:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, path)

    # ---------- Preview ----------

    def update_preview(self):
        path = self.input_entry.get()
        try:
            self.preview_image = Image.open(path).convert("RGB")
            self.zoom = 1.0
            self.pan_x = 0
            self.pan_y = 0
            self.add_node_btn.config(state="normal")
            self.add_corner_btn.config(state="normal")
            self.clear_points_btn.config(state="normal")
            self.add_text_btn.config(state="normal")
            self.clear_texts_btn.config(state="normal")
        except Exception:
            self.preview_image = None
            self.add_node_btn.config(state="disabled")
            self.add_corner_btn.config(state="disabled")
            self.clear_points_btn.config(state="disabled")
            self.add_text_btn.config(state="disabled")
            self.clear_texts_btn.config(state="disabled")
        self.points.clear()
        self.redraw_preview()

    def redraw_preview(self, event=None, force=False):
        self.preview_canvas.delete("all")
        if not self.preview_image:
            return

        canvas_w = self.preview_canvas.winfo_width()
        canvas_h = self.preview_canvas.winfo_height()
        img_w = self.preview_image.width
        img_h = self.preview_image.height
        base_ratio = min(canvas_w / img_w, canvas_h / img_h)
        ratio = base_ratio * self.zoom
        self.base_ratio = base_ratio

        # Draw image
        if force or self.zoomed_image is None or self.zoom != self.last_zoom:
            new_w = int(self.preview_image.width * ratio)
            new_h = int(self.preview_image.height * ratio)
            self.zoomed_image = self.preview_image.resize((new_w, new_h), Image.NEAREST)
            self.last_zoom = self.zoom
            self.last_pan = None

        if force or self.last_pan is None or self.last_pan != (self.pan_x, self.pan_y):
            self.tk_image = ImageTk.PhotoImage(self.zoomed_image)
            self.last_pan = (self.pan_x, self.pan_y)

        self.preview_canvas.create_image(
            canvas_w//2 + self.pan_x,
            canvas_h//2 + self.pan_y,
            image=self.tk_image,
            anchor="center"
        )

        # Draw segments
        for seg in self.segments:
            (x1, y1), (x2, y2) = seg
            p1 = Point(x1, y1)
            p2 = Point(x2, y2)

            def shorten(p_start, p_end, offset=5):
                dx = p_end.x - p_start.x
                dy = p_end.y - p_start.y
                length = (dx**2 + dy**2)**0.5
                if length == 0:
                    return None
                factor = offset / length
                new_start = Point(p_start.x + dx*factor, p_start.y + dy*factor)
                new_end = Point(p_end.x - dx*factor, p_end.y - dy*factor)
                return new_start, new_end

            result = shorten(p1, p2, offset=0)
            if result is not None:
                s1, s2 = result
                sx1, sy1 = self.image_to_screen(s1)
                sx2, sy2 = self.image_to_screen(s2)
                self.preview_canvas.create_line(sx1, sy1, sx2, sy2, fill="green", width=7)

        # Draw texts
        for i, txt in enumerate(self.texts):
            x1, y1, x2, y2 = txt["bbox"]
            cx_img = x1
            cy_img = y1 + (y2 - y1) / 2
            cx, cy = self.image_to_screen(Point(cx_img, cy_img))

            self.preview_canvas.create_text(cx, cy, text=txt["text"], tags=("text", f"text_{i}"), fill="red", font=("Arial", 12), anchor="w")

        # Draw points
        for pt in self.points:
            screen_x, screen_y = self.image_to_screen(pt)
            color = "red" if pt.type == "node" else "blue"
            r = 5
            self.preview_canvas.create_oval(screen_x-r, screen_y-r, screen_x+r, screen_y+r, fill=color, outline="black")

    # ---------- Coordinate transforms ----------

    def screen_to_image(self, pt):
        canvas_w = self.preview_canvas.winfo_width()
        canvas_h = self.preview_canvas.winfo_height()
        img_w = self.preview_image.width
        img_h = self.preview_image.height
        ratio = self.base_ratio * self.zoom
        x = (pt[0] - canvas_w//2 - self.pan_x)/ratio
        y = (pt[1] - canvas_h//2 - self.pan_y)/ratio
        return x, y

    def image_to_screen(self, pt):
        canvas_w = self.preview_canvas.winfo_width()
        canvas_h = self.preview_canvas.winfo_height()
        ratio = self.base_ratio * self.zoom
        x = canvas_w//2 + self.pan_x + pt.x * ratio
        y = canvas_h//2 + self.pan_y + pt.y * ratio
        return x, y

    # ---------- Zoom ----------

    def on_zoom(self, event):
        if not self.preview_image:
            return

        cursor_x = event.x
        cursor_y = event.y

        img_x_before, img_y_before = self.screen_to_image((cursor_x, cursor_y))

        zoom_factor = 1.1 if (getattr(event, "delta", 0) > 0 or getattr(event, "num", None) == 4) else 0.9
        self.zoom *= zoom_factor

        min_zoom = 0.1
        max_zoom = 7.0
        self.zoom = max(min(self.zoom, max_zoom), min_zoom)

        screen_x_after, screen_y_after = self.image_to_screen(Point(img_x_before, img_y_before))
        self.pan_x += cursor_x - screen_x_after
        self.pan_y += cursor_y - screen_y_after

        self.redraw_preview()

    # ---------- Interaction ----------

    def get_hovered_point(self, event, threshold=5):
        for pt in self.points:
            sx, sy = self.image_to_screen(pt)
            if abs(event.x - sx) <= threshold and abs(event.y - sy) <= threshold:
                return pt
        return None

    def get_hovered_text_id(self, event):
        items = self.preview_canvas.find_overlapping(
            event.x, event.y,
            event.x, event.y
        )

        for item in reversed(items):
            for tag in self.preview_canvas.gettags(item):
                if tag.startswith("text_"):
                    return int(tag.split("_")[1])
        return None

    def start_interaction(self, event):
        # Check if clicking on a point or a text
        self.selected_point = self.get_hovered_point(event)
        self.selected_text_id = self.get_hovered_text_id(event)

        if event.num == 3:
            self.drag_start = (event.x, event.y)
            self.drag_type = "right"
            return

        if self.selected_point is not None:
            return

        if self.selected_text_id is not None or self.add_mode is None:
            self.drag_start = (event.x, event.y)
            self.drag_type = "left"
            return

        if self.add_mode == "node" or self.add_mode == "corner":
            # Add point
            x, y = self.screen_to_image((event.x, event.y))
            self.points.append(Point(x, y, self.add_mode))
            self.update_segments()
            self.redraw_preview()
        elif self.add_mode == "text":
            x, y = self.screen_to_image((event.x, event.y))
            self.texts.append({
                "text": "Text",
                "bbox": (x, y, x, y)
            })
            self.redraw_preview()

    def do_interaction(self, event):
        if self.drag_type == "right" or (self.drag_type == "left" and self.selected_point is None and self.selected_text_id is None):
            self.selected_point = None
            self.selected_text_id = None

            # Drag image
            dx = event.x - self.drag_start[0]
            dy = event.y - self.drag_start[1]
            self.pan_x += dx
            self.pan_y += dy
            self.drag_start = (event.x, event.y)
        elif self.selected_point is not None:
            # Drag point
            x, y = self.screen_to_image((event.x, event.y))
            self.selected_point.x = x
            self.selected_point.y = y
            self.update_segments()
        elif self.selected_text_id is not None:
            txt = self.texts[self.selected_text_id]
            px, py = self.screen_to_image(self.drag_start)
            cx, cy = self.screen_to_image((event.x, event.y))
            dx = cx - px
            dy = cy - py
            x1, y1, x2, y2 = txt["bbox"]
            txt["bbox"] = (
                x1 + dx,
                y1 + dy,
                x2 + dx,
                y2 + dy,
            )
            self.drag_start = (event.x, event.y)

        self.redraw_preview()

    def end_interaction(self, event):
        if self.zoom >= 4:
                self.redraw_preview()

        if event.num == 3:
            if self.selected_point is not None:
                self.points.remove(self.selected_point)
                self.update_segments()
            elif self.selected_text_id is not None:
                self.texts.pop(self.selected_text_id)
            self.redraw_preview()

        self.drag_start = None
        self.drag_type = None
        self.selected_point = None
        self.selected_text_id = None

    def clear_points(self):
        self.points.clear()
        self.update_segments()
        self.redraw_preview()

    def clear_texts(self):
        self.texts.clear()
        self.redraw_preview()

    def update_cursor(self, event):
        hovered_point = self.get_hovered_point(event)
        hovered_text = self.get_hovered_text_id(event)

        if hovered_point is not None or hovered_text is not None:
            self.preview_canvas.config(cursor="hand2")
        else:
            self.preview_canvas.config(cursor="")

    def edit_text(self, event):
        index = self.get_hovered_text_id(event)

        x1, y1, x2, y2 = self.texts[index]["bbox"]
        cx_img = x1 + (x2 - x1) / 2
        cy_img = y1 + (y2 - y1) / 2
        cx, cy = self.image_to_screen(Point(cx_img, cy_img))

        old_text = self.texts[index]["text"]

        entry = tk.Entry(self.preview_canvas)
        entry.insert(0, self.texts[index]["text"])

        entry_width = 110
        entry_height = 30
        entry.place(x=cx - entry_width/2 - entry_width, y=cy - entry_height/2, width=entry_width, height=entry_height)
        entry.focus_set()

        def save_text(event):
            self.texts[index]["text"] = entry.get()
            entry.destroy()
            self.redraw_preview()

        def cancel_edit(event=None):
            entry.destroy()
            self.redraw_preview()

        entry.bind("<Return>", save_text)
        entry.bind("<FocusOut>", save_text)
        entry.bind("<Escape>", cancel_edit)

    def update_segments(self):
        if not self.preview_image or not self.points:
            self.segments = []
            return

        img_w = self.preview_image.width
        img_h = self.preview_image.height

        points_norm = scale_points(self.points, scale_width=1.0/img_w, scale_height=1.0/img_h)
        segments = build_segments(points_norm)
        self.segments = segments if segments else []
        self.segments = scale_segments(self.segments, scale_width=img_w, scale_height=img_h)

    # ---------- Output ----------

    def show_output(self, text):
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.config(state="disabled")

    # ---------- Conversion ----------

    def run_conversion(self):
        try:
            input_path = self.input_entry.get().strip()
            output_file = self.output_entry.get().strip()
            model_name = self.model_name_var.get()

            if not input_path or not model_name or model_name == "No models":
                raise ValueError("Invalid input or model")

            img_w = self.preview_image.width
            img_h = self.preview_image.height

            self.texts = [{
                "text": "test",
                "bbox": (20, 20, 70, 50)
            }]

            model = get_model(model_name)
            model.load(
                cropping_model_weights_path_or_url=self.weights_overrides.get("Cropping"),
                denoising_model_weights_path_or_url=self.weights_overrides.get("Denoising"),
                nodesdetection_model_weights_path_or_url=self.weights_overrides.get("Nodes Detection")
            )

            if self.current_model_module is None:
                newick = model.convert(np.array(self.preview_image))
                newick_str = newick.to_string()
            else:
                for step in self.current_model_steps:
                    step.enabled = self.step_vars[step.name].get()

                if self.points:
                    points_norm = scale_points(self.points, scale_width=1.0/img_w, scale_height=1.0/img_h)
                    newick = model.build_newick(points_norm, texts=self.texts)
                    newick_str = newick.to_string()
                else:
                    newick, points, texts = self.current_model_module.run_steps(model, np.array(self.preview_image), steps=self.current_model_steps)

                    self.points = scale_points(points, scale_width=img_w, scale_height=img_h)
                    self.texts = scale_texts(texts, scale_width=img_w, scale_height=img_h)
                    self.update_segments()

                    newick_str = newick.to_string()

            if output_file:
                with open(output_file, "w") as f:
                    f.write(newick_str)

            self.root.after(0, lambda: self.show_output(newick_str))
            self.root.after(0, lambda: messagebox.showinfo("Done", "Generation finished"))

        except Exception as e:
            self.root.after(0, lambda e=e: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, self.progress.stop)
            self.root.after(0, self.progress.grid_remove)
            self.root.after(0, lambda: self.convert_button.config(state="normal"))
            self.root.after(0, self.redraw_preview)

    def convert(self):
        self.convert_button.config(state="disabled")
        self.progress.grid()
        self.progress.start(10)
        threading.Thread(target=self.run_conversion).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = ITPTGUI(root)
    root.mainloop()
