import os
import tkinter as tk
import threading
import importlib
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk
from itpt.models import get_list, get_model
from itpt.core.newick import Point

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
        self.tk_image = None
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start = None

        # ---- Points ----

        self.points = []  # list of Point objects
        self.add_mode = None  # "node", "corner", None
        self.selected_point = None  # for drag

        # ---- Texts ----

        self.texts = []

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

        # Canvas
        ttk.Label(root, text="Image preview:").grid(row=3, column=0, columnspan=3, sticky="w", padx=5, pady=5)
        self.preview_canvas = tk.Canvas(root, bg="white")
        self.preview_canvas.grid(row=4, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)

        # Point buttons (toggle exclusive)
        self.add_node_btn = ttk.Button(root, text="Add Node", command=lambda: self.toggle_mode("node"), state="disabled")
        self.add_corner_btn = ttk.Button(root, text="Add Corner", command=lambda: self.toggle_mode("corner"), state="disabled")
        self.add_node_btn.grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.add_corner_btn.grid(row=5, column=1, padx=5, pady=5, sticky="w")
        self.clear_points_btn = ttk.Button(root, text="Clear Points", command=self.clear_points, state="disabled")
        self.clear_points_btn.grid(row=5, column=2, padx=5, pady=5, sticky="w")

        # Output text
        ttk.Label(root, text="Output:").grid(row=6, column=0, columnspan=3, sticky="w", padx=5, pady=5)
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
        for widget in self.steps_frame.winfo_children():
            widget.destroy()

        model_name = self.model_name_var.get()
        if not model_name:
            return

        model_name = model_name.replace(" ", "_")
        try:
            model_module = importlib.import_module(f"gui_v1.models.{model_name}")
        except ModuleNotFoundError:
            self.current_model_module = None
            return

        self.step_vars.clear()
        for i, step in enumerate(getattr(model_module, "STEPS")):
            var = tk.BooleanVar(value=step.default_enabled)
            self.step_vars[step.name] = var
            cb = ttk.Checkbutton(self.steps_frame, text=step.name, variable=var)
            cb.grid(row=0, column=i, sticky="w", padx=5, pady=5)

        self.current_model_module = model_module
        self.current_model_steps = getattr(model_module, "STEPS")

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

        self.model_combobox.bind("<<ComboboxSelected>>", self.on_model_change)
        self.on_model_change()

    def on_model_change(self, event=None):
        self.update_steps_ui()

    # ---------- Toggle modes ----------

    def toggle_mode(self, mode):
        if self.add_mode == mode:
            self.add_mode = None
        else:
            self.add_mode = mode

        # Ensure mutual exclusivity
        if self.add_mode == "node":
            self.add_node_btn.state(["pressed"])
            self.add_corner_btn.state(["!pressed"])
        elif self.add_mode == "corner":
            self.add_node_btn.state(["!pressed"])
            self.add_corner_btn.state(["pressed"])
        else:
            self.add_node_btn.state(["!pressed"])
            self.add_corner_btn.state(["!pressed"])

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
        except Exception:
            self.preview_image = None
            self.add_node_btn.config(state="disabled")
            self.add_corner_btn.config(state="disabled")
            self.clear_points_btn.config(state="disabled")
        self.points.clear()
        self.redraw_preview()

    def redraw_preview(self, event=None):
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
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)
        resized = self.preview_image.resize((new_w, new_h))
        self.tk_image = ImageTk.PhotoImage(resized)
        self.preview_canvas.create_image(
            canvas_w//2 + self.pan_x,
            canvas_h//2 + self.pan_y,
            image=self.tk_image,
            anchor="center"
        )

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
        if event.delta > 0 or getattr(event, "num") == 4:
            self.zoom *= 1.1
        else:
            self.zoom *= 0.9
        self.redraw_preview()

    # ---------- Interaction ----------

    def get_hovered_point(self, event, threshold=5):
        for pt in self.points:
            sx, sy = self.image_to_screen(pt)
            if abs(event.x - sx) <= threshold and abs(event.y - sy) <= threshold:
                return pt
        return None

    def start_interaction(self, event):
        # Check if clicking on a point
        self.selected_point = self.get_hovered_point(event)

        if event.num == 3:
            self.drag_start = (event.x, event.y)
        else:
            if self.selected_point is None and self.add_mode:
                # Add point
                x, y = self.screen_to_image((event.x, event.y))
                self.points.append(Point(x, y, self.add_mode))
                self.redraw_preview()
            elif self.selected_point is None:
                self.drag_start = (event.x, event.y)

    def do_interaction(self, event):
        if self.drag_start is not None:
            self.selected_point = None

            # Drag image
            dx = event.x - self.drag_start[0]
            dy = event.y - self.drag_start[1]
            self.pan_x += dx
            self.pan_y += dy
            self.drag_start = (event.x, event.y)
            if self.zoom < 4:
                self.redraw_preview()
        elif self.selected_point:
            # Drag point
            x, y = self.screen_to_image((event.x, event.y))
            self.selected_point.x = x
            self.selected_point.y = y
            self.redraw_preview()

    def end_interaction(self, event):
        if self.zoom >= 4:
                self.redraw_preview()

        if event.num == 3 and self.selected_point is not None:
            self.points.remove(self.selected_point)
            self.redraw_preview()

        self.drag_start = None
        self.selected_point = None

    def clear_points(self):
        self.points.clear()
        self.redraw_preview()

    def update_cursor(self, event):
        cursor_hand = False

        hovered_point = self.get_hovered_point(event)

        if hovered_point is not None:
            cursor_hand = True

        self.preview_canvas.config(cursor="hand2" if cursor_hand else "")

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

            if not input_path or model_name not in self.model_names:
                raise ValueError("Invalid input or model")

            img_w = self.preview_image.width
            img_h = self.preview_image.height

            model = get_model(model_name)
            model.load()

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
                    self.redraw_preview()

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

    def convert(self):
        self.convert_button.config(state="disabled")
        self.progress.grid()
        self.progress.start(10)
        threading.Thread(target=self.run_conversion).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = ITPTGUI(root)
    root.mainloop()
