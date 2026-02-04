import os
import tkinter as tk
import threading
import time
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk
from itpt.models import get_list, get_model

class ITPTGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ITPTGUI")
        self.root.geometry("900x700")
        self.model_names = get_list()

        ttk.Label(root, text="Input image:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.input_entry = ttk.Entry(root)
        self.input_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.input_entry.bind("<KeyRelease>", lambda e: self.update_preview())
        ttk.Button(root, text="Browse", command=self.browse_input).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(root, text="Output file (optional):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.output_entry = ttk.Entry(root)
        self.output_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(root, text="Browse", command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)

        ttk.Label(root, text="Select model:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        if not self.model_names:
            self.model_name_var = tk.StringVar(value="No models available")
            self.model_combobox = ttk.Combobox(root, textvariable=self.model_name_var, values=self.model_names, state="disabled")
        else:
            self.model_name_var = tk.StringVar(value=self.model_names[0])
            self.model_combobox = ttk.Combobox(root, textvariable=self.model_name_var, values=self.model_names, state="readonly")
        self.model_combobox.grid(row=2, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(root, text="Image preview:").grid(row=3, column=0, columnspan=3, sticky="w", padx=5, pady=5)
        self.preview_canvas = tk.Canvas(root, bg="white")
        self.preview_canvas.grid(row=4, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        self.preview_canvas.bind("<Configure>", self.redraw_preview)
        self.preview_image = None
        self.tk_image = None

        ttk.Label(root, text="Output:").grid(row=5, column=0, columnspan=3, sticky="w", padx=5, pady=5)

        output_frame = ttk.Frame(root)
        output_frame.grid(row=6, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)

        self.output_text = tk.Text(output_frame, height=5, wrap="word")
        self.output_text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(output_frame, orient="vertical", command=self.output_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.output_text.configure(yscrollcommand=scrollbar.set)

        self.convert_button = ttk.Button(root, text="Convert", command=self.convert)
        self.convert_button.grid(row=7, column=0, columnspan=3, padx=5, pady=5)

        self.progress = ttk.Progressbar(
            root,
            mode="indeterminate"
        )
        self.progress.grid(row=8, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        self.progress.grid_remove()

        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(4, weight=1)

    def browse_input(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if path:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, path)
            self.update_preview()

    def browse_output(self):
        path = filedialog.asksaveasfilename(defaultextension=".nwk", filetypes=[("Newick files", "*.nwk")])
        if path:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, path)

    def update_preview(self):
        path = self.input_entry.get()
        if path and path.lower().endswith((".png", ".gif", ".pgm", ".ppm")):
            try:
                self.preview_image = Image.open(path)
            except Exception:
                self.preview_image = None
        else:
            self.preview_image = None
        self.redraw_preview()

    def redraw_preview(self, event=None):
        self.preview_canvas.delete("all")
        if not self.preview_image:
            return

        canvas_w = self.preview_canvas.winfo_width()
        canvas_h = self.preview_canvas.winfo_height()

        img_w = self.preview_image.width
        img_h = self.preview_image.height

        ratio_w = canvas_w / img_w
        ratio_h = canvas_h / img_h
        ratio = min(ratio_w, ratio_h)

        new_w = max(1, int(img_w * ratio))
        new_h = max(1, int(img_h * ratio))

        resized = self.preview_image.resize((new_w, new_h))

        self.tk_image = ImageTk.PhotoImage(resized)

        self.preview_canvas.create_image(
            canvas_w // 2,
            canvas_h // 2,
            image=self.tk_image,
            anchor="center"
        )

    def show_output(self, text: str):
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.config(state="disabled")

    def _run_conversion(self):
        try:
            input_file = self.input_entry.get().strip()
            output_file = self.output_entry.get().strip()
            model_name = self.model_name_var.get()

            if not input_file or model_name not in self.model_names:
                raise ValueError("Invalid input or model")

            model = get_model(model_name)
            model.load()
            tree = model.convert(input_file)

            #time.sleep(5)

            if output_file:
                with open(output_file, "w") as f:
                    f.write(tree)

            self.root.after(0, lambda: self.show_output(tree))
            self.root.after(0, lambda: messagebox.showinfo("Done", "Generation finished"))

        except Exception as e:
            err = str(e)
            self.root.after(0, lambda: messagebox.showerror("Error", err))

        finally:
            self.root.after(0, self.progress.stop)
            self.root.after(0, self.progress.grid_remove)
            self.root.after(0, lambda: self.convert_button.config(state="normal"))

    def convert(self):
        self.convert_button.config(state="disabled")

        self.progress.grid()
        self.progress.start(10)

        thread = threading.Thread(target=self._run_conversion)
        thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = ITPTGUI(root)
    root.mainloop()
