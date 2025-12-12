import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

MODELS = ["model_a", "model_b", "model_c"]  # TODO

class ITPTGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ITPTGUI")
        self.root.geometry("900x700")

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
        self.model_var = tk.StringVar(value=MODELS[0])
        ttk.Combobox(root, textvariable=self.model_var, values=MODELS, state="readonly").grid(row=2, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(root, text="Image preview:").grid(row=3, column=0, columnspan=3, sticky="w", padx=5, pady=5)
        self.preview_canvas = tk.Canvas(root, bg="white")
        self.preview_canvas.grid(row=4, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        self.preview_canvas.bind("<Configure>", self.resize_preview)
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
                self.preview_image = tk.PhotoImage(file=path)
            except Exception:
                self.preview_image = None
        else:
            self.preview_image = None
        self.redraw_preview()

    def resize_preview(self, event):
        self.redraw_preview()

    def redraw_preview(self):
        self.preview_canvas.delete("all")
        if not self.preview_image:
            return

        canvas_w = self.preview_canvas.winfo_width()
        canvas_h = self.preview_canvas.winfo_height()
        img_w = self.preview_image.width()
        img_h = self.preview_image.height()

        ratio_w = img_w / canvas_w
        ratio_h = img_h / canvas_h
        ratio = max(ratio_w, ratio_h, 1)

        if ratio > 1:
            self.tk_image = self.preview_image.subsample(int(ratio), int(ratio))
        else:
            self.tk_image = self.preview_image

        self.preview_canvas.create_image(canvas_w//2, canvas_h//2, image=self.tk_image, anchor="center")

    def show_output(self, text: str):
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.config(state="disabled")

    def convert(self):
        self.convert_button.config(state="disabled")

        input_file = self.input_entry.get()
        output_file = self.output_entry.get()
        model = self.model_var.get()

        if not input_file:
            messagebox.showerror("Error", "Please select an input file.")
            self.convert_button.config(state="normal")
            return

        # TODO appel à ITPT

        self.show_output("test")

        messagebox.showinfo("Done", f"Generation finished using '{model}'.")
        self.convert_button.config(state="normal")

if __name__ == "__main__":
    root = tk.Tk()
    app = ITPTGUI(root)
    root.mainloop()
