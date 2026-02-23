import os
import customtkinter as ctk
from tkinter import filedialog, messagebox

from gui_vctk.core.pipeline import run_pipeline
from gui_vctk.core.newick import compute_newick


class Page2(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, corner_radius=0)

        self.folder_path: str | None = None
        self._running = False

        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self,
            text="Traitement par lot",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(20, 10))

        self.folder_label = ctk.CTkLabel(self, text="Aucun dossier sélectionné")
        self.folder_label.pack(pady=(0, 10))

        controls = ctk.CTkFrame(self, fg_color="transparent")
        controls.pack(pady=5)

        self.btn_choose = ctk.CTkButton(controls, text="Choisir un dossier", command=self.choose_folder)
        self.btn_choose.pack(side="left", padx=6)

        self.btn_run = ctk.CTkButton(controls, text="Calculer les Newick", command=self.process_folder)
        self.btn_run.pack(side="left", padx=6)

        # --- "tqdm" GUI ---
        self.progress_label = ctk.CTkLabel(self, text="Progression: 0/0")
        self.progress_label.pack(pady=(10, 4))

        self.progress = ctk.CTkProgressBar(self)
        self.progress.pack(fill="x", padx=20, pady=(0, 10))
        self.progress.set(0)

        self.log_box = ctk.CTkTextbox(self, height=420)
        self.log_box.pack(fill="both", expand=True, padx=20, pady=(0, 20))

    def log(self, text: str):
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.update_idletasks()

    def _set_running(self, running: bool):
        self._running = running
        state = "disabled" if running else "normal"
        self.btn_choose.configure(state=state)
        self.btn_run.configure(state=state)

    def choose_folder(self):
        if self._running:
            return
        path = filedialog.askdirectory(title="Choisir un dossier contenant les PNG")
        if path:
            self.folder_path = path
            self.folder_label.configure(text=path)

    def process_folder(self):
        if self._running:
            return
        if not self.folder_path:
            messagebox.showwarning("Info", "Choisissez un dossier d'abord.")
            return

        png_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(".png")]
        png_files.sort()

        if not png_files:
            messagebox.showwarning("Info", "Aucun PNG trouvé.")
            return

        newick_dir = os.path.join(self.folder_path, "newick")
        os.makedirs(newick_dir, exist_ok=True)

        total = len(png_files)
        self.progress.set(0)
        self.progress_label.configure(text=f"Progression: 0/{total}")
        self.log_box.delete("1.0", "end")

        self._set_running(True)
        self.log(f"Traitement de {total} image(s)...")
        self.log(f"Dossier sortie: {newick_dir}")

        ok_count = 0

        for i, file in enumerate(png_files, start=1):
            img_path = os.path.join(self.folder_path, file)
            name = os.path.splitext(file)[0]

            # update tqdm-like UI
            self.progress.set(i / total)
            self.progress_label.configure(text=f"Progression: {i}/{total}")
            self.update_idletasks()

            try:
                self.log(f"\n[{i}/{total}] → Pipeline : {file}")
                points = run_pipeline(img_path)

                self.log(f"[{i}/{total}] → Newick : {file}")
                newick_str = compute_newick(points)

                out_path = os.path.join(newick_dir, name + ".nwk")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(newick_str if newick_str.endswith("\n") else newick_str + "\n")

                self.log(f"[{i}/{total}] ✔ Sauvé : {name}.nwk")
                ok_count += 1

            except Exception as e:
                self.log(f"[{i}/{total}] ✖ Erreur sur {file} : {str(e)}")

        self._set_running(False)
        messagebox.showinfo("Terminé", f"OK: {ok_count}/{total}\nNewick générés dans :\n{newick_dir}")
