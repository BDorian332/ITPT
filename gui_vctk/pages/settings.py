import customtkinter as ctk
from gui_vctk.core.settings_state import SETTINGS


class SettingsPage(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, corner_radius=0)

        ctk.CTkLabel(
            self,
            text="Paramètres",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(20, 10))

        # --------------------
        # VERSION (uniquement v1)
        # --------------------
        version_box = ctk.CTkFrame(self)
        version_box.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(version_box, text="Pipeline").pack(anchor="w", padx=10, pady=(10, 6))

        self.version_var = ctk.StringVar(value="v1")

        ctk.CTkRadioButton(
            version_box,
            text="v1",
            variable=self.version_var,
            value="v1",
            command=self._update
        ).pack(anchor="w", padx=10, pady=(0, 10))

        SETTINGS.version = "v1"

        # --------------------
        # PREPROCESSING
        # --------------------
        pre_box = ctk.CTkFrame(self)
        pre_box.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(pre_box, text="Pré-traitement").pack(anchor="w", padx=10, pady=(10, 6))

        self.crop_var = ctk.BooleanVar(value=getattr(SETTINGS, "cropping", True))
        self.denoise_var = ctk.BooleanVar(value=getattr(SETTINGS, "denoising", True))

        ctk.CTkCheckBox(
            pre_box, text="Cropping",
            variable=self.crop_var,
            command=self._update
        ).pack(anchor="w", padx=10)

        ctk.CTkCheckBox(
            pre_box, text="Denoising",
            variable=self.denoise_var,
            command=self._update
        ).pack(anchor="w", padx=10, pady=(0, 10))

        # --------------------
        # POSTPROCESSING
        # --------------------
        post_box = ctk.CTkFrame(self)
        post_box.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(post_box, text="Post-traitement").pack(anchor="w", padx=10, pady=(10, 6))

        self.correction_var = ctk.BooleanVar(value=getattr(SETTINGS, "correction", True))

        ctk.CTkCheckBox(
            post_box,
            text="Correction",
            variable=self.correction_var,
            command=self._update
        ).pack(anchor="w", padx=10, pady=(0, 10))

        # --------------------
        # NEWICK
        # --------------------
        newick_box = ctk.CTkFrame(self)
        newick_box.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(newick_box, text="Newick").pack(anchor="w", padx=10, pady=(10, 6))

        # margin default 5
        default_margin = int(getattr(SETTINGS, "newick_margin", 5))
        self.margin_var = ctk.IntVar(value=default_margin)

        row = ctk.CTkFrame(newick_box, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=(0, 6))

        ctk.CTkLabel(row, text="Margin").pack(side="left")

        self.margin_entry = ctk.CTkEntry(row, width=80)
        self.margin_entry.pack(side="left", padx=(10, 0))
        self.margin_entry.insert(0, str(default_margin))

        ctk.CTkLabel(
            newick_box,
            text="(plus grand = plus tolérant pour relier les branches)",
            text_color="gray"
        ).pack(anchor="w", padx=10, pady=(0, 10))

        # update when user leaves field / presses enter
        self.margin_entry.bind("<Return>", lambda e: self._update())
        self.margin_entry.bind("<FocusOut>", lambda e: self._update())

        # push initial values
        self._update()

    def _update(self):
        SETTINGS.version = "v1"
        SETTINGS.cropping = self.crop_var.get()
        SETTINGS.denoising = self.denoise_var.get()
        SETTINGS.correction = self.correction_var.get()

        # margin (int >= 0) with fallback to 5
        try:
            v = int(self.margin_entry.get().strip())
            if v < 0:
                v = 0
        except Exception:
            v = 5
            self.margin_entry.delete(0, "end")
            self.margin_entry.insert(0, "5")

        SETTINGS.newick_margin = v
