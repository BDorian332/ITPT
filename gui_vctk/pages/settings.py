import customtkinter as ctk
from gui_vctk.core.settings_state import SETTINGS


class SettingsPage(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, corner_radius=0)

        ctk.CTkLabel(
            self,
            text="Pipeline Settings",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(20, 10))

        # --------------------
        # VERSION (uniquement v1)
        # --------------------
        version_box = ctk.CTkFrame(self)
        version_box.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(version_box, text="Pipeline Version").pack(anchor="w", padx=10, pady=(10, 6))

        self.version_var = ctk.StringVar(value="v1")

        ctk.CTkRadioButton(
            version_box,
            text="v1",
            variable=self.version_var,
            value="v1",
            command=self._update
        ).pack(anchor="w", padx=10)

        # --------------------
        # PREPROCESSING
        # --------------------
        pre_box = ctk.CTkFrame(self)
        pre_box.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(pre_box, text="Pre-processing").pack(anchor="w", padx=10, pady=(10, 6))

        self.crop_var = ctk.BooleanVar(value=SETTINGS.cropping)
        self.denoise_var = ctk.BooleanVar(value=SETTINGS.denoising)

        ctk.CTkCheckBox(
            pre_box,
            text="Cropping",
            variable=self.crop_var,
            command=self._update
        ).pack(anchor="w", padx=10)

        ctk.CTkCheckBox(
            pre_box,
            text="Denoising",
            variable=self.denoise_var,
            command=self._update
        ).pack(anchor="w", padx=10)

        # --------------------
        # POSTPROCESSING (uniquement "Correction")
        # --------------------
        post_box = ctk.CTkFrame(self)
        post_box.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(post_box, text="Post-processing").pack(anchor="w", padx=10, pady=(10, 6))

        self.correction_var = ctk.BooleanVar(value=getattr(SETTINGS, "correction", True))

        ctk.CTkCheckBox(
            post_box,
            text="Correction",
            variable=self.correction_var,
            command=self._update
        ).pack(anchor="w", padx=10)

        # force version v1 in settings
        SETTINGS.version = "v1"

    def _update(self):
        SETTINGS.version = "v1"
        SETTINGS.cropping = self.crop_var.get()
        SETTINGS.denoising = self.denoise_var.get()
        SETTINGS.correction = self.correction_var.get()
