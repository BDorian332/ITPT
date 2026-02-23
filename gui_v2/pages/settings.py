import customtkinter as ctk
from gui_v2.core.settings_state import SETTINGS


class SettingsPage(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, corner_radius=0)

        ctk.CTkLabel(
            self,
            text="Pipeline Settings",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(20, 10))

        # --------------------
        # VERSION
        # --------------------
        version_box = ctk.CTkFrame(self)
        version_box.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(version_box, text="Pipeline Version").pack(anchor="w", padx=10, pady=(10, 6))

        self.version_var = ctk.StringVar(value=SETTINGS.version)

        ctk.CTkRadioButton(version_box, text="v0", variable=self.version_var, value="v0",
                           command=self._update).pack(anchor="w", padx=10)
        ctk.CTkRadioButton(version_box, text="v1", variable=self.version_var, value="v1",
                           command=self._update).pack(anchor="w", padx=10)

        # --------------------
        # PREPROCESSING
        # --------------------
        pre_box = ctk.CTkFrame(self)
        pre_box.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(pre_box, text="Pre-processing").pack(anchor="w", padx=10, pady=(10, 6))

        self.crop_var = ctk.BooleanVar(value=SETTINGS.cropping)
        self.denoise_var = ctk.BooleanVar(value=SETTINGS.denoising)

        ctk.CTkCheckBox(pre_box, text="Cropping", variable=self.crop_var,
                        command=self._update).pack(anchor="w", padx=10)
        ctk.CTkCheckBox(pre_box, text="Denoising", variable=self.denoise_var,
                        command=self._update).pack(anchor="w", padx=10)

        # --------------------
        # POSTPROCESSING
        # --------------------
        post_box = ctk.CTkFrame(self)
        post_box.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(post_box, text="Post-processing").pack(anchor="w", padx=10, pady=(10, 6))

        self.clean_var = ctk.BooleanVar(value=SETTINGS.post_clean)
        self.merge_var = ctk.BooleanVar(value=SETTINGS.post_merge)

        ctk.CTkCheckBox(post_box, text="Cleaning", variable=self.clean_var,
                        command=self._update).pack(anchor="w", padx=10)
        ctk.CTkCheckBox(post_box, text="Merge Nodes", variable=self.merge_var,
                        command=self._update).pack(anchor="w", padx=10)

    def _update(self):
        SETTINGS.version = self.version_var.get()
        SETTINGS.cropping = self.crop_var.get()
        SETTINGS.denoising = self.denoise_var.get()
