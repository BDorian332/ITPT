import tkinter as tk
import customtkinter as ctk

def _make_root():
    # Drag&Drop optionnel: si tkinterdnd2 est installé
    try:
        from tkinterdnd2 import TkinterDnD  # type: ignore
        root = TkinterDnD.Tk()
    except Exception:
        root = tk.Tk()
    return root

def run_app():
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")

    root = _make_root()
    app = WicklogenicsApp(root)
    root.mainloop()

class WicklogenicsApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Wicklogenics")
        self.root.geometry("1200x750")
        self.root.minsize(1000, 650)

        # Layout grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        from gui_vctk.componnents.navbar import Navbar
        from gui_vctk.pages.home import HomePage
        from gui_vctk.pages.page2 import Page2
        from gui_vctk.pages.settings import SettingsPage

        self.navbar = Navbar(self.root, on_navigate=self.show_page)
        self.navbar.grid(row=0, column=0, sticky="nsw")

        self.container = ctk.CTkFrame(self.root, corner_radius=0)
        self.container.grid(row=0, column=1, sticky="nsew")
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.pages: dict[str, ctk.CTkFrame] = {
            "Accueil": HomePage(self.container),
            "Traitement par lot": Page2(self.container),
            "Paramètres": SettingsPage(self.container),
        }
        for p in self.pages.values():
            p.grid(row=0, column=0, sticky="nsew")

        self.show_page("Accueil")

    def show_page(self, key: str):
        page = self.pages.get(key)
        if page:
            page.tkraise()
            self.navbar.set_active(key)


if __name__ == "__main__":
    run_app()

