import sys
import queue
import threading
import traceback
import customtkinter as ctk


class _QueueWriter:
    def __init__(self, q: "queue.Queue[str]"):
        self.q = q

    def write(self, s: str):
        if s:
            self.q.put(s)

    def flush(self):
        pass


class LogPopup(ctk.CTkToplevel):
    def __init__(self, master, title: str = "Pipeline logs"):
        super().__init__(master)
        self.title(title)
        self.geometry("820x420")
        self.minsize(640, 320)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.text = ctk.CTkTextbox(self, wrap="word")
        self.text.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        bottom = ctk.CTkFrame(self)
        bottom.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 12))
        bottom.grid_columnconfigure(0, weight=1)

        self.status = ctk.CTkLabel(bottom, text="Running...")
        self.status.grid(row=0, column=0, sticky="w")


        ctk.CTkButton(bottom, text="Close", command=self.destroy).grid(row=0, column=2)

        self._q: "queue.Queue[str]" = queue.Queue()
        self._orig_out = None
        self._orig_err = None
        self._running = False

        self.after(50, self._poll)

    def clear(self):
        self.text.delete("1.0", "end")

    def append(self, s: str):
        self.text.insert("end", s)
        self.text.see("end")

    def set_status(self, s: str):
        self.status.configure(text=s)

    def start_capture(self):
        if self._running:
            return
        self._running = True
        self._orig_out = sys.stdout
        self._orig_err = sys.stderr
        sys.stdout = _QueueWriter(self._q)
        sys.stderr = _QueueWriter(self._q)

    def stop_capture(self):
        if not self._running:
            return
        sys.stdout = self._orig_out
        sys.stderr = self._orig_err
        self._running = False

    def _poll(self):
        try:
            while True:
                s = self._q.get_nowait()
                self.append(s)
        except queue.Empty:
            pass
        self.after(50, self._poll)

    def run_in_thread(self, fn, on_success=None, on_error=None):
        """
        fn: fonction longue (pipeline)
        on_success(result)
        on_error(exception)
        """
        self.start_capture()

        def _worker():
            try:
                result = fn()
            except Exception as e:
                self._q.put("\n" + traceback.format_exc() + "\n")
                self.after(0, lambda: self.set_status("Error"))
                self.after(0, self.stop_capture)
                if on_error:
                    self.after(0, lambda: on_error(e))
                return

            self.after(0, lambda: self.set_status("Done"))
            self.after(0, self.stop_capture)
            if on_success:
                self.after(0, lambda: on_success(result))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
