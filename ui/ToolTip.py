import tkinter as tk

# =========================================================
#Módulo 1: TRATAMENTO DE DADOS E PREPARAÇÃO
# ========================================================
# UI/UX: Tooltip (aparece ao passar o mouse)
class ToolTip:
    """Tooltip simples para Tkinter/ttk (mostra texto ao passar o mouse)."""
    def __init__(self, widget, text: str, delay_ms: int = 250):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.tip = None
        self._after_id = None

        self.widget.bind("<Enter>", self._schedule)
        self.widget.bind("<Leave>", self._hide)
        self.widget.bind("<ButtonPress>", self._hide)

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        if self.tip or not self.text:
            return

        try:
            x = self.widget.winfo_rootx() + 18
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        except Exception:
            return

        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=("Arial", 9)
        )
        label.pack(ipadx=8, ipady=4)

    def _hide(self, _event=None):
        self._cancel()
        if self.tip:
            try:
                self.tip.destroy()
            except Exception:
                pass
            self.tip = None