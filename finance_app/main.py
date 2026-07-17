import tkinter as tk

from .config import CONFIG
from .database import init_db
from .gui import FinanceAnalyticsApp
from .utils import setup_logging


def main() -> None:
    setup_logging(CONFIG.log_path)
    init_db(CONFIG.db_path)
    root = tk.Tk()
    FinanceAnalyticsApp(root, CONFIG)
    root.mainloop()


if __name__ == "__main__":
    main()

