import logging
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .analytics import (
    anomalous_days,
    company_summary,
    dataset_overview,
    financial_ratios,
    return_correlation,
    technical_indicators,
    technical_signal,
)
from .config import AppConfig
from .database import FINANCE_COLUMNS, STOCK_COLUMNS, clear_database, execute, init_db, query_df
from .forecasting import ALGORITHMS, backtest_series, finance_forecast, forecast_series
from .importer import import_base_data_folder, import_finance_csv, import_stock_csv

try:
    from ttkbootstrap import Style  # type: ignore
    HAS_TTKBOOTSTRAP = True
except Exception:
    Style = None  # type: ignore
    HAS_TTKBOOTSTRAP = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except Exception:
    plt = None  # type: ignore
    FigureCanvasTkAgg = None  # type: ignore
    Figure = None  # type: ignore
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)


class FinanceAnalyticsApp:
    def __init__(self, root: tk.Tk, config: AppConfig):
        self.root = root
        self.config = config
        self.root.title("Finance Analytics: анализ больших финансовых данных")
        self.root.geometry("1400x850")

        if HAS_TTKBOOTSTRAP and Style is not None:
            Style(theme=self.config.theme)

        self.main = ttk.Frame(root)
        self.main.pack(fill=tk.BOTH, expand=True)
        self.left_container = ttk.Frame(self.main)
        self.left_container.pack(side=tk.LEFT, fill=tk.Y)
        self.left_canvas = tk.Canvas(self.left_container, width=285, highlightthickness=0)
        self.left_scrollbar = ttk.Scrollbar(self.left_container, orient="vertical", command=self.left_canvas.yview)
        self.left_canvas.configure(yscrollcommand=self.left_scrollbar.set)
        self.left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.left_canvas.pack(side=tk.LEFT, fill=tk.Y)
        self.left = ttk.Frame(self.left_canvas, padding=10)
        self.left_window = self.left_canvas.create_window((0, 0), window=self.left, anchor="nw")
        self.left.bind("<Configure>", self._update_left_scrollregion)
        self.left_canvas.bind("<Configure>", self._resize_left_panel)
        self.left_canvas.bind("<Enter>", lambda _: self._bind_mousewheel())
        self.left_canvas.bind("<Leave>", lambda _: self._unbind_mousewheel())
        self.right = ttk.Frame(self.main, padding=10)
        self.right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(self.left, text="Finance Analytics", font=("Segoe UI", 16, "bold")).pack(pady=8)
        self._build_controls()

        self.text = tk.Text(self.right, wrap=tk.WORD, font=("Consolas", 11))
        self.text.pack(fill=tk.BOTH, expand=True)

        self.ensure_db()
        self.show(
            "Готово.\n\n"
            "Приложение загружает данные Yahoo Finance из CSV, сохраняет их в SQLite, "
            "позволяет просматривать и редактировать записи, строит аналитические показатели, "
            "графики и прогнозы.\n\n"
            "Рекомендуемый старт: импортируйте папку finance_bigdata_project, затем откройте обзор датасета."
        )

    def _build_controls(self) -> None:
        db_box = ttk.LabelFrame(self.left, text="База данных и импорт", padding=8)
        db_box.pack(fill=tk.X, pady=5)
        self._btn(db_box, "Инициализировать БД", self.ensure_db, "secondary").pack(fill=tk.X, pady=2)
        self._btn(db_box, "Очистить базу данных", self.clear_database_window, "danger").pack(fill=tk.X, pady=2)
        self._btn(db_box, "Импорт CSV котировок", self.import_stock_csv_window, "primary").pack(fill=tk.X, pady=2)
        self._btn(db_box, "Импорт CSV отчетности", self.import_finance_csv_window, "primary").pack(fill=tk.X, pady=2)
        self._btn(db_box, "Импорт папки base data", self.import_base_data_folder_window, "info").pack(fill=tk.X, pady=2)
        self._btn(db_box, "Обзор датасета", self.dataset_overview, "info").pack(fill=tk.X, pady=2)

        view_box = ttk.LabelFrame(self.left, text="Просмотр", padding=8)
        view_box.pack(fill=tk.X, pady=5)
        self._btn(view_box, "Показать акции", lambda: self.load_table("stock_prices"), "info").pack(fill=tk.X, pady=2)
        self._btn(view_box, "Показать отчетность", lambda: self.load_table("finance_statement"), "info").pack(fill=tk.X, pady=2)
        self._btn(view_box, "Таблица акций", lambda: self.view_table_window("stock_prices"), "secondary").pack(fill=tk.X, pady=2)
        self._btn(view_box, "Таблица отчетности", lambda: self.view_table_window("finance_statement"), "secondary").pack(fill=tk.X, pady=2)

        stock_crud = ttk.LabelFrame(self.left, text="CRUD: акции", padding=8)
        stock_crud.pack(fill=tk.X, pady=5)
        self._btn(stock_crud, "Добавить акцию", self.add_stock_window, "success").pack(fill=tk.X, pady=2)
        self._btn(stock_crud, "Редактировать акцию", self.edit_stock_window, "warning").pack(fill=tk.X, pady=2)
        self._btn(stock_crud, "Удалить акцию", lambda: self.delete_record_window("stock_prices"), "danger").pack(fill=tk.X, pady=2)

        fin_crud = ttk.LabelFrame(self.left, text="CRUD: отчетность", padding=8)
        fin_crud.pack(fill=tk.X, pady=5)
        self._btn(fin_crud, "Добавить отчетность", self.add_finance_window, "success").pack(fill=tk.X, pady=2)
        self._btn(fin_crud, "Редактировать отчетность", self.edit_finance_window, "warning").pack(fill=tk.X, pady=2)
        self._btn(fin_crud, "Удалить отчетность", lambda: self.delete_record_window("finance_statement"), "danger").pack(fill=tk.X, pady=2)

        analysis_box = ttk.LabelFrame(self.left, text="Анализ и прогнозы", padding=8)
        analysis_box.pack(fill=tk.X, pady=5)
        self._btn(analysis_box, "Сводка по компаниям", self.analysis_summary, "dark").pack(fill=tk.X, pady=2)
        self._btn(analysis_box, "Технический анализ", self.technical_analysis_window, "dark").pack(fill=tk.X, pady=2)
        self._btn(analysis_box, "Корреляция", self.correlation_analysis, "dark").pack(fill=tk.X, pady=2)
        self._btn(analysis_box, "Аномальные дни", self.anomalies_analysis, "dark").pack(fill=tk.X, pady=2)
        self._btn(analysis_box, "Финансовые коэффициенты", self.financial_ratios_window, "dark").pack(fill=tk.X, pady=2)
        self._btn(analysis_box, "Прогноз акций", self.stock_forecast_window, "dark").pack(fill=tk.X, pady=2)
        self._btn(analysis_box, "Backtesting моделей", self.backtesting_window, "dark").pack(fill=tk.X, pady=2)
        self._btn(analysis_box, "Прогноз отчетности", self.finance_forecast_window, "dark").pack(fill=tk.X, pady=2)

        charts_box = ttk.LabelFrame(self.left, text="Графики", padding=8)
        charts_box.pack(fill=tk.X, pady=5)
        self._btn(charts_box, "Цена и индикаторы", self.price_chart_window, "info").pack(fill=tk.X, pady=2)
        self._btn(charts_box, "Сравнение доходности", self.returns_chart_window, "info").pack(fill=tk.X, pady=2)
        self._btn(charts_box, "Тепловая карта корреляции", self.correlation_heatmap, "info").pack(fill=tk.X, pady=2)

    def _btn(self, parent, text: str, command, bootstyle: Optional[str] = None):
        if HAS_TTKBOOTSTRAP and bootstyle:
            return ttk.Button(parent, text=text, command=command, bootstyle=bootstyle)
        return ttk.Button(parent, text=text, command=command)

    def _update_left_scrollregion(self, _event=None) -> None:
        self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))

    def _resize_left_panel(self, event) -> None:
        self.left_canvas.itemconfigure(self.left_window, width=event.width)

    def _bind_mousewheel(self) -> None:
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self) -> None:
        self.root.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event) -> None:
        self.left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def show(self, text: str) -> None:
        self.text.delete(1.0, tk.END)
        self.text.insert(tk.END, text)

    def _fail(self, title: str, err: Exception) -> None:
        logger.exception("%s: %s", title, err)
        messagebox.showerror(title, str(err))

    def ensure_db(self) -> None:
        try:
            init_db(self.config.db_path)
        except Exception as exc:
            self._fail("Ошибка БД", exc)

    def clear_database_window(self) -> None:
        ok = messagebox.askyesno(
            "Очистка базы данных",
            "Удалить все котировки и финансовую отчетность из базы? Это действие нельзя отменить.",
        )
        if not ok:
            return
        try:
            clear_database(self.config.db_path)
            self.show("База данных очищена.\n\nТаблицы stock_prices и finance_statement пустые. Можно заново выполнить импорт данных.")
            messagebox.showinfo("Готово", "База данных очищена")
        except Exception as exc:
            self._fail("Ошибка очистки базы", exc)

    def _query_df(self, sql: str, params=()) -> pd.DataFrame:
        return query_df(self.config.db_path, sql, params)

    def _execute(self, sql: str, params=()) -> int:
        return execute(self.config.db_path, sql, params)

    def import_stock_csv_window(self) -> None:
        path = filedialog.askopenfilename(title="CSV котировок", filetypes=[("CSV", "*.csv"), ("Все файлы", "*.*")])
        if not path:
            return
        try:
            report = import_stock_csv(self.config.db_path, path, self.config.csv_chunksize)
            self.show("Импорт котировок завершен.\n\n" + report.summary())
        except Exception as exc:
            self._fail("Ошибка импорта", exc)

    def import_finance_csv_window(self) -> None:
        path = filedialog.askopenfilename(title="CSV финансовой отчетности", filetypes=[("CSV", "*.csv"), ("Все файлы", "*.*")])
        if not path:
            return
        try:
            report = import_finance_csv(self.config.db_path, path)
            self.show("Импорт отчетности завершен.\n\n" + report.summary())
        except Exception as exc:
            self._fail("Ошибка импорта", exc)

    def import_base_data_folder_window(self) -> None:
        folder = filedialog.askdirectory(title="Выберите папку finance_bigdata_project")
        if not folder:
            return
        try:
            stock_report, finance_report = import_base_data_folder(self.config.db_path, folder, self.config.csv_chunksize)
            self.show(
                "Импорт папки base data завершен.\n\n"
                "Котировки:\n"
                f"{stock_report.summary()}\n\n"
                "Финансовая отчетность:\n"
                f"{finance_report.summary()}"
            )
        except Exception as exc:
            self._fail("Ошибка импорта папки", exc)

    def load_table(self, table: str) -> None:
        try:
            if table == "stock_prices":
                df = self._query_df("SELECT * FROM stock_prices ORDER BY company,date LIMIT 200")
            else:
                df = self._query_df("SELECT * FROM finance_statement ORDER BY company,statement_type,item,period LIMIT 300")
            self.show("Таблица пустая" if df.empty else df.to_string(index=False))
        except Exception as exc:
            self._fail("Ошибка", exc)

    def view_table_window(self, table: str) -> None:
        try:
            df = self._query_df(f"SELECT * FROM {table} ORDER BY id")
            win = tk.Toplevel(self.root)
            win.title(f"Просмотр: {table}")
            win.geometry("1250x720")
            frame = ttk.Frame(win)
            frame.pack(fill=tk.BOTH, expand=True)
            cols = list(df.columns) if not df.empty else self._table_columns(table)
            tree = ttk.Treeview(frame, columns=cols, show="headings")
            y = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            x = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=y.set, xscrollcommand=x.set)
            tree.grid(row=0, column=0, sticky="nsew")
            y.grid(row=0, column=1, sticky="ns")
            x.grid(row=1, column=0, sticky="ew")
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            reverse_map: Dict[str, bool] = {}

            def sort_column(col: str) -> None:
                reverse = reverse_map.get(col, False)
                data = [(tree.set(item, col), item) for item in tree.get_children("")]
                try:
                    data.sort(key=lambda z: float(z[0]) if z[0] not in ("", "None") else -np.inf, reverse=reverse)
                except Exception:
                    data.sort(key=lambda z: str(z[0]).lower(), reverse=reverse)
                for idx, (_, item_id) in enumerate(data):
                    tree.move(item_id, "", idx)
                reverse_map[col] = not reverse

            for col in cols:
                tree.heading(col, text=col, command=lambda c=col: sort_column(c))
                tree.column(col, width=130, anchor="center")
            for row in df.itertuples(index=False):
                tree.insert("", tk.END, values=row)
        except Exception as exc:
            self._fail("Ошибка", exc)

    def _table_columns(self, table: str) -> List[str]:
        return STOCK_COLUMNS if table == "stock_prices" else FINANCE_COLUMNS

    def add_stock_window(self) -> None:
        self._stock_form_window("Добавить акцию")

    def edit_stock_window(self) -> None:
        self._stock_form_window("Редактировать акцию", edit=True)

    def _stock_form_window(self, title: str, edit: bool = False) -> None:
        win = tk.Toplevel(self.root)
        win.title(title)
        fields = ["company", "ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
        entries = {}
        row_start = 0
        id_entry = None
        if edit:
            ttk.Label(win, text="ID").grid(row=0, column=0, sticky="w", padx=6, pady=3)
            id_entry = ttk.Entry(win, width=25)
            id_entry.grid(row=0, column=1, padx=6, pady=3)
            row_start = 1
        for index, field in enumerate(fields, row_start):
            ttk.Label(win, text=field).grid(row=index, column=0, sticky="w", padx=6, pady=3)
            entry = ttk.Entry(win, width=45)
            entry.grid(row=index, column=1, padx=6, pady=3)
            entries[field] = entry

        def load_one() -> None:
            try:
                rid = int(id_entry.get())  # type: ignore
                df = self._query_df("SELECT * FROM stock_prices WHERE id=?", (rid,))
                if df.empty:
                    raise ValueError("ID не найден")
                row = df.iloc[0]
                for field in fields:
                    entries[field].delete(0, tk.END)
                    entries[field].insert(0, "" if pd.isna(row[field]) else str(row[field]))
            except Exception as exc:
                self._fail("Ошибка", exc)

        def save() -> None:
            try:
                vals = {field: entries[field].get().strip() for field in fields}
                if not (vals["company"] and vals["ticker"] and vals["date"] and vals["close"]):
                    raise ValueError("Обязательные поля: company, ticker, date, close")
                vals["date"] = pd.to_datetime(vals["date"], errors="raise").strftime("%Y-%m-%d")
                for num in ["open", "high", "low", "close", "adj_close", "volume"]:
                    vals[num] = None if vals[num] == "" else float(vals[num])
                vals["company"] = str(vals["company"]).lower()
                vals["ticker"] = str(vals["ticker"]).upper()
                params = (
                    vals["company"], vals["ticker"], vals["date"], vals["open"], vals["high"],
                    vals["low"], vals["close"], vals["adj_close"], vals["volume"],
                )
                if edit:
                    rid = int(id_entry.get())  # type: ignore
                    count = self._execute(
                        """
                        UPDATE stock_prices
                        SET company=?, ticker=?, date=?, open=?, high=?, low=?, close=?, adj_close=?, volume=?
                        WHERE id=?
                        """,
                        params + (rid,),
                    )
                    if count == 0:
                        raise ValueError("ID не найден")
                else:
                    self._execute(
                        """
                        INSERT INTO stock_prices(company,ticker,date,open,high,low,close,adj_close,volume)
                        VALUES (?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(company,ticker,date) DO UPDATE SET
                            open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close,
                            adj_close=excluded.adj_close, volume=excluded.volume
                        """,
                        params,
                    )
                messagebox.showinfo("Успех", "Запись сохранена в stock_prices")
                win.destroy()
                self.load_table("stock_prices")
            except Exception as exc:
                self._fail("Ошибка", exc)

        btn_row = row_start + len(fields)
        if edit:
            ttk.Button(win, text="Загрузить по ID", command=load_one).grid(row=btn_row, column=0, padx=6, pady=10)
        ttk.Button(win, text="Сохранить", command=save).grid(row=btn_row, column=1, padx=6, pady=10)

    def add_finance_window(self) -> None:
        self._finance_form_window("Добавить отчетность")

    def edit_finance_window(self) -> None:
        self._finance_form_window("Редактировать отчетность", edit=True)

    def _finance_form_window(self, title: str, edit: bool = False) -> None:
        win = tk.Toplevel(self.root)
        win.title(title)
        fields = ["company", "ticker", "statement_type", "item", "period", "value"]
        entries = {}
        row_start = 0
        id_entry = None
        if edit:
            ttk.Label(win, text="ID").grid(row=0, column=0, sticky="w", padx=6, pady=3)
            id_entry = ttk.Entry(win, width=25)
            id_entry.grid(row=0, column=1, padx=6, pady=3)
            row_start = 1
        for index, field in enumerate(fields, row_start):
            ttk.Label(win, text=field).grid(row=index, column=0, sticky="w", padx=6, pady=3)
            entry = ttk.Entry(win, width=60)
            entry.grid(row=index, column=1, padx=6, pady=3)
            entries[field] = entry
        entries["statement_type"].insert(0, "financials")

        def load_one() -> None:
            try:
                rid = int(id_entry.get())  # type: ignore
                df = self._query_df("SELECT * FROM finance_statement WHERE id=?", (rid,))
                if df.empty:
                    raise ValueError("ID не найден")
                row = df.iloc[0]
                for field in fields:
                    entries[field].delete(0, tk.END)
                    entries[field].insert(0, "" if pd.isna(row[field]) else str(row[field]))
            except Exception as exc:
                self._fail("Ошибка", exc)

        def save() -> None:
            try:
                vals = {field: entries[field].get().strip() for field in fields}
                if not all(vals[field] for field in ["company", "ticker", "statement_type", "item", "period"]):
                    raise ValueError("Заполните все поля кроме value")
                vals["period"] = pd.to_datetime(vals["period"], errors="raise").strftime("%Y-%m-%d")
                vals["value"] = None if vals["value"] == "" else float(vals["value"])
                vals["company"] = vals["company"].lower()
                vals["ticker"] = vals["ticker"].upper()
                params = (
                    vals["company"], vals["ticker"], vals["statement_type"], vals["item"], vals["period"], vals["value"],
                )
                if edit:
                    rid = int(id_entry.get())  # type: ignore
                    count = self._execute(
                        """
                        UPDATE finance_statement
                        SET company=?,ticker=?,statement_type=?,item=?,period=?,value=?
                        WHERE id=?
                        """,
                        params + (rid,),
                    )
                    if count == 0:
                        raise ValueError("ID не найден")
                else:
                    self._execute(
                        """
                        INSERT INTO finance_statement(company,ticker,statement_type,item,period,value)
                        VALUES (?,?,?,?,?,?)
                        ON CONFLICT(company,ticker,statement_type,item,period) DO UPDATE SET value=excluded.value
                        """,
                        params,
                    )
                messagebox.showinfo("Успех", "Запись сохранена в finance_statement")
                win.destroy()
                self.load_table("finance_statement")
            except Exception as exc:
                self._fail("Ошибка", exc)

        btn_row = row_start + len(fields)
        if edit:
            ttk.Button(win, text="Загрузить по ID", command=load_one).grid(row=btn_row, column=0, padx=6, pady=10)
        ttk.Button(win, text="Сохранить", command=save).grid(row=btn_row, column=1, padx=6, pady=10)

    def delete_record_window(self, table: str) -> None:
        win = tk.Toplevel(self.root)
        win.title(f"Удалить из {table}")
        ttk.Label(win, text="ID записи").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        id_entry = ttk.Entry(win, width=25)
        id_entry.grid(row=0, column=1, padx=6, pady=6)

        def delete() -> None:
            try:
                rid = int(id_entry.get())
                count = self._execute(f"DELETE FROM {table} WHERE id=?", (rid,))
                if count == 0:
                    raise ValueError("ID не найден")
                messagebox.showinfo("Успех", f"Удалено из {table}")
                win.destroy()
                self.load_table(table)
            except Exception as exc:
                self._fail("Ошибка", exc)

        ttk.Button(win, text="Удалить", command=delete).grid(row=1, columnspan=2, pady=10)

    def dataset_overview(self) -> None:
        try:
            stock = self._query_df("SELECT * FROM stock_prices")
            finance = self._query_df("SELECT * FROM finance_statement")
            self.show(dataset_overview(stock, finance))
        except Exception as exc:
            self._fail("Ошибка обзора", exc)

    def analysis_summary(self) -> None:
        try:
            df = self._query_df("SELECT company,ticker,date,close,volume FROM stock_prices ORDER BY company,date")
            summary = company_summary(df)
            self.show("=== СВОДКА ПО КОМПАНИЯМ ===\n\n" + summary.to_string(index=False, formatters=self._summary_formatters()))
        except Exception as exc:
            self._fail("Ошибка анализа", exc)

    def technical_analysis_window(self) -> None:
        win = self._company_input_window("Технический анализ")

        def run(company: str) -> None:
            try:
                df = self._query_df("SELECT date,close,volume FROM stock_prices WHERE company=? ORDER BY date", (company,))
                indicators = technical_indicators(df)
                self.show(f"company: {company}\n\n" + technical_signal(indicators))
                win.destroy()
            except Exception as exc:
                self._fail("Ошибка", exc)

        self._attach_company_runner(win, run, "Анализ")

    def correlation_analysis(self) -> None:
        try:
            df = self._query_df("SELECT company,date,close FROM stock_prices ORDER BY date")
            corr = return_correlation(df)
            self.show("=== КОРРЕЛЯЦИЯ ДНЕВНОЙ ДОХОДНОСТИ ===\n\n" + corr.round(3).to_string())
        except Exception as exc:
            self._fail("Ошибка корреляции", exc)

    def anomalies_analysis(self) -> None:
        try:
            df = self._query_df("SELECT company,ticker,date,close FROM stock_prices ORDER BY company,date")
            res = anomalous_days(df)
            if res.empty:
                self.show("Аномальные дни не найдены.")
            else:
                self.show("=== АНОМАЛЬНЫЕ ДНИ ПО ДОХОДНОСТИ ===\n\n" + res.head(100).to_string(index=False))
        except Exception as exc:
            self._fail("Ошибка анализа аномалий", exc)

    def financial_ratios_window(self) -> None:
        win = self._company_input_window("Финансовые коэффициенты")

        def run(company: str) -> None:
            try:
                df = self._query_df("SELECT * FROM finance_statement WHERE company=?", (company,))
                ratios = financial_ratios(df)
                self.show(f"=== ФИНАНСОВЫЕ КОЭФФИЦИЕНТЫ ===\ncompany: {company}\n\n" + ratios.round(4).to_string())
                win.destroy()
            except Exception as exc:
                self._fail("Ошибка", exc)

        self._attach_company_runner(win, run, "Рассчитать")

    def stock_forecast_window(self) -> None:
        win, entries = self._forecast_input_window("Прогноз акций")

        def run() -> None:
            try:
                company = entries["company"].get().strip().lower()
                column = entries["metric"].get().strip()
                algorithm = entries["algorithm"].get().strip()
                days = int(entries["days"].get().strip())
                if days <= 0 or days > 365:
                    raise ValueError("days должен быть от 1 до 365")
                df = self._query_df(f"SELECT date,{column} FROM stock_prices WHERE company=? ORDER BY date", (company,))
                if len(df) < 80:
                    raise ValueError("Нужно минимум 80 строк данных")
                df["date"] = pd.to_datetime(df["date"])
                series = pd.to_numeric(df[column], errors="coerce").dropna().astype(float).reset_index(drop=True)
                forecast = forecast_series(series, days, algorithm)
                future_dates = pd.bdate_range(df["date"].max() + pd.Timedelta(days=1), periods=days)
                out = [
                    "=== ПРОГНОЗ АКЦИЙ ===",
                    f"company: {company}",
                    f"metric: {column}",
                    f"algorithm: {algorithm}",
                    f"history rows: {len(series)}",
                    "",
                    "Прогноз:",
                ]
                out.extend(f"{date.date()}: {float(value):,.2f}" for date, value in zip(future_dates, forecast))
                self.show("\n".join(out))
                if HAS_MATPLOTLIB:
                    history = pd.DataFrame({"date": df["date"], "value": pd.to_numeric(df[column], errors="coerce")}).dropna().tail(180)
                    pred_df = pd.DataFrame({"date": future_dates, "value": forecast})
                    self._show_line_chart(
                        f"Прогноз {company} ({algorithm})",
                        [(history["date"], history["value"], "История"), (pred_df["date"], pred_df["value"], "Прогноз")],
                        y_label=column,
                    )
                win.destroy()
            except Exception as exc:
                self._fail("Ошибка прогноза", exc)

        ttk.Button(win, text="Построить прогноз", command=run).grid(row=4, column=1, padx=6, pady=10)

    def backtesting_window(self) -> None:
        win, entries = self._forecast_input_window("Backtesting моделей")
        entries["days"].delete(0, tk.END)
        entries["days"].insert(0, "60")

        def run() -> None:
            try:
                company = entries["company"].get().strip().lower()
                column = entries["metric"].get().strip()
                test_size = int(entries["days"].get().strip())
                df = self._query_df(f"SELECT {column} FROM stock_prices WHERE company=? ORDER BY date", (company,))
                result = backtest_series(pd.to_numeric(df[column], errors="coerce"), ALGORITHMS, test_size)
                self.show(
                    "=== BACKTESTING МОДЕЛЕЙ ПРОГНОЗИРОВАНИЯ ===\n"
                    f"company: {company}\nmetric: {column}\ntest_size: {test_size}\n\n"
                    + result.to_string(index=False)
                )
                win.destroy()
            except Exception as exc:
                self._fail("Ошибка backtesting", exc)

        ttk.Button(win, text="Проверить модели", command=run).grid(row=4, column=1, padx=6, pady=10)

    def finance_forecast_window(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Прогноз отчетности")
        entries = {}

        ttk.Label(win, text="company").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        entries["company"] = ttk.Combobox(win, values=self._get_companies("finance_statement"), width=52, state="readonly")
        entries["company"].grid(row=0, column=1, padx=6, pady=3)

        ttk.Label(win, text="statement_type").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        entries["statement_type"] = ttk.Combobox(win, values=[], width=52, state="readonly")
        entries["statement_type"].grid(row=1, column=1, padx=6, pady=3)

        ttk.Label(win, text="item").grid(row=2, column=0, sticky="w", padx=6, pady=3)
        entries["item"] = ttk.Combobox(win, values=[], width=52, state="readonly")
        entries["item"].grid(row=2, column=1, padx=6, pady=3)

        ttk.Label(win, text="periods").grid(row=3, column=0, sticky="w", padx=6, pady=3)
        entries["periods"] = ttk.Entry(win, width=10)
        entries["periods"].grid(row=3, column=1, padx=6, pady=3, sticky="w")
        entries["periods"].insert(0, "4")

        def refresh_statement_types(_event=None) -> None:
            company = entries["company"].get().strip().lower()
            values = self._get_statement_types(company)
            entries["statement_type"].configure(values=values)
            entries["item"].configure(values=[])
            entries["item"].set("")
            if values:
                entries["statement_type"].set(values[0])
                refresh_items()

        def refresh_items(_event=None) -> None:
            company = entries["company"].get().strip().lower()
            statement_type = entries["statement_type"].get().strip()
            values = self._get_finance_items(company, statement_type)
            entries["item"].configure(values=values)
            if values:
                entries["item"].set(values[0])

        entries["company"].bind("<<ComboboxSelected>>", refresh_statement_types)
        entries["statement_type"].bind("<<ComboboxSelected>>", refresh_items)
        companies = list(entries["company"]["values"])
        if companies:
            entries["company"].set(companies[0])
            refresh_statement_types()

        def run() -> None:
            try:
                company = entries["company"].get().strip().lower()
                statement_type = entries["statement_type"].get().strip()
                item = entries["item"].get().strip()
                periods = int(entries["periods"].get().strip())
                if not company or not statement_type or not item:
                    raise ValueError("Выберите company, statement_type и item из списков")
                df = self._query_df(
                    """
                    SELECT period,value FROM finance_statement
                    WHERE company=? AND statement_type=? AND item=?
                    ORDER BY period
                    """,
                    (company, statement_type, item),
                )
                pred = finance_forecast(df, periods)
                self.show(
                    "=== ПРОГНОЗ ОТЧЕТНОСТИ ===\n"
                    f"company: {company}\nstatement_type: {statement_type}\nitem: {item}\n\n"
                    "История:\n"
                    + df.to_string(index=False)
                    + "\n\nПрогноз:\n"
                    + pred.to_string(index=False)
                )
                win.destroy()
            except Exception as exc:
                self._fail("Ошибка", exc)

        ttk.Button(win, text="Сделать прогноз", command=run).grid(row=4, column=1, padx=6, pady=10)

    def price_chart_window(self) -> None:
        win = self._company_input_window("График цены и индикаторов")

        def run(company: str) -> None:
            try:
                self._require_matplotlib()
                df = self._query_df("SELECT date,close,volume FROM stock_prices WHERE company=? ORDER BY date", (company,))
                indicators = technical_indicators(df)
                fig = Figure(figsize=(10, 7), dpi=100)
                ax1 = fig.add_subplot(311)
                ax2 = fig.add_subplot(312)
                ax3 = fig.add_subplot(313)
                ax1.plot(indicators["date"], indicators["close"], label="Close")
                ax1.plot(indicators["date"], indicators["SMA20"], label="SMA20")
                ax1.plot(indicators["date"], indicators["SMA50"], label="SMA50")
                ax1.set_title(f"{company}: цена и скользящие средние")
                ax1.legend()
                ax2.bar(indicators["date"], indicators["volume"], label="Volume")
                ax2.set_title("Объем торгов")
                ax3.plot(indicators["date"], indicators["RSI14"], label="RSI14")
                ax3.axhline(70, color="red", linestyle="--", linewidth=1)
                ax3.axhline(30, color="green", linestyle="--", linewidth=1)
                ax3.set_title("RSI")
                ax3.legend()
                fig.tight_layout()
                self._show_figure(fig, f"График: {company}")
                win.destroy()
            except Exception as exc:
                self._fail("Ошибка графика", exc)

        self._attach_company_runner(win, run, "Построить")

    def returns_chart_window(self) -> None:
        try:
            self._require_matplotlib()
            df = self._query_df("SELECT company,date,close FROM stock_prices ORDER BY date")
            pivot = df.pivot_table(index="date", columns="company", values="close")
            returns = pivot.pct_change().fillna(0)
            cumulative = (1 + returns).cumprod() - 1
            colors = [
                "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#003f5c", "#ffa600",
                "#58508d", "#bc5090", "#ff6361", "#006400", "#000000", "#4b0082",
            ]
            linestyles = ["-", "--", "-.", ":"]
            fig = Figure(figsize=(11, 7), dpi=100)
            ax = fig.add_subplot(111)
            dates = pd.to_datetime(cumulative.index)
            for index, company in enumerate(cumulative.columns):
                ax.plot(
                    dates,
                    cumulative[company] * 100,
                    label=company,
                    color=colors[index % len(colors)],
                    linestyle=linestyles[(index // len(colors)) % len(linestyles)],
                    linewidth=2.2,
                    alpha=0.95,
                )
            ax.axhline(0, color="#222222", linewidth=1, alpha=0.7)
            ax.set_title("Сравнение накопленной доходности")
            ax.set_ylabel("Доходность, %")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=True)
            fig.tight_layout(rect=(0, 0, 0.82, 1))
            self._show_figure(fig, "Сравнение накопленной доходности")
        except Exception as exc:
            self._fail("Ошибка графика", exc)

    def correlation_heatmap(self) -> None:
        try:
            self._require_matplotlib()
            df = self._query_df("SELECT company,date,close FROM stock_prices ORDER BY date")
            corr = return_correlation(df)
            fig = Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111)
            image = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.index)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticklabels(corr.index)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title("Корреляция дневной доходности")
            fig.tight_layout()
            self._show_figure(fig, "Тепловая карта корреляции")
        except Exception as exc:
            self._fail("Ошибка графика", exc)

    def _company_input_window(self, title: str) -> tk.Toplevel:
        win = tk.Toplevel(self.root)
        win.title(title)
        ttk.Label(win, text="company").grid(row=0, column=0, padx=6, pady=3)
        entry = ttk.Combobox(win, values=self._get_companies(), width=30, state="readonly")
        entry.grid(row=0, column=1, padx=6, pady=3)
        values = list(entry["values"])
        if values:
            entry.set(values[0])
        win.company_entry = entry  # type: ignore[attr-defined]
        return win

    def _attach_company_runner(self, win: tk.Toplevel, callback, button_text: str) -> None:
        def run() -> None:
            company = win.company_entry.get().strip().lower()  # type: ignore[attr-defined]
            if not company:
                messagebox.showwarning("Проверка ввода", "Введите company, например apple")
                return
            callback(company)

        ttk.Button(win, text=button_text, command=run).grid(row=1, column=1, padx=6, pady=10)

    def _forecast_input_window(self, title: str):
        win = tk.Toplevel(self.root)
        win.title(title)
        entries = {}
        ttk.Label(win, text="company").grid(row=0, column=0, padx=6, pady=3)
        entries["company"] = ttk.Combobox(win, values=self._get_companies("stock_prices"), width=30, state="readonly")
        entries["company"].grid(row=0, column=1, padx=6, pady=3)
        company_values = list(entries["company"]["values"])
        if company_values:
            entries["company"].set(company_values[0])
        ttk.Label(win, text="metric").grid(row=1, column=0, padx=6, pady=3)
        entries["metric"] = ttk.Combobox(win, values=["close", "open", "high", "low", "adj_close", "volume"], width=27, state="readonly")
        entries["metric"].grid(row=1, column=1, padx=6, pady=3)
        entries["metric"].set("close")
        ttk.Label(win, text="algorithm").grid(row=2, column=0, padx=6, pady=3)
        entries["algorithm"] = ttk.Combobox(win, values=ALGORITHMS, width=27, state="readonly")
        entries["algorithm"].grid(row=2, column=1, padx=6, pady=3)
        entries["algorithm"].set("linear")
        ttk.Label(win, text="days/test size").grid(row=3, column=0, padx=6, pady=3)
        entries["days"] = ttk.Entry(win, width=10)
        entries["days"].grid(row=3, column=1, padx=6, pady=3, sticky="w")
        entries["days"].insert(0, "30")
        return win, entries

    def _get_companies(self, table: Optional[str] = None) -> List[str]:
        try:
            if table == "stock_prices":
                df = self._query_df("SELECT DISTINCT company FROM stock_prices ORDER BY company")
            elif table == "finance_statement":
                df = self._query_df("SELECT DISTINCT company FROM finance_statement ORDER BY company")
            else:
                df = self._query_df(
                    """
                    SELECT company FROM stock_prices
                    UNION
                    SELECT company FROM finance_statement
                    ORDER BY company
                    """
                )
            return [str(value) for value in df["company"].dropna().tolist()]
        except Exception:
            logger.exception("Не удалось загрузить список компаний")
            return []

    def _get_statement_types(self, company: str) -> List[str]:
        if not company:
            return []
        try:
            df = self._query_df(
                "SELECT DISTINCT statement_type FROM finance_statement WHERE company=? ORDER BY statement_type",
                (company,),
            )
            return [str(value) for value in df["statement_type"].dropna().tolist()]
        except Exception:
            logger.exception("Не удалось загрузить типы отчетности")
            return []

    def _get_finance_items(self, company: str, statement_type: str) -> List[str]:
        if not company or not statement_type:
            return []
        try:
            df = self._query_df(
                """
                SELECT DISTINCT item FROM finance_statement
                WHERE company=? AND statement_type=?
                ORDER BY item
                """,
                (company, statement_type),
            )
            return [str(value) for value in df["item"].dropna().tolist()]
        except Exception:
            logger.exception("Не удалось загрузить строки отчетности")
            return []

    def _require_matplotlib(self) -> None:
        if not HAS_MATPLOTLIB:
            raise ValueError("Для графиков установите matplotlib: pip install matplotlib")

    def _show_line_chart(self, title: str, series_list, y_label: str) -> None:
        self._require_matplotlib()
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        for x, y, label in series_list:
            ax.plot(x, y, label=label)
        ax.set_title(title)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        self._show_figure(fig, title)

    def _show_figure(self, fig, title: str) -> None:
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("1100x750")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _summary_formatters(self):
        return {
            "last_close": "{:,.2f}".format,
            "total_return_%": "{:,.2f}".format,
            "annual_return_%": "{:,.2f}".format,
            "annual_volatility_%": "{:,.2f}".format,
            "sharpe_ratio": "{:,.3f}".format,
            "max_drawdown_%": "{:,.2f}".format,
            "avg_volume": "{:,.0f}".format,
        }

