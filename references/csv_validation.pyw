from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from PIL import Image, ImageTk

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
BASELINE_FILE = HERE / "baseline_original.csv"
STATE_FILE = HERE / "review_state.json"
FORMS = {"bulk", "solid"}
MATERIALS = {"metal", "wood", "plastic", "glass", "paper", "mineral"}
FIELDS = ("x1", "y1", "x2", "y2", "score", "form", "material", "volume", "mass", "take")
COLORS = {"base": "#35c3e6", "candidate": "#ffa43a", "added": "#44cc76",
          "removed": "#f16c6c", "correct": "#44cc76", "incorrect": "#f16c6c",
          "unreviewed": "#35c3e6"}


def parse_prediction(text: str, image_id: str) -> list[dict[str, Any]]:
    text = (text or "").strip()
    if not text or text.lower() == "none":
        return []
    result = []
    for number, item in enumerate(text.split(";"), 1):
        t = item.split()
        if len(t) != 10:
            raise ValueError(f"{image_id}: object {number} has {len(t)} fields, expected 10")
        try:
            nums = [float(v) for v in (*t[:5], t[7], t[8], t[9])]
        except ValueError as exc:
            raise ValueError(f"{image_id}: object {number} has an invalid number") from exc
        x1, y1, x2, y2, score, volume, mass, take = nums
        form, material = t[5].lower(), t[6].lower()
        if not all(math.isfinite(v) for v in nums):
            raise ValueError(f"{image_id}: object {number} contains NaN or infinity")
        if x2 <= x1 or y2 <= y1 or not 0 <= score <= 1:
            raise ValueError(f"{image_id}: object {number} has invalid box/score")
        if form not in FORMS or material not in MATERIALS or volume <= 0 or mass < 0 or take not in (0, 1):
            raise ValueError(f"{image_id}: object {number} has an invalid attribute")
        result.append(dict(zip(FIELDS, (x1, y1, x2, y2, score, form, material, volume, mass, int(take))),
                           status="unreviewed"))
    if len(result) > 30:
        raise ValueError(f"{image_id}: more than 30 objects")
    return result


def read_csv(path: Path) -> dict[str, list[dict[str, Any]]]:
    data = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["image_id", "prediction"]:
            raise ValueError("CSV must have columns image_id,prediction")
        for row in reader:
            image_id = (row.get("image_id") or "").strip()
            if not image_id or image_id in data:
                raise ValueError(f"Empty or duplicate image_id: {image_id!r}")
            data[image_id] = parse_prediction(row.get("prediction") or "", image_id)
    return data


def write_csv(path: Path, frame_ids: list[str], data: dict[str, list[dict[str, Any]]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("image_id", "prediction"))
        for image_id in frame_ids:
            objects = data.get(image_id, [])
            prediction = "none" if not objects else ";".join(
                " ".join(str(obj[k]) for k in FIELDS) for obj in objects
            )
            writer.writerow((image_id, prediction))


def material_reference(base, candidate):
    """Find a reference for an extra candidate that overlaps a baseline box."""
    reference = max(base, key=lambda obj: box_iou(obj, candidate), default=None)
    if reference is not None and box_iou(reference, candidate) >= 0.5:
        return reference
    return None


def box_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    w = max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    h = max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = w * h
    aa = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    ab = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    return inter / max(aa + ab - inter, 1e-9)


def match(base: list[dict[str, Any]], cand: list[dict[str, Any]]):
    pairs = sorted((box_iou(a, b), i, j) for i, a in enumerate(base)
                   for j, b in enumerate(cand) if box_iou(a, b) >= 0.5)[::-1]
    used_a, used_b, found = set(), set(), []
    for overlap, i, j in pairs:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            found.append((i, j, overlap))
    return found, [i for i in range(len(base)) if i not in used_a], [j for j in range(len(cand)) if j not in used_b]


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Astramis CSV validation")
        root.geometry("1450x900")
        root.minsize(1050, 680)
        self.baseline_path = BASELINE_FILE
        self.baseline = {}
        self.review = {}
        self.drafts = {}
        self._loading_fields = False
        self._save_job = None
        self.candidates = {}
        self.frames = []
        self.required = set()
        self.current = ""
        self.image_root = PROJECT / "dataset" / "real" / "test"
        self.mode = "review"
        self.selected = -1
        self.photo = None
        self._photo_cache_key = None
        self.scale = 1
        self.offset = (0, 0)
        self.image_size = (1, 1)
        self.drag = None
        self.force_new_box = False
        self.vars = {key: tk.StringVar() for key in FIELDS}
        for variable in self.vars.values():
            variable.trace_add("write", self._field_changed)
        self.status = tk.StringVar()
        self._build()
        self._load_default()
        root.protocol("WM_DELETE_WINDOW", self.close)

    def _build(self):
        bar = ttk.Frame(self.root, padding=7)
        bar.pack(fill="x")
        for title, action in (
            ("Открыть baseline CSV", self.open_baseline),
            ("Папка RGB", self.open_images),
            ("Загрузить CSV-кандидаты", self.open_candidates),
            ("Ручная проверка", lambda: self.set_mode("review")),
            ("Сравнение", lambda: self.set_mode("compare")),
            ("Экспорт исправленного CSV", self.export_csv),
        ):
            ttk.Button(bar, text=title, command=action).pack(side="left", padx=2)
        ttk.Label(bar, textvariable=self.status).pack(side="right", padx=6)
        panes = ttk.Panedwindow(self.root, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=8, pady=8)
        left, right = ttk.Frame(panes), ttk.Frame(panes, width=445)
        panes.add(left, weight=3)
        panes.add(right, weight=2)
        self.canvas = tk.Canvas(left, bg="#111722", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self.draw())
        self.canvas.bind("<ButtonPress-1>", self.canvas_press)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.canvas_release)
        ttk.Label(left, text="Рамка: перетащите пустое место, чтобы добавить объект; тяните рамку для перемещения, угол — для изменения размера.").pack(fill="x", padx=6, pady=3)
        nav = ttk.Frame(left, padding=6)
        nav.pack(fill="x")
        ttk.Button(nav, text="←", command=lambda: self.step(-1)).pack(side="left")
        self.frame_combo = ttk.Combobox(nav, state="readonly", width=18)
        self.frame_combo.pack(side="left", padx=5)
        self.frame_combo.bind("<<ComboboxSelected>>", self.choose_frame)
        ttk.Button(nav, text="→", command=lambda: self.step(1)).pack(side="left")
        self.search = tk.StringVar()
        ttk.Entry(nav, textvariable=self.search, width=18).pack(side="right")
        self.search.trace_add("write", lambda *_: self.refresh_frames())

        self.review_panel = ttk.Frame(right)
        ttk.Label(self.review_panel, text="Объекты baseline", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        columns = ("class", "volume", "mass", "take", "status")
        self.tree = ttk.Treeview(self.review_panel, columns=columns, show="headings", height=6)
        self.tree.heading("class", text="Форма / материал")
        self.tree.heading("volume", text="Объём мл")
        self.tree.heading("mass", text="Масса г")
        self.tree.heading("take", text="Забрать")
        self.tree.heading("status", text="Статус")
        self.tree.column("class", width=132, minwidth=105)
        self.tree.column("volume", width=67, minwidth=55, anchor="e")
        self.tree.column("mass", width=62, minwidth=50, anchor="e")
        self.tree.column("take", width=52, minwidth=45, anchor="center")
        self.tree.column("status", width=80, minwidth=65)
        self.tree.tag_configure("correct", background="#d9f2df")
        self.tree.tag_configure("incorrect", background="#f7dddd")
        self.tree.tag_configure("unreviewed", background="#ffffff")
        self.tree.pack(fill="x", pady=6)
        self.tree.bind("<<TreeviewSelect>>", self.select_object)
        fields = ttk.LabelFrame(self.review_panel, text="Поля объекта", padding=7)
        fields.pack(fill="x")
        self.widgets = {}
        options = {"form": sorted(FORMS), "material": sorted(MATERIALS), "take": ("0", "1")}
        field_labels = {
            "x1": "Лево, px", "y1": "Верх, px", "x2": "Право, px", "y2": "Низ, px",
            "score": "Уверенность", "form": "Форма", "material": "Материал",
            "volume": "Объём, мл", "mass": "Масса, г", "take": "Забирать (0/1)",
        }
        for index, key in enumerate(FIELDS):
            row, col = divmod(index, 2)
            ttk.Label(fields, text=field_labels[key]).grid(row=row, column=col*2, sticky="w", padx=3, pady=3)
            if key in options:
                widget = ttk.Combobox(fields, textvariable=self.vars[key], values=options[key], state="readonly", width=11)
            else:
                widget = ttk.Entry(fields, textvariable=self.vars[key], width=13)
            widget.grid(row=row, column=col*2+1, sticky="ew", padx=3, pady=3)
            self.widgets[key] = widget
        ttk.Button(fields, text="Применить поля", command=self.apply_fields).grid(row=5, column=0, columnspan=4, sticky="ew", pady=5)
        actions = ttk.Frame(self.review_panel)
        actions.pack(fill="x", pady=6)
        for column, (title, status) in enumerate((("✓ Корректно", "correct"), ("✗ Ошибка", "incorrect"), ("Сброс", "unreviewed"))):
            ttk.Button(actions, text=title, command=lambda s=status: self.mark(s)).grid(row=0, column=column, sticky="w", padx=2)
        ttk.Button(actions, text="Добавить рамку", command=self.start_add_box).grid(row=1, column=0, sticky="w", padx=2)
        ttk.Button(actions, text="Удалить", command=self.delete_object).grid(row=1, column=1, sticky="w", padx=2)

        self.compare_panel = ttk.Frame(right)
        ttk.Label(self.compare_panel, text="Отличия от baseline", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.candidate_choice = tk.StringVar()
        self.candidate_combo = ttk.Combobox(self.compare_panel, state="readonly", textvariable=self.candidate_choice)
        self.candidate_combo.pack(fill="x", pady=6)
        self.candidate_combo.bind("<<ComboboxSelected>>", lambda _e: self.draw())
        self.diff = tk.Text(self.compare_panel, height=5, wrap="word", bg="#111722", fg="#e8edf5")
        self.diff.pack(fill="both", expand=True)
        self.diff.tag_configure("error", foreground="#ff8585")
        self.set_mode("review")

    def _load_default(self):
        if not BASELINE_FILE.exists():
            self.status.set("Выберите baseline CSV")
            return
        self.baseline = read_csv(BASELINE_FILE)
        self.baseline_path = BASELINE_FILE
        self.frames = sorted(self.baseline)
        self.required = set(self.frames)
        self.review = {k: [dict(o) for o in v] for k, v in self.baseline.items()}
        self.restore_state()
        self.current = self.frames[0] if self.frames else ""
        self.refresh_frames()
        self.draw()

    def restore_state(self):
        if not STATE_FILE.exists():
            self.status.set(f"Baseline: {self.baseline_path.name}")
            return
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            if state.get("baseline_source") != str(self.baseline_path.resolve()):
                return
            saved = state.get("frames", {})
            if set(saved) == set(self.frames):
                self.review = saved
                self.drafts = state.get("drafts", {})
                self.status.set("Восстановлены сохранённые правки и статусы")
        except Exception as exc:
            self.status.set(f"Не удалось восстановить review_state.json: {exc}")

    def save_state(self):
        if not self.review:
            return
        payload = {"baseline_source": str(self.baseline_path.resolve()),
                   "saved_at": datetime.now(timezone.utc).isoformat(),
                   "frames": self.review, "drafts": self.drafts}
        fd, temp = tempfile.mkstemp(dir=HERE, suffix=".tmp", prefix="review_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
            os.replace(temp, STATE_FILE)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def open_baseline(self):
        name = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not name:
            return
        try:
            data = read_csv(Path(name))
            if self.required and set(data) != self.required:
                raise ValueError("Набор image_id baseline должен совпадать с исходным")
            self.baseline_path = Path(name)
            self.baseline = data
            self.frames = sorted(data)
            self.required = set(self.frames)
            self.review = {k: [dict(o) for o in v] for k, v in data.items()}
            self.restore_state()
            self.current = self.frames[0] if self.frames else ""
            self.refresh_frames()
            self.draw()
        except Exception as exc:
            messagebox.showerror("Ошибка baseline", str(exc))

    def open_images(self):
        name = filedialog.askdirectory(title="Выберите dataset/real/test")
        if name:
            self.image_root = Path(name)
            self.draw()

    def open_candidates(self):
        names = filedialog.askopenfilenames(filetypes=[("CSV", "*.csv")])
        if not names:
            return
        staged = {}
        try:
            for name in names:
                data = read_csv(Path(name))
                if set(data) != self.required:
                    raise ValueError(f"{Path(name).name}: кадры не совпадают с baseline")
                staged[Path(name).name] = data
            self.candidates.update(staged)
            labels = sorted(self.candidates)
            self.candidate_combo.configure(values=labels)
            self.candidate_choice.set(Path(names[-1]).name)
            self.set_mode("compare")
            self.draw()
        except Exception as exc:
            messagebox.showerror("Ошибка CSV кандидата", str(exc))

    def set_mode(self, mode):
        self.mode = mode
        self.review_panel.pack_forget()
        self.compare_panel.pack_forget()
        if mode == "review":
            self.tree.configure(height=6)
            self.review_panel.pack(fill="both", expand=True)
        else:
            self.tree.configure(height=3)
            self.review_panel.pack(fill="x")
            self.compare_panel.pack(fill="both", expand=True)
        self.draw()

    def refresh_frames(self):
        query = self.search.get().lower().strip()
        visible = [x for x in self.frames if query in x.lower()]
        self.frame_combo.configure(values=visible)
        if visible and self.current not in visible:
            self.current = visible[0]
        self.frame_combo.set(self.current if self.current in visible else "")
        self.draw()

    def choose_frame(self, _event=None):
        self.current = self.frame_combo.get()
        self.selected = -1
        self.draw()

    def step(self, amount):
        visible = list(self.frame_combo.cget("values"))
        if self.current in visible and visible:
            i = max(0, min(len(visible)-1, visible.index(self.current)+amount))
            self.current = visible[i]
            self.frame_combo.set(self.current)
            self.selected = -1
            self.draw()

    def objects(self):
        return self.review.get(self.current, [])

    def selected_object(self):
        objects = self.objects()
        return objects[self.selected] if 0 <= self.selected < len(objects) else None

    def select_object(self, _event=None):
        selection = self.tree.selection()
        if selection and int(selection[0]) != self.selected:
            self.selected = int(selection[0])
            self.fill_fields()
            self.draw(refresh_panel=False)

    def fill_fields(self):
        obj = self.selected_object()
        self._loading_fields = True
        try:
            draft = self.drafts.get(self.current, {}).get(str(self.selected))
            for key in FIELDS:
                self.vars[key].set("" if obj is None else str(draft.get(key, obj[key]) if draft else obj[key]))
        finally:
            self._loading_fields = False

    def _field_changed(self, *_args):
        if self._loading_fields or not self.current or self.selected_object() is None:
            return
        values = {key: value.get() for key, value in self.vars.items()}
        self.drafts.setdefault(self.current, {})[str(self.selected)] = values
        try:
            candidate = parse_prediction(" ".join(values[key] for key in FIELDS), self.current)[0]
        except (ValueError, IndexError):
            candidate = None
        if candidate is not None:
            candidate["status"] = self.selected_object().get("status", "unreviewed")
            self.review[self.current][self.selected] = candidate
        if self._save_job is not None:
            self.root.after_cancel(self._save_job)
        self._save_job = self.root.after(500, self._save_draft_state)

    def _save_draft_state(self):
        self._save_job = None
        self.save_state()
        if self.mode == "compare":
            self.draw(refresh_panel=False)
            self.show_differences()

    def apply_fields(self):
        obj = self.selected_object()
        if obj is None:
            return
        values = []
        for key in FIELDS:
            value = self.vars[key].get().strip()
            values.append(value if key in ("form", "material") else value)
        try:
            new_obj = parse_prediction(" ".join(values), self.current)[0]
            new_obj["status"] = obj.get("status", "unreviewed")
            self.review[self.current][self.selected] = new_obj
            self.drafts.get(self.current, {}).pop(str(self.selected), None)
            self.save_state()
            self.draw()
        except Exception as exc:
            messagebox.showerror("Некорректные поля", str(exc))

    def mark(self, status):
        obj = self.selected_object()
        if obj is None:
            self.status.set("Сначала выберите объект: нажмите на его рамку или строку в списке")
            self.root.bell()
            messagebox.showwarning("Объект не выбран",
                                   "Сначала щёлкните по рамке на изображении или по строке объекта в списке.")
            return
        obj["status"] = status
        self.save_state()
        self.draw(refresh_panel=False)
        self._refresh_right()
        label = {"correct": "корректно", "incorrect": "ошибка", "unreviewed": "без отметки"}[status]
        self.status.set(f"Объект {self.selected+1}: отмечен как «{label}» и сохранён")

    def add_object(self):
        self.start_add_box()

    def delete_object(self):
        if self.selected_object():
            del self.review[self.current][self.selected]
            self.selected = -1
            self.save_state()
            self.draw()

    def export_csv(self):
        path = filedialog.asksaveasfilename(initialfile="baseline_reviewed.csv",
            defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            if self.selected >= 0:
                self.apply_fields()
            write_csv(Path(path), self.frames, self.review)
            self.save_state()
            messagebox.showinfo("Сохранено", f"Baseline сохранён:\n{path}")
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def image_path(self):
        for name in ("color.png", "color.jpg", "rgb.png", "rgb.jpg"):
            path = self.image_root / self.current / name
            if path.is_file():
                return path
        return None

    def _canvas_to_image(self, event):
        if self.scale <= 0:
            return None
        x = (event.x-self.offset[0])/self.scale
        y = (event.y-self.offset[1])/self.scale
        width, height = self.image_size
        if x < 0 or y < 0 or x >= width or y >= height:
            return None
        return x, y

    def start_add_box(self):
        self.force_new_box = True
        self.status.set("Перетащите мышью рамку вокруг нового объекта")

    def canvas_press(self, event):
        if not self.image_path():
            return
        point = self._canvas_to_image(event)
        if point is None:
            return
        x, y = point
        hits = [(max(0, (o["x2"]-o["x1"]) * (o["y2"]-o["y1"])), i)
                for i, o in enumerate(self.objects())
                if o["x1"] <= x <= o["x2"] and o["y1"] <= y <= o["y2"]]
        if hits and not self.force_new_box:
            self.selected = min(hits)[1]
            obj = self.selected_object()
            tolerance = max(5.0, 10.0/self.scale)
            corners = {
                "nw": (obj["x1"], obj["y1"], obj["x2"], obj["y2"]),
                "ne": (obj["x2"], obj["y1"], obj["x1"], obj["y2"]),
                "sw": (obj["x1"], obj["y2"], obj["x2"], obj["y1"]),
                "se": (obj["x2"], obj["y2"], obj["x1"], obj["y1"]),
            }
            nearest = min(corners, key=lambda k: math.hypot(x-corners[k][0], y-corners[k][1]))
            if math.hypot(x-corners[nearest][0], y-corners[nearest][1]) <= tolerance:
                self.drag = {"kind": "resize", "index": self.selected, "handle": nearest,
                             "anchor": corners[nearest][2:], "orig": dict(obj),
                             "start": (x, y), "moved": False}
            else:
                self.drag = {"kind": "move", "index": self.selected,
                             "orig": dict(obj), "start": (x, y), "moved": False}
            self.draw()
            return
        self.drag = {"kind": "new", "start": (x, y), "box": (x, y, x, y), "moved": False}
        self.selected = -1
        self.draw()

    def canvas_drag(self, event):
        if not self.drag:
            return
        point = self._canvas_to_image(event)
        if point is None:
            width, height = self.image_size
            point = (min(max(0, (event.x-self.offset[0])/self.scale), width),
                     min(max(0, (event.y-self.offset[1])/self.scale), height))
        x, y = point
        drag = self.drag
        if math.hypot((x-drag["start"][0])*self.scale, (y-drag["start"][1])*self.scale) >= 3:
            drag["moved"] = True
        if not drag["moved"]:
            return
        width, height = self.image_size
        if drag["kind"] == "new":
            x0, y0 = drag["start"]
            drag["box"] = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
        elif drag["kind"] == "move":
            orig, start = drag["orig"], drag["start"]
            dx, dy = x-start[0], y-start[1]
            dx = min(max(dx, -orig["x1"]), width-orig["x2"])
            dy = min(max(dy, -orig["y1"]), height-orig["y2"])
            obj = self.objects()[drag["index"]]
            obj.update(x1=orig["x1"]+dx, x2=orig["x2"]+dx,
                       y1=orig["y1"]+dy, y2=orig["y2"]+dy)
        else:
            ax, ay = drag["anchor"]
            obj = self.objects()[drag["index"]]
            px, py = min(max(x, 0), width), min(max(y, 0), height)
            obj["x1"], obj["x2"] = sorted((px, ax))
            obj["y1"], obj["y2"] = sorted((py, ay))
            if obj["x2"]-obj["x1"] < 2 or obj["y2"]-obj["y1"] < 2:
                obj.update({k: drag["orig"][k] for k in ("x1", "y1", "x2", "y2")})
        self.draw(refresh_panel=False)

    def canvas_release(self, _event=None):
        drag, self.drag = self.drag, None
        if not drag:
            return
        if not drag["moved"]:
            self.force_new_box = False
            return
        if drag["kind"] == "new" and drag["moved"]:
            x1, y1, x2, y2 = drag["box"]
            if x2-x1 >= 3 and y2-y1 >= 3:
                self.review[self.current].append(dict(
                    x1=x1, y1=y1, x2=x2, y2=y2, score=1.0,
                    form="solid", material="plastic", volume=1.0, mass=0.0,
                    take=0, status="unreviewed"))
                self.selected = len(self.review[self.current])-1
        if drag["moved"]:
            self.save_state()
        self.force_new_box = False
        self.draw()

    def draw_box(self, obj, color, label, selected=False):
        ox, oy = self.offset
        x1, y1, x2, y2 = [v*self.scale for v in (obj["x1"], obj["y1"], obj["x2"], obj["y2"])]
        x1, y1, x2, y2 = x1+ox, y1+oy, x2+ox, y2+oy
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=3, tags="overlay")
        tag_y = max(oy, y1-21)
        text_id = self.canvas.create_text(x1+4, tag_y+2, anchor="nw", text=label,
                                          fill="#101010", font=("Segoe UI", 9, "bold"), tags="overlay")
        box = self.canvas.bbox(text_id)
        if box:
            bg = self.canvas.create_rectangle(box[0]-3, box[1]-1, box[2]+3, box[3]+1,
                                               fill=color, outline=color, tags="overlay")
            self.canvas.tag_lower(bg, text_id)
        if selected:
            for x, y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
                self.canvas.create_rectangle(x-5, y-5, x+5, y+5,
                                             fill="#ffffff", outline="#111111", tags=("overlay", "handle"))

    def draw(self, refresh_panel=True):
        if not hasattr(self, "canvas"):
            return
        self.canvas.delete("overlay")
        cw,ch=max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height())
        path=self.image_path()
        if not path:
            self.canvas.delete("rgb")
            self._photo_cache_key = None
            self.scale=1
            self.offset=(10,10)
            self.canvas.create_text(20,25,anchor="nw",fill="white",
                text=f"{self.current}: RGB не найден. Выберите папку dataset/real/test.", tags="overlay")
            if refresh_panel:
                self._refresh_right()
            return
        try:
            stat = path.stat()
            cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
            cache_key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size, cw, ch)
            if cache_key != self._photo_cache_key:
                with Image.open(path) as source:
                    image = source.convert("RGB")
                self.image_size = image.size
                self.scale=min(max(0.01,(cw-20)/image.width),
                               max(0.01,(ch-20)/image.height),1.0)
                size = (max(1,int(image.width*self.scale)), max(1,int(image.height*self.scale)))
                self.offset=((cw-size[0])//2,(ch-size[1])//2)
                self.photo=ImageTk.PhotoImage(image.resize(size))
                self.canvas.delete("rgb")
                self.canvas.create_image(*self.offset, anchor="nw", image=self.photo, tags="rgb")
                self._photo_cache_key = cache_key
        except Exception as exc:
            self.canvas.delete("rgb")
            self._photo_cache_key = None
            self.status.set(f"Ошибка чтения RGB: {exc}")
            return
        if self.mode=="review":
            for i,obj in enumerate(self.objects()):
                color=COLORS.get(obj.get("status","unreviewed"),COLORS["unreviewed"])
                self.draw_box(obj,color,f"{i+1} {obj['form']}/{obj['material']}",i==self.selected)
        else:
            base=self.review.get(self.current,[])
            cand=self.candidates.get(self.candidate_choice.get(),{}).get(self.current,[])
            pairs,missing,added=match(base,cand)
            candidate_boxes = []
            for i,j,_ in pairs:
                self.draw_box(base[i],COLORS["base"],"B"+str(i+1),i==self.selected)
                candidate_boxes.append((base[i]["material"] != cand[j]["material"], j, f"C{j+1}"))
            for i in missing:
                self.draw_box(base[i],COLORS["base"],"-B"+str(i+1),i==self.selected)
            for j in added:
                reference = material_reference(base, cand[j])
                wrong = reference is not None and reference["material"] != cand[j]["material"]
                candidate_boxes.append((wrong, j, f"+C{j+1}"))
            for wrong, j, label in sorted(candidate_boxes):
                self.draw_box(cand[j], COLORS["incorrect" if wrong else "candidate"], label)
        if self.drag and self.drag["kind"] == "new" and self.drag["moved"]:
            x1, y1, x2, y2 = self.drag["box"]
            ox, oy = self.offset
            self.canvas.create_rectangle(x1*self.scale+ox, y1*self.scale+oy,
                x2*self.scale+ox, y2*self.scale+oy, outline=COLORS["added"], width=3, tags="overlay")
        self.canvas.tag_raise("handle")
        if refresh_panel:
            self._refresh_right()

    def _refresh_right(self):
        for i,obj in enumerate(self.objects()):
            iid = str(i)
            values=(obj["form"]+" / "+obj["material"], f"{obj['volume']:g}",
                    f"{obj['mass']:g}", obj["take"], obj.get("status","unreviewed"))
            if self.tree.exists(iid):
                self.tree.item(iid, values=values, tags=(obj.get("status","unreviewed"),))
            else:
                self.tree.insert("","end",iid=iid,values=values,
                                 tags=(obj.get("status","unreviewed"),))
        for iid in self.tree.get_children():
            if int(iid) >= len(self.objects()):
                self.tree.delete(iid)
        if 0 <= self.selected < len(self.objects()):
            if self.tree.selection() != (str(self.selected),):
                self.tree.selection_set(str(self.selected))
            self.tree.see(str(self.selected))
        elif self.tree.selection():
            self.tree.selection_remove(self.tree.selection())
        self.fill_fields()
        if self.mode=="review":
            done=sum(o.get("status")!="unreviewed" for o in self.objects())
            self.status.set(f"{self.current} | объектов: {len(self.objects())} | отмечено на кадре: {done}")
        else:
            self.show_differences()

    def show_differences(self):
        self.diff.configure(state="normal")
        self.diff.delete("1.0","end")
        base=self.review.get(self.current,[])
        cand=self.candidates.get(self.candidate_choice.get(),{}).get(self.current,[])
        pairs,missing,added=match(base,cand)
        self.diff.insert("end",f"Baseline {len(base)} | candidate {len(cand)}\nСопоставлено {len(pairs)} | пропало {len(missing)} | добавлено {len(added)}\n\n")
        for i,j,overlap in pairs:
            changes=[]
            for key in FIELDS:
                a,b=base[i][key],cand[j][key]
                tol=2 if key in ("x1","y1","x2","y2") else .01 if key=="score" else 1 if key=="volume" else 100 if key=="mass" else 0
                changed=abs(a-b)>tol if isinstance(a,(int,float)) and isinstance(b,(int,float)) else a!=b
                if changed:
                    if key == "mass":
                        changes.append((f"Масса: {a:g} → {b:g} г (разница {abs(a-b):g} г)", ()))
                    elif key == "take":
                        actions = {0: "объехать", 1: "забрать"}
                        changes.append((f"Решение: {actions[a]} → {actions[b]}", ()))
                    elif key == "material":
                        changes.append((f"Материал: {a} → {b}", "error"))
                    else:
                        changes.append((f"{key}: {a} → {b}", ()))
            if changes:
                self.diff.insert("end", f"B{i+1} ↔ C{j+1} IoU={overlap:.2f}\n")
                for description, tag in changes:
                    self.diff.insert("end", description + "\n", tag)
                self.diff.insert("end", "\n")
        for i in missing:
            self.diff.insert("end",f"ПРОПАЛ B{i+1}: {base[i]['form']}/{base[i]['material']}\n")
        for j in added:
            self.diff.insert("end",f"ДОБАВЛЕН C{j+1}: {cand[j]['form']}/{cand[j]['material']} p={cand[j]['score']:.3f}\n")
            reference = material_reference(base, cand[j])
            if reference is not None and reference["material"] != cand[j]["material"]:
                self.diff.insert("end", f"Материал: {reference['material']} → {cand[j]['material']}\n", "error")
        self.diff.configure(state="disabled")

    def close(self):
        if self._save_job is not None:
            self.root.after_cancel(self._save_job)
            self._save_job = None
        self.save_state()
        self.root.destroy()


def main():
    root=tk.Tk()
    App(root)
    root.mainloop()


if __name__=="__main__":
    main()
