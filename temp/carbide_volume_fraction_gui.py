#!/usr/bin/env python3
"""
Carbide Volume Fraction Analyzer

Purpose:
- Load metallographic / cross-section images.
- Correct left-to-right brightness/contrast gradients.
- Segment angular carbide particles.
- Estimate carbide area fraction as a practical volume fraction estimate.
- Save overlay, mask, and CSV summary.

Install:
    pip install opencv-python pillow numpy

Run:
    python carbide_volume_fraction_gui.py
"""

import csv
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageTk


@dataclass
class AnalysisResult:
    carbide_area_px: int = 0
    analyzed_area_px: int = 0
    carbide_fraction: float = 0.0
    threshold: int = 140
    dark_exclude: int = 20
    min_particle_area: int = 25
    max_particle_area: int = 999999
    notes: str = ""


class CarbideAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Carbide Volume Fraction Analyzer")
        self.root.geometry("1320x850")

        self.original_bgr = None
        self.working_bgr = None
        self.gray = None
        self.mask = None
        self.overlay_bgr = None
        self.current_path = None
        self.result = AnalysisResult()

        self.zoom = 1.0
        self.roi_start = None
        self.roi_rect_id = None
        self.roi = None  # x1, y1, x2, y2 in original image coordinates

        self._build_ui()

    def _build_ui(self):
        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(main, width=310, padx=8, pady=8)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        right = tk.Frame(main, padx=8, pady=8)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        title = tk.Label(left, text="Carbide Volume Fraction", font=("Arial", 15, "bold"))
        title.pack(anchor="w", pady=(0, 8))

        tk.Button(left, text="Load Image", command=self.load_image, height=2).pack(fill=tk.X, pady=3)

        self.auto_lr_btn = tk.Button(
            left,
            text="AUTO LEFT → RIGHT\nBRIGHTNESS CORRECTION",
            command=self.auto_left_right_correction,
            height=3,
            bg="#d9edf7",
            activebackground="#c4e3f3",
            font=("Arial", 10, "bold")
        )
        self.auto_lr_btn.pack(fill=tk.X, pady=6)

        tk.Button(left, text="Auto Threshold", command=self.auto_threshold, height=2).pack(fill=tk.X, pady=3)
        tk.Button(left, text="Clear ROI / Use Full Image", command=self.clear_roi, height=2).pack(fill=tk.X, pady=3)

        tk.Label(left, text="\nSegmentation Controls", font=("Arial", 12, "bold")).pack(anchor="w")

        self.threshold_var = tk.IntVar(value=140)
        self.dark_exclude_var = tk.IntVar(value=20)
        self.blur_var = tk.IntVar(value=1)
        self.open_var = tk.IntVar(value=1)
        self.close_var = tk.IntVar(value=1)
        self.min_area_var = tk.IntVar(value=25)
        self.max_area_var = tk.IntVar(value=999999)
        self.invert_var = tk.BooleanVar(value=False)
        self.show_edges_var = tk.BooleanVar(value=True)

        self._slider(left, "Carbide threshold", self.threshold_var, 0, 255)
        self._slider(left, "Exclude black pits below", self.dark_exclude_var, 0, 100)
        self._slider(left, "Median blur", self.blur_var, 0, 9)
        self._slider(left, "Open/remove specks", self.open_var, 0, 7)
        self._slider(left, "Close/fill gaps", self.close_var, 0, 7)
        self._slider(left, "Minimum particle area px", self.min_area_var, 0, 2000)
        self._slider(left, "Maximum particle area px", self.max_area_var, 100, 999999)

        tk.Checkbutton(left, text="Invert threshold direction", variable=self.invert_var, command=self.update_analysis).pack(anchor="w")
        tk.Checkbutton(left, text="Show particle outlines", variable=self.show_edges_var, command=self.update_analysis).pack(anchor="w")

        self.result_label = tk.Label(
            left,
            text="Load an image to begin.",
            justify=tk.LEFT,
            anchor="w",
            bg="#f3f3f3",
            padx=8,
            pady=8,
            font=("Consolas", 10)
        )
        self.result_label.pack(fill=tk.X, pady=8)

        tk.Button(left, text="Save Overlay PNG", command=self.save_overlay, height=2).pack(fill=tk.X, pady=3)
        tk.Button(left, text="Save Mask PNG", command=self.save_mask, height=2).pack(fill=tk.X, pady=3)
        tk.Button(left, text="Export CSV Summary", command=self.export_csv, height=2).pack(fill=tk.X, pady=3)

        help_text = (
            "\nWorkflow:\n"
            "1. Load image\n"
            "2. Optional: drag ROI on image\n"
            "3. Click Auto Left → Right Correction\n"
            "4. Click Auto Threshold\n"
            "5. Fine tune sliders\n"
            "6. Save overlay / mask / CSV\n\n"
            "The reported area fraction is the\n"
            "2D estimate of volume fraction."
        )
        tk.Label(left, text=help_text, justify=tk.LEFT, anchor="w").pack(fill=tk.X, pady=8)

        self.canvas = tk.Canvas(right, bg="black", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_roi_start)
        self.canvas.bind("<B1-Motion>", self.on_roi_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_roi_end)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)
        self.canvas.bind("<Button-5>", self.on_mousewheel)

        self.image_on_canvas = None
        self.tk_img = None

    def _slider(self, parent, label, var, minval, maxval):
        row = tk.Frame(parent)
        row.pack(fill=tk.X, pady=2)

        tk.Label(row, text=label).pack(anchor="w")
        spin = tk.Spinbox(row, from_=minval, to=maxval, textvariable=var, width=9, command=self.update_analysis)
        spin.pack(side=tk.RIGHT)

        scale = tk.Scale(
            row,
            from_=minval,
            to=maxval,
            orient=tk.HORIZONTAL,
            variable=var,
            showvalue=False,
            command=lambda _v: self.update_analysis()
        )
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", "Could not load image.")
            return

        self.current_path = path
        self.original_bgr = img
        self.working_bgr = img.copy()
        self.roi = None
        self.zoom = 1.0
        self.update_analysis()

    def get_roi_slice(self):
        if self.working_bgr is None:
            return None, None

        h, w = self.working_bgr.shape[:2]
        if self.roi is None:
            return self.working_bgr, (0, 0, w, h)

        x1, y1, x2, y2 = self.roi
        x1, x2 = sorted([max(0, min(w - 1, x1)), max(0, min(w, x2))])
        y1, y2 = sorted([max(0, min(h - 1, y1)), max(0, min(h, y2))])

        if x2 <= x1 or y2 <= y1:
            return self.working_bgr, (0, 0, w, h)

        return self.working_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)

    def auto_left_right_correction(self):
        """
        Corrects a broad left-to-right brightness gradient.

        Method:
        - Convert to grayscale.
        - Estimate the column-wise background/illumination trend using median intensity.
        - Smooth the trend heavily so fine carbide texture is preserved.
        - Add/subtract the correction in grayscale, then apply it to the colour channels.
        """
        if self.working_bgr is None:
            return

        roi_img, bounds = self.get_roi_slice()
        x1, y1, x2, y2 = bounds

        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Column median is robust against particles, pits, and local texture.
        column_profile = np.median(gray, axis=0)

        # Smooth over a large portion of the width to model only gradual illumination changes.
        width = max(3, gray.shape[1])
        kernel = max(31, int(width * 0.12))
        if kernel % 2 == 0:
            kernel += 1

        profile_img = column_profile.reshape(1, -1).astype(np.float32)
        smooth = cv2.GaussianBlur(profile_img, (kernel, 1), 0).flatten()

        target = float(np.median(smooth))
        correction = target - smooth
        correction_2d = np.tile(correction, (gray.shape[0], 1))

        corrected_roi = roi_img.astype(np.float32)
        for c in range(3):
            corrected_roi[:, :, c] = corrected_roi[:, :, c] + correction_2d

        corrected_roi = np.clip(corrected_roi, 0, 255).astype(np.uint8)

        corrected_full = self.working_bgr.copy()
        corrected_full[y1:y2, x1:x2] = corrected_roi
        self.working_bgr = corrected_full

        self.update_analysis()

    def auto_threshold(self):
        if self.working_bgr is None:
            return

        roi_img, _ = self.get_roi_slice()
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

        # Exclude very dark background/pits before estimating Otsu threshold.
        usable = gray[gray > self.dark_exclude_var.get()]
        if usable.size < 100:
            usable = gray.flatten()

        # Otsu needs an image. Make a narrow one from the usable intensity values.
        temp = usable.reshape(-1, 1).astype(np.uint8)
        thresh, _ = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        self.threshold_var.set(int(thresh))
        self.update_analysis()

    def update_analysis(self):
        if self.working_bgr is None:
            return

        roi_img, bounds = self.get_roi_slice()
        x1, y1, x2, y2 = bounds
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

        blur_size = self.blur_var.get()
        if blur_size > 1:
            if blur_size % 2 == 0:
                blur_size += 1
            gray_proc = cv2.medianBlur(gray, blur_size)
        else:
            gray_proc = gray.copy()

        thresh = int(self.threshold_var.get())
        dark_exclude = int(self.dark_exclude_var.get())

        if self.invert_var.get():
            raw_mask = gray_proc <= thresh
        else:
            raw_mask = gray_proc >= thresh

        # Exclude black pits/background so they do not count as carbide.
        analysis_area = gray > dark_exclude
        raw_mask = raw_mask & analysis_area

        mask = raw_mask.astype(np.uint8) * 255

        open_k = self.open_var.get()
        if open_k > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k * 2 + 1, open_k * 2 + 1))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        close_k = self.close_var.get()
        if close_k > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k * 2 + 1, close_k * 2 + 1))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Remove particles outside area limits.
        min_area = int(self.min_area_var.get())
        max_area = int(self.max_area_var.get())

        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        filtered = np.zeros_like(mask)

        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area <= area <= max_area:
                filtered[labels == i] = 255

        self.mask = filtered

        carbide_area = int(np.count_nonzero(filtered))
        analyzed_area = int(np.count_nonzero(analysis_area))
        fraction = carbide_area / analyzed_area if analyzed_area > 0 else 0.0

        self.result = AnalysisResult(
            carbide_area_px=carbide_area,
            analyzed_area_px=analyzed_area,
            carbide_fraction=fraction,
            threshold=thresh,
            dark_exclude=dark_exclude,
            min_particle_area=min_area,
            max_particle_area=max_area
        )

        # Build overlay for full image.
        overlay_full = self.working_bgr.copy()
        roi_overlay = roi_img.copy()

        # Tint carbide mask light/bright without hiding original structure.
        color_layer = roi_overlay.copy()
        color_layer[filtered > 0] = (0, 255, 255)  # yellow in BGR
        roi_overlay = cv2.addWeighted(roi_overlay, 0.68, color_layer, 0.32, 0)

        if self.show_edges_var.get():
            contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(roi_overlay, contours, -1, (0, 0, 255), 1)

        overlay_full[y1:y2, x1:x2] = roi_overlay

        # Show ROI rectangle if present.
        if self.roi is not None:
            cv2.rectangle(overlay_full, (x1, y1), (x2, y2), (255, 255, 255), 2)

        self.overlay_bgr = overlay_full

        self.update_result_label()
        self.display_image(overlay_full)

    def update_result_label(self):
        vf = self.result.carbide_fraction * 100.0
        txt = (
            f"Carbide fraction: {vf:0.1f} vol%\n"
            f"Carbide pixels:   {self.result.carbide_area_px:,}\n"
            f"Analyzed pixels:  {self.result.analyzed_area_px:,}\n"
            f"Threshold:        {self.result.threshold}\n"
            f"Dark exclude:     {self.result.dark_exclude}\n"
            f"Min area px:      {self.result.min_particle_area}\n"
            f"Max area px:      {self.result.max_particle_area}"
        )
        self.result_label.config(text=txt)

    def display_image(self, bgr):
        if bgr is None:
            return

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        canvas_w = max(200, self.canvas.winfo_width())
        canvas_h = max(200, self.canvas.winfo_height())

        w, h = img.size
        fit_scale = min(canvas_w / w, canvas_h / h)
        scale = fit_scale * self.zoom

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        self.tk_img = ImageTk.PhotoImage(img_resized)
        self.canvas.delete("all")
        self.image_on_canvas = self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.tk_img, anchor=tk.CENTER)

    def canvas_to_image_coords(self, cx, cy):
        if self.working_bgr is None:
            return None

        canvas_w = max(200, self.canvas.winfo_width())
        canvas_h = max(200, self.canvas.winfo_height())
        h, w = self.working_bgr.shape[:2]
        fit_scale = min(canvas_w / w, canvas_h / h)
        scale = fit_scale * self.zoom

        shown_w = w * scale
        shown_h = h * scale
        left = canvas_w / 2 - shown_w / 2
        top = canvas_h / 2 - shown_h / 2

        ix = int((cx - left) / scale)
        iy = int((cy - top) / scale)
        ix = max(0, min(w, ix))
        iy = max(0, min(h, iy))
        return ix, iy

    def on_roi_start(self, event):
        if self.working_bgr is None:
            return
        self.roi_start = self.canvas_to_image_coords(event.x, event.y)
        self.roi_canvas_start = (event.x, event.y)
        if self.roi_rect_id is not None:
            self.canvas.delete(self.roi_rect_id)

    def on_roi_drag(self, event):
        if self.working_bgr is None or self.roi_start is None:
            return
        if self.roi_rect_id is not None:
            self.canvas.delete(self.roi_rect_id)
        x0, y0 = self.roi_canvas_start
        self.roi_rect_id = self.canvas.create_rectangle(x0, y0, event.x, event.y, outline="white", width=2)

    def on_roi_end(self, event):
        if self.working_bgr is None or self.roi_start is None:
            return
        end = self.canvas_to_image_coords(event.x, event.y)
        if end is None:
            return
        x1, y1 = self.roi_start
        x2, y2 = end
        if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
            self.roi = (x1, y1, x2, y2)
        self.roi_start = None
        self.update_analysis()

    def clear_roi(self):
        self.roi = None
        self.update_analysis()

    def on_mousewheel(self, event):
        if self.working_bgr is None:
            return

        # Windows/Mac event.delta, Linux Button-4/5.
        if getattr(event, "delta", 0) > 0 or getattr(event, "num", None) == 4:
            self.zoom *= 1.15
        else:
            self.zoom /= 1.15
        self.zoom = max(0.2, min(8.0, self.zoom))
        self.display_image(self.overlay_bgr if self.overlay_bgr is not None else self.working_bgr)

    def save_overlay(self):
        if self.overlay_bgr is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
            initialfile=self._default_name("overlay.png")
        )
        if path:
            cv2.imwrite(path, self.overlay_bgr)
            messagebox.showinfo("Saved", f"Overlay saved:\n{path}")

    def save_mask(self):
        if self.mask is None:
            return

        # Save full-size mask, not just ROI. ROI mask is placed into full image coordinates.
        h, w = self.working_bgr.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.uint8)
        _, bounds = self.get_roi_slice()
        x1, y1, x2, y2 = bounds
        full_mask[y1:y2, x1:x2] = self.mask

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
            initialfile=self._default_name("mask.png")
        )
        if path:
            cv2.imwrite(path, full_mask)
            messagebox.showinfo("Saved", f"Mask saved:\n{path}")

    def export_csv(self):
        if self.working_bgr is None:
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=self._default_name("summary.csv")
        )
        if not path:
            return

        rows = [
            ["timestamp", datetime.now().isoformat(timespec="seconds")],
            ["source_image", self.current_path or ""],
            ["carbide_fraction_percent", f"{self.result.carbide_fraction * 100.0:.3f}"],
            ["carbide_area_px", self.result.carbide_area_px],
            ["analyzed_area_px", self.result.analyzed_area_px],
            ["threshold", self.result.threshold],
            ["dark_exclude", self.result.dark_exclude],
            ["median_blur", self.blur_var.get()],
            ["open_remove_specks", self.open_var.get()],
            ["close_fill_gaps", self.close_var.get()],
            ["min_particle_area_px", self.min_area_var.get()],
            ["max_particle_area_px", self.max_area_var.get()],
            ["invert_threshold_direction", self.invert_var.get()],
            ["roi", self.roi if self.roi is not None else "full image"],
            ["method_note", "2D carbide area fraction used as estimated volume fraction"],
        ]

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        messagebox.showinfo("Saved", f"CSV summary saved:\n{path}")

    def _default_name(self, suffix):
        if not self.current_path:
            return f"carbide_{suffix}"
        base = os.path.splitext(os.path.basename(self.current_path))[0]
        return f"{base}_carbide_{suffix}"


if __name__ == "__main__":
    root = tk.Tk()
    app = CarbideAnalyzerGUI(root)
    root.mainloop()
