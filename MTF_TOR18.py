import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import customtkinter as ctk
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox, simpledialog

from PIL import Image, ImageTk
from scipy.optimize import curve_fit

import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader


# =========================================================
# CONFIG UI
# =========================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

APP_TITLE = "Analyse MTF TOR18"
DEFAULT_REPORT_TITLE = "Rapport d'analyse MTF - TOR18"
DEFAULT_PDF_FILENAME = "mtf_tor18_report.pdf"

AUTHOR_NAME = "le CHR Metz-Thionville"
LINKEDIN_URL = "https://www.linkedin.com/in/motchy-saleh/"
EMAIL_ADDRESS = "motchy.saleh@chr-metz-thionville.fr"

# =========================================================
# OUTILS IMAGE
# =========================================================
def is_dicom(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            preamble = f.read(132)
            return len(preamble) >= 132 and preamble[128:132] == b"DICM"
    except Exception:
        return False


def _normalize_to_uint16(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    minv, maxv = np.nanmin(arr), np.nanmax(arr)
    if maxv <= minv:
        return np.zeros(arr.shape, dtype=np.uint16)
    norm = (arr - minv) / (maxv - minv)
    return (norm * 65535.0).clip(0, 65535).astype(np.uint16)


def load_dicom_image(path: str, frame_index: int = 0, use_voi_lut: bool = True) -> np.ndarray:
    ds = pydicom.dcmread(path, force=True)
    px = ds.pixel_array

    if px.ndim == 3 and getattr(ds, "NumberOfFrames", 1) > 1:
        px = px[frame_index]

    if px.ndim == 3 and px.shape[-1] == 3:
        px = cv2.cvtColor(px.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    try:
        px = apply_modality_lut(px, ds)
    except Exception:
        pass

    if use_voi_lut:
        try:
            px = apply_voi_lut(px, ds)
        except Exception:
            pass

    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if str(photometric).upper() == "MONOCHROME1":
        px = (np.max(px) - px)

    if px.dtype != np.uint16:
        px = _normalize_to_uint16(px)

    return px


def load_standard_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if image is None:
        try:
            pil_img = Image.open(path)
            pil_img = pil_img.convert("I")
            image = np.array(pil_img)
        except Exception:
            raise ValueError(f"Échec de lecture: {path}")

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.dtype == np.bool_:
        image = image.astype(np.uint8) * 255

    if image.dtype == np.uint8:
        image = (image.astype(np.uint16) * 257)
    elif image.dtype == np.uint16:
        pass
    else:
        image = _normalize_to_uint16(image)

    return image


def load_image_any(path: str) -> np.ndarray:
    if is_dicom(path):
        return load_dicom_image(path, frame_index=0, use_voi_lut=True)
    return load_standard_image(path)


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated


def window_image_to_uint8(image: np.ndarray, wl: int, ww: int) -> np.ndarray:
    ww = max(1, int(ww))
    wl = int(wl)

    min_val = wl - ww // 2
    max_val = wl + ww // 2

    img_display = np.clip(image.astype(np.int32), min_val, max_val)
    img_display = ((img_display - min_val) / max(1, (max_val - min_val)) * 255).astype(np.uint8)
    return img_display


def preprocess(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype not in [np.uint8, np.uint16]:
        raise ValueError(f"Type d'image non supporté : {image.dtype}")
    return clahe.apply(image)


# =========================================================
# DÉTECTION / DESSIN / CALCUL
# =========================================================
def detect_bar_rois(image: np.ndarray, threshold: int = 110, min_area: int = 50, erosion_iter: int = 0):
    if image.dtype == np.uint16:
        thresh_16 = int(threshold * 256)
        _, binary_16 = cv2.threshold(image, thresh_16, 65535, cv2.THRESH_BINARY_INV)
        binary = (binary_16 / 256).astype(np.uint8)
    else:
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)

    if erosion_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=erosion_iter)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bars = [c for c in contours if cv2.contourArea(c) > min_area]
    bars = sorted(bars, key=lambda c: cv2.contourArea(c), reverse=True)
    return bars


def extract_bar_rois(image: np.ndarray, bars):
    rois = []
    for cnt in bars:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = image[y:y + h, x:x + w]
        rois.append(roi)
    return rois


def draw_rois_with_numbers(image: np.ndarray, bars, numbers):
    if image.dtype == np.uint16:
        image_8bit = (image / 256).astype(np.uint8)
    else:
        image_8bit = image.copy()

    if image_8bit.ndim == 2:
        image_rgb = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2BGR)
    else:
        image_rgb = image_8bit.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, cnt in enumerate(bars):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 0, 255), 1)

        text = str(numbers[i])
        target_h = max(10, int(h * 0.45))
        font_scale = max(0.20, min(0.55, target_h / 22.0))
        thickness = 1

        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        max_text_w = max(8, int(w * 0.75))

        if tw > max_text_w:
            shrink = max_text_w / max(1, tw)
            font_scale = max(0.20, font_scale * shrink)
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

        cx = x + (w - tw) // 2
        cy = y + (h + th) // 2

        cv2.putText(
            image_rgb,
            text,
            (cx, cy),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
            lineType=cv2.LINE_AA
        )

    return image_rgb


def compute_ftm_reglementaire(roi_groupe: np.ndarray, sigma_fond: float, ct_espace: float, ct_mat: float) -> float:
    valid_pixels = roi_groupe[roi_groupe < 65535]
    if valid_pixels.size == 0:
        return 0.0

    sigma_groupe = np.std(valid_pixels)
    numerator = np.sqrt(np.abs(sigma_groupe ** 2 - sigma_fond ** 2))
    denominator = abs(ct_mat - ct_espace)

    if denominator == 0:
        return 0.0

    return (np.pi / np.sqrt(2)) * (numerator / denominator)


def exp_func(x, a, b):
    return a * np.exp(b * x)


def compute_ftm_threshold_frequency(a, b, target):
    if a <= 0 or target <= 0 or b == 0:
        return None

    ratio = target / a
    if ratio <= 0:
        return None

    x = np.log(ratio) / b
    if not np.isfinite(x):
        return None
    return float(x)


def generate_pdf_report(
    pdf_path,
    annotated_image_path,
    graph_image_path,
    image_name,
    angle_deg,
    wl,
    ww,
    sigma_fond,
    ct_espace,
    ct_mat,
    exp_a,
    exp_b,
    ftm50,
    ftm10,
    report_title=DEFAULT_REPORT_TITLE,
):
    c = pdf_canvas.Canvas(pdf_path, pagesize=A4)
    page_w, page_h = A4

    margin = 15 * mm
    usable_w = page_w - 2 * margin
    y = page_h - margin

    def draw_centered_title(text, y_pos, size=16):
        c.setFont("Helvetica-Bold", size)
        text_w = c.stringWidth(text, "Helvetica-Bold", size)
        c.drawString((page_w - text_w) / 2, y_pos, text)

    def draw_centered_image(img_path, y_top, max_w, max_h):
        img = ImageReader(img_path)
        iw, ih = img.getSize()
        scale = min(max_w / iw, max_h / ih)
        draw_w = iw * scale
        draw_h = ih * scale
        x = (page_w - draw_w) / 2
        y_img = y_top - draw_h
        c.drawImage(img_path, x, y_img, width=draw_w, height=draw_h, preserveAspectRatio=True, mask='auto')
        return y_img, draw_h

    draw_centered_title(report_title, y, size=16)
    y -= 10 * mm

    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Image : {os.path.basename(image_name) if image_name else 'N/A'}")
    y -= 5 * mm
    c.drawString(margin, y, f"Rotation : {angle_deg} deg")
    y -= 5 * mm
    c.drawString(margin, y, f"WL : {wl}    WW : {ww}")
    y -= 8 * mm

    if os.path.exists(annotated_image_path):
        draw_centered_title("Image TOR18 segmentée", y, size=13)
        y -= 5 * mm
        _, draw_h = draw_centered_image(annotated_image_path, y, usable_w, 85 * mm)
        y -= draw_h + 8 * mm

    if os.path.exists(graph_image_path):
        draw_centered_title("Courbe FTM", y, size=13)
        y -= 5 * mm
        _, draw_h = draw_centered_image(graph_image_path, y, usable_w, 80 * mm)
        y -= draw_h + 8 * mm

    if y < 50 * mm:
        c.showPage()
        y = page_h - margin

    draw_centered_title("Résultats", y, size=14)
    y -= 10 * mm

    ftm50_txt = f"{ftm50:.4f} lp/mm" if ftm50 is not None and np.isfinite(ftm50) else "Non définie"
    ftm10_txt = f"{ftm10:.4f} lp/mm" if ftm10 is not None and np.isfinite(ftm10) else "Non définie"

    c.setFont("Helvetica", 11)
    c.drawString(margin, y, f"Sigma fond : {sigma_fond:.4f}")
    y -= 6 * mm
    c.drawString(margin, y, f"CT espace : {ct_espace:.4f}")
    y -= 6 * mm
    c.drawString(margin, y, f"CT matériau : {ct_mat:.4f}")
    y -= 6 * mm
    c.drawString(margin, y, f"Ajustement exponentiel : y = {exp_a:.6g} * exp({exp_b:.6g} * x)")
    y -= 8 * mm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, f"FTM 50 % : {ftm50_txt}")
    y -= 7 * mm
    c.drawString(margin, y, f"FTM 10 % : {ftm10_txt}")
    c.save()


# =========================================================
# APPLICATION CTK
# =========================================================
class MTFApplication:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1450x900")
        self.root.minsize(1250, 800)

        self.steps = [
            "1. Sélection",
            "2. Rotation / Windowing",
            "3. ROIs recherche",
            "4. Fond / Matériau",
            "5. Édition",
            "6. Calcul / Export",
        ]
        self.current_step = 0

        self.image_path = None
        self.image = None
        self.image_rotated = None
        self.image_display = None
        self.image_eq = None
        self.image_base_rotated = None
        self.image_base_display = None
        self.image_base_eq = None

        self.zoom_factor = 1.0
        self.view_x0 = 0.0
        self.view_y0 = 0.0
        self.view_w = None
        self.view_h = None
        self.wl = 0
        self.ww = 1
        self.angle = 0

        self.rois_search = []
        self.roi_fond_coords = None
        self.roi_mat_coords = None
        self.rois_temp = []

        self.bars = []
        self.numbers = []
        self.threshold_value = 110
        self.min_area_value = 5
        self.erosion_value = 0
        self.resegment_target_index = None
        self.threshold_preview_mode = False

        self.history_stack = []
        self.initial_state = None
        self.edit_mode = "number"
        self.auto_number_order = "top_right_to_bottom_left"
        self.last_modified_text = "Prêt"

        self.display_scale = 1.0
        self.display_offset_x = 0
        self.display_offset_y = 0
        self.displayed_image_shape = None

        self.current_roi = None
        self.roi_start = None
        self.selecting_rois = False
        self.selecting_single_roi = False
        self.roi_type = None

        self.results_ready = False
        self.last_results = {}
        self.report_title_var = tk.StringVar(value=DEFAULT_REPORT_TITLE)

        self.img_tk = None
        self.step_badges = []

        self.setup_ui()
        self.update_step()
        self.set_status("Application prête.")

    # ---------------------------
    # UI
    # ---------------------------
    def open_linkedin(self):
        try:
            webbrowser.open(LINKEDIN_URL)
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'ouvrir LinkedIn.\n\n{e}")

    def open_email(self):
        try:
            webbrowser.open(f"mailto:{EMAIL_ADDRESS}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'ouvrir le client mail.\n\n{e}")
    def setup_ui(self):
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.topbar = ctk.CTkFrame(self.root, corner_radius=0, height=64)
        self.topbar.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.topbar.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            self.topbar,
            text=APP_TITLE,
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.grid(row=0, column=0, padx=18, pady=(10, 2), sticky="w")

        self.steps_frame = ctk.CTkFrame(self.topbar, fg_color="transparent")
        self.steps_frame.grid(row=1, column=0, padx=16, pady=(0, 8), sticky="ew")
        self.steps_frame.grid_columnconfigure(tuple(range(len(self.steps))), weight=1)

        for i, step in enumerate(self.steps):
            badge = ctk.CTkLabel(
                self.steps_frame,
                text=step,
                corner_radius=16,
                fg_color=("#2b2b2b", "#2b2b2b"),
                text_color="#b8b8b8",
                padx=14,
                pady=8,
                font=ctk.CTkFont(size=13, weight="bold")
            )
            badge.grid(row=0, column=i, padx=6, pady=2, sticky="ew")
            self.step_badges.append(badge)

        self.sidebar = ctk.CTkFrame(self.root, width=330, corner_radius=0)
        self.sidebar.grid(row=1, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        self.viewer_frame = ctk.CTkFrame(self.root, corner_radius=0)
        self.viewer_frame.grid(row=1, column=1, sticky="nsew")
        self.viewer_frame.grid_rowconfigure(1, weight=1)
        self.viewer_frame.grid_columnconfigure(0, weight=1)

        self.viewer_header = ctk.CTkFrame(self.viewer_frame, height=44)
        self.viewer_header.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))
        self.viewer_header.grid_columnconfigure(1, weight=1)
        # =========================
        # Footer contact
        # =========================
        self.footer = ctk.CTkFrame(self.root, height=40)
        self.footer.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.footer.grid_columnconfigure(0, weight=1)
        self.footer.grid_columnconfigure(1, weight=1)
        self.footer_left = ctk.CTkFrame(self.footer, fg_color="transparent")
        self.footer_left.grid(row=0, column=0, sticky="w", padx=14, pady=6)

        self.footer_right = ctk.CTkFrame(self.footer, fg_color="transparent")
        self.footer_right.grid(row=0, column=1, sticky="e", padx=14, pady=6)

        self.author_label = ctk.CTkLabel(
            self.footer_left,
            text=f"Développé par {AUTHOR_NAME}",
            font=ctk.CTkFont(size=12)
        )
        self.author_label.pack(side="left", padx=(0, 12))

        self.linkedin_link = ctk.CTkLabel(
            self.footer_right,
            text="LinkedIn",
            text_color="#4da6ff",
            cursor="hand2",
            font=ctk.CTkFont(size=12, underline=True)
        )
        self.linkedin_link.pack(side="left", padx=6)

        self.linkedin_link.bind("<Button-1>", lambda e: self.open_linkedin())

        self.mail_link = ctk.CTkLabel(
            self.footer_right,
            text=EMAIL_ADDRESS,
            text_color="#4da6ff",
            cursor="hand2",
            font=ctk.CTkFont(size=12, underline=True)
        )
        self.mail_link.pack(side="left", padx=6)

        self.mail_link.bind("<Button-1>", lambda e: self.open_email())

        ctk.CTkLabel(
            self.viewer_header,
            text="Vue de travail",
            font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, padx=12, pady=8, sticky="w")

        self.status_label = ctk.CTkLabel(
            self.viewer_header,
            text="",
            text_color="#9ecbff",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.grid(row=0, column=1, padx=12, pady=8, sticky="e")

        self.canvas_container = ctk.CTkFrame(self.viewer_frame, corner_radius=14)
        self.canvas_container.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        self.canvas_container.grid_rowconfigure(0, weight=1)
        self.canvas_container.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.canvas_container, bg="#111111", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)
        self.canvas.bind("<Button-5>", self.on_mousewheel)

        self.sidebar_scroll = ctk.CTkScrollableFrame(self.sidebar, corner_radius=0)
        self.sidebar_scroll.pack(fill="both", expand=True, padx=12, pady=12)

        self.card_action = ctk.CTkFrame(self.sidebar_scroll, corner_radius=14)
        self.card_action.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(self.card_action, text="Action", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=14, pady=(12, 6))
        self.ok_button = ctk.CTkButton(self.card_action, text="Valider l'étape", height=40, command=self.next_step)
        self.ok_button.pack(fill="x", padx=14, pady=(0, 14))

        self.card_step = ctk.CTkFrame(self.sidebar_scroll, corner_radius=14)
        self.card_step.pack(fill="x")

        self.select_button = ctk.CTkButton(self.card_step, text="Sélectionner une image", command=self.select_image)

        self.slider_frame = ctk.CTkFrame(self.card_step, fg_color="transparent")
        self.angle_value_label = ctk.CTkLabel(self.slider_frame, text="0°")
        self.wl_value_label = ctk.CTkLabel(self.slider_frame, text="0")
        self.ww_value_label = ctk.CTkLabel(self.slider_frame, text="1")

        ctk.CTkLabel(self.slider_frame, text="Angle de rotation", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=8, pady=(6, 0))
        self.angle_slider = ctk.CTkSlider(self.slider_frame, from_=-180, to=180, number_of_steps=360, command=self.update_rotation)
        self.angle_slider.pack(fill="x", padx=8, pady=(6, 2))
        self.angle_value_label.pack(anchor="e", padx=10)

        ctk.CTkLabel(self.slider_frame, text="Window Level (WL)", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=8, pady=(12, 0))
        self.wl_slider = ctk.CTkSlider(self.slider_frame, from_=0, to=65535, number_of_steps=65535, command=self.update_windowing)
        self.wl_slider.pack(fill="x", padx=8, pady=(6, 2))
        self.wl_value_label.pack(anchor="e", padx=10)

        ctk.CTkLabel(self.slider_frame, text="Window Width (WW)", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=8, pady=(12, 0))
        self.ww_slider = ctk.CTkSlider(self.slider_frame, from_=1, to=65535, number_of_steps=65534, command=self.update_windowing)
        self.ww_slider.pack(fill="x", padx=8, pady=(6, 2))
        self.ww_value_label.pack(anchor="e", padx=10, pady=(0, 6))

        self.roi_button = ctk.CTkButton(
            self.card_step,
            text="Sélectionner ROIs\nde recherche",
            height=58,
            command=self.select_rois
        )
        self.fond_button = ctk.CTkButton(self.card_step, text="Sélectionner zone fond", height=58, command=self.select_fond)
        self.mat_button = ctk.CTkButton(self.card_step, text="Sélectionner zone \n matériau", height=58, command=self.select_mat)

        self.edit_frame = ctk.CTkFrame(self.card_step, fg_color="transparent")
        self.edit_buttons = {}
        edit_specs = [
            ("number", "Modifier numéros"),
            ("roi", "Re-segmenter"),
            ("split", "Séparation verticale"),
            ("delete", "Supprimer"),
        ]
        for mode, text in edit_specs:
            btn = ctk.CTkButton(self.edit_frame, text=text, command=lambda m=mode: self.set_edit_mode(m))
            btn.pack(fill="x", padx=8, pady=4)
            self.edit_buttons[mode] = btn

        ctk.CTkButton(self.edit_frame, text="Annuler dernière", command=self.undo_edit).pack(fill="x", padx=8, pady=4)
        ctk.CTkButton(self.edit_frame, text="Reset", command=self.reset_edit).pack(fill="x", padx=8, pady=4)
        ctk.CTkButton(self.edit_frame, text="Auto numéros", command=self.auto_number).pack(fill="x", padx=8, pady=4)
        ctk.CTkButton(self.edit_frame, text="Changer ordre", command=self.toggle_auto_order).pack(fill="x", padx=8, pady=4)

        self.threshold_frame = ctk.CTkFrame(self.card_step, corner_radius=12)
        ctk.CTkLabel(self.threshold_frame, text="Réglages détection", font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=10, pady=(10, 2))

        self.threshold_value_label = ctk.CTkLabel(self.threshold_frame, text=str(self.threshold_value))
        self.min_area_value_label = ctk.CTkLabel(self.threshold_frame, text=str(self.min_area_value))
        self.erosion_value_label = ctk.CTkLabel(self.threshold_frame, text=str(self.erosion_value))

        ctk.CTkLabel(self.threshold_frame, text="Seuil").pack(anchor="w", padx=10, pady=(6, 0))
        self.threshold_slider = ctk.CTkSlider(self.threshold_frame, from_=0, to=255, number_of_steps=255, command=self.on_threshold_change)
        self.threshold_slider.pack(fill="x", padx=10, pady=(4, 0))
        self.threshold_value_label.pack(anchor="e", padx=10)

        ctk.CTkLabel(self.threshold_frame, text="Aire min").pack(anchor="w", padx=10, pady=(8, 0))
        self.min_area_slider = ctk.CTkSlider(self.threshold_frame, from_=1, to=1000, number_of_steps=999, command=self.on_threshold_change)
        self.min_area_slider.pack(fill="x", padx=10, pady=(4, 0))
        self.min_area_value_label.pack(anchor="e", padx=10)

        ctk.CTkLabel(self.threshold_frame, text="Érosion").pack(anchor="w", padx=10, pady=(8, 0))
        self.erosion_slider = ctk.CTkSlider(self.threshold_frame, from_=0, to=5, number_of_steps=5, command=self.on_threshold_change)
        self.erosion_slider.pack(fill="x", padx=10, pady=(4, 0))
        self.erosion_value_label.pack(anchor="e", padx=10, pady=(0, 10))

        self.final_frame = ctk.CTkFrame(self.card_step, corner_radius=12)
        ctk.CTkLabel(self.final_frame, text="Export", font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=12, pady=(12, 4))
        ctk.CTkLabel(self.final_frame, text="Titre du rapport PDF").pack(anchor="w", padx=12, pady=(4, 0))
        self.report_title_entry = ctk.CTkEntry(self.final_frame, textvariable=self.report_title_var)
        self.report_title_entry.pack(fill="x", padx=12, pady=(6, 8))
        self.save_pdf_button = ctk.CTkButton(self.final_frame, text="Enregistrer le PDF", command=self.save_pdf_report)
        self.save_pdf_button.pack(fill="x", padx=12, pady=6)
        self.restart_button = ctk.CTkButton(self.final_frame, text="Refaire une analyse", fg_color="#444", hover_color="#555", command=self.restart_analysis)
        self.restart_button.pack(fill="x", padx=12, pady=(6, 12))

        self.info_card = ctk.CTkFrame(self.sidebar_scroll, corner_radius=14)
        self.info_card.pack(fill="x", pady=(12, 0))
        ctk.CTkLabel(self.info_card, text="Informations", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=14, pady=(12, 6))
        self.info_label = ctk.CTkLabel(
            self.info_card,
            text="Charge une image pour commencer.",
            justify="left",
            anchor="w",
            text_color="#cccccc"
        )
        self.info_label.pack(fill="x", padx=14, pady=(0, 12))
        self.info_label.pack(anchor="w", padx=14, pady=(0, 12))
        self.info_card.bind("<Configure>", self.update_info_wraplength)
        self.threshold_slider.set(self.threshold_value)
        self.min_area_slider.set(self.min_area_value)
        self.erosion_slider.set(self.erosion_value)
        self.root.after(200, self.update_info_wraplength)

    def update_info_wraplength(self, event=None):
        try:
            card_width = self.info_card.winfo_width()
            wrap = max(120, card_width - 28)  # marge interne gauche/droite
            self.info_label.configure(wraplength=wrap)
        except Exception:
            pass
    def set_status(self, text: str):
        self.last_modified_text = text
        self.status_label.configure(text=text)
        self.info_label.configure(text=text)

    def clear_step_card(self):
        for widget in self.card_step.winfo_children():
            widget.pack_forget()

    def update_step(self):
        for i, badge in enumerate(self.step_badges):
            if i < self.current_step:
                badge.configure(fg_color="#1f6f43", text_color="white")
            elif i == self.current_step:
                badge.configure(fg_color="#1f538d", text_color="white")
            else:
                badge.configure(fg_color="#2b2b2b", text_color="#b8b8b8")

        self.clear_step_card()
        ctk.CTkLabel(self.card_step, text=self.steps[self.current_step], font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=14, pady=(12, 8))

        if self.current_step == 0:
            self.select_button.pack(fill="x", padx=14, pady=(0, 14))
            self.set_status("Sélectionne une image DICOM ou standard.")

        elif self.current_step == 1:
            self.slider_frame.pack(fill="x", padx=8, pady=(0, 12))
            self.update_image_display()
            self.set_status("Ajuste la rotation, le WL et le WW. Molette = zoom.")

        elif self.current_step == 2:
            self.roi_button.pack(fill="x", padx=14, pady=(0, 14))
            self.update_image_display()
            self.set_status("Sélectionne une ou plusieurs ROIs de recherche.")

        elif self.current_step == 3:
            self.fond_button.pack(fill="x", padx=14, pady=(0, 8))
            self.mat_button.pack(fill="x", padx=14, pady=(0, 14))
            self.update_image_display()
            self.set_status("Sélectionne la zone fond puis la zone matériau.")

        elif self.current_step == 4:
            self.edit_frame.pack(fill="x", padx=8, pady=(0, 12))
            self.threshold_frame.pack(fill="x", padx=14, pady=(0, 14))
            self.display_bars_for_edit()
            self.set_status("Édite les barres détectées puis valide l'étape.")

        elif self.current_step == 5:
            self.canvas.unbind("<Button-1>")
            self.final_frame.pack(fill="x", padx=14, pady=(0, 14))
            if not self.results_ready:
                self.compute_ftm()
            self.set_status("Choisis le titre du rapport puis enregistre le PDF.")

    # ---------------------------
    # Base images / viewport
    # ---------------------------
    def rebuild_base_images(self):
        if self.image is None:
            return
        self.image_base_rotated = rotate_image(self.image, self.angle)
        self.image_base_display = window_image_to_uint8(self.image_base_rotated, self.wl, self.ww)
        self.image_base_eq = preprocess(self.image_base_display)

    def reset_viewport(self):
        if self.image_base_display is None:
            return
        h, w = self.image_base_display.shape[:2]
        self.zoom_factor = 1.0
        self.view_x0 = 0.0
        self.view_y0 = 0.0
        self.view_w = float(w)
        self.view_h = float(h)

    def build_working_image(self):
        if self.image_base_display is None:
            return None

        base = self.image_base_display
        h, w = base.shape[:2]

        if self.view_w is None or self.view_h is None:
            self.view_x0 = 0.0
            self.view_y0 = 0.0
            self.view_w = float(w)
            self.view_h = float(h)

        x0 = int(round(self.view_x0))
        y0 = int(round(self.view_y0))
        x1 = int(round(self.view_x0 + self.view_w))
        y1 = int(round(self.view_y0 + self.view_h))

        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(x0 + 1, min(w, x1))
        y1 = max(y0 + 1, min(h, y1))

        cropped = base[y0:y1, x0:x1]
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return resized

    # ---------------------------
    # Canvas display
    # ---------------------------
    def show_image_on_canvas(self, img_gray_or_rgb: np.ndarray):
        if img_gray_or_rgb is None:
            return

        img_to_show = img_gray_or_rgb.copy()
        h, w = img_to_show.shape[:2]
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())

        scale = min(canvas_w / w, canvas_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(img_to_show, (new_w, new_h), interpolation=interp)

        img_pil = Image.fromarray(resized)
        self.img_tk = ImageTk.PhotoImage(img_pil)

        self.canvas.delete("all")
        offset_x = (canvas_w - new_w) // 2
        offset_y = (canvas_h - new_h) // 2
        self.canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self.img_tk)

        self.display_scale = scale
        self.display_offset_x = offset_x
        self.display_offset_y = offset_y
        self.displayed_image_shape = (h, w)

    def show_zoomed_overlay(self, img_full):
        if img_full is None:
            return

        h, w = img_full.shape[:2]
        x0 = int(round(self.view_x0))
        y0 = int(round(self.view_y0))
        x1 = int(round(self.view_x0 + self.view_w))
        y1 = int(round(self.view_y0 + self.view_h))

        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(x0 + 1, min(w, x1))
        y1 = max(y0 + 1, min(h, y1))

        cropped = img_full[y0:y1, x0:x1]
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        if resized.ndim == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        self.show_image_on_canvas(resized)

    def canvas_to_image_coords(self, x_canvas, y_canvas):
        if self.displayed_image_shape is None:
            return None, None

        x_img = int((x_canvas - self.display_offset_x) / self.display_scale)
        y_img = int((y_canvas - self.display_offset_y) / self.display_scale)

        h, w = self.displayed_image_shape
        x_img = max(0, min(w - 1, x_img))
        y_img = max(0, min(h - 1, y_img))
        return x_img, y_img

    def canvas_to_base_coords(self, x_canvas, y_canvas):
        if self.displayed_image_shape is None or self.image_base_display is None:
            return None, None

        x_img = (x_canvas - self.display_offset_x) / self.display_scale
        y_img = (y_canvas - self.display_offset_y) / self.display_scale

        hdisp, wdisp = self.displayed_image_shape
        x_img = max(0, min(wdisp - 1, x_img))
        y_img = max(0, min(hdisp - 1, y_img))

        bx = self.view_x0 + (x_img / wdisp) * self.view_w
        by = self.view_y0 + (y_img / hdisp) * self.view_h

        hbase, wbase = self.image_base_display.shape[:2]
        bx = int(max(0, min(wbase - 1, round(bx))))
        by = int(max(0, min(hbase - 1, round(by))))
        return bx, by

    # ---------------------------
    # Image controls
    # ---------------------------
    def select_image(self):
        image_path = filedialog.askopenfilename(
            title="Sélectionnez une image",
            filetypes=[
                ("Tous les fichiers supportés", "*.dcm;*.dicom;*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.webp"),
                ("Fichiers DICOM", "*.dcm;*.dicom"),
                ("Images standards", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.webp"),
                ("Tous fichiers", "*.*"),
            ]
        )
        if not image_path:
            return

        try:
            self.image_path = image_path
            self.image = load_image_any(image_path)
            self.wl = int((int(np.max(self.image)) + int(np.min(self.image))) / 2)
            self.ww = max(1, int(np.max(self.image)) - int(np.min(self.image)))
            self.angle = 0

            self.rebuild_base_images()
            self.reset_viewport()
            self.update_sliders()
            self.update_image_display()
            self.set_status(f"Image chargée : {os.path.basename(image_path)}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger l'image.\n\n{e}")

    def update_sliders(self):
        self.angle_slider.set(self.angle)
        self.wl_slider.set(self.wl)
        self.ww_slider.set(self.ww)
        self.angle_value_label.configure(text=f"{int(self.angle)}°")
        self.wl_value_label.configure(text=str(int(self.wl)))
        self.ww_value_label.configure(text=str(int(self.ww)))

    def update_rotation(self, _val):
        self.angle = int(round(self.angle_slider.get()))
        self.angle_value_label.configure(text=f"{self.angle}°")

        if self.image is None:
            return

        old_view_x0 = self.view_x0
        old_view_y0 = self.view_y0
        old_view_w = self.view_w
        old_view_h = self.view_h
        old_zoom = self.zoom_factor

        self.rebuild_base_images()
        h, w = self.image_base_display.shape[:2]

        if old_view_w is None or old_view_h is None:
            self.reset_viewport()
        else:
            self.view_w = min(float(w), old_view_w)
            self.view_h = min(float(h), old_view_h)
            self.view_x0 = max(0.0, min(w - self.view_w, old_view_x0))
            self.view_y0 = max(0.0, min(h - self.view_h, old_view_y0))
            self.zoom_factor = old_zoom

        self.update_image_display()

    def update_windowing(self, _val):
        self.wl = int(round(self.wl_slider.get()))
        self.ww = max(1, int(round(self.ww_slider.get())))
        self.wl_value_label.configure(text=str(self.wl))
        self.ww_value_label.configure(text=str(self.ww))

        if self.image is None:
            return

        self.rebuild_base_images()
        self.update_image_display()

    def update_image_display(self):
        if self.image_base_display is None:
            return

        self.image_display = self.build_working_image()

        if self.current_step == 4:
            if self.resegment_target_index is not None and self.edit_mode == "roi":
                self.preview_resegment_single_bar()
            else:
                self.update_bars_display()
        else:
            self.show_image_on_canvas(self.image_display)

    def on_mousewheel(self, event):
        if self.image is None or self.displayed_image_shape is None:
            return

        if hasattr(event, "delta"):
            zoom_mult = 1.15 if event.delta > 0 else 1 / 1.15
        else:
            zoom_mult = 1.15 if event.num == 4 else 1 / 1.15

        old_zoom = self.zoom_factor
        new_zoom = max(1.0, min(8.0, old_zoom * zoom_mult))
        if abs(new_zoom - old_zoom) < 1e-6:
            return

        base_x, base_y = self.canvas_to_base_coords(event.x, event.y)
        if base_x is None:
            return

        h, w = self.image_base_display.shape[:2]
        if self.view_w is None or self.view_h is None:
            self.reset_viewport()

        src_x = float(base_x)
        src_y = float(base_y)
        new_view_w = w / new_zoom
        new_view_h = h / new_zoom

        canvas_img_x, canvas_img_y = self.canvas_to_image_coords(event.x, event.y)
        if canvas_img_x is None:
            return

        new_x0 = src_x - (canvas_img_x / w) * new_view_w
        new_y0 = src_y - (canvas_img_y / h) * new_view_h
        new_x0 = max(0.0, min(w - new_view_w, new_x0))
        new_y0 = max(0.0, min(h - new_view_h, new_y0))

        self.zoom_factor = new_zoom
        self.view_x0 = new_x0
        self.view_y0 = new_y0
        self.view_w = new_view_w
        self.view_h = new_view_h

        self.image_rotated = self.build_working_image()
        self.image_display = window_image_to_uint8(self.image_rotated, self.wl, self.ww)
        self.image_eq = preprocess(self.image_display)
        self.update_image_display()

    # ---------------------------
    # ROI selection
    # ---------------------------
    def select_rois(self):
        if self.image_display is None:
            messagebox.showerror("Erreur", "Charge d'abord une image.")
            return

        self.selecting_rois = True
        self.rois_temp = []
        self.canvas.bind("<ButtonPress-1>", self.start_roi)
        self.canvas.bind("<B1-Motion>", self.draw_roi)
        self.canvas.bind("<ButtonRelease-1>", self.end_roi)
        self.root.bind("<Return>", self.finish_roi_selection)
        self.set_status("Sélection multiple active. Trace les ROIs puis appuie sur Entrée.")

    def start_roi(self, event):
        if not self.selecting_rois:
            return
        self.roi_start = (event.x, event.y)
        self.current_roi = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#ff4d4d", width=2)

    def draw_roi(self, event):
        if not self.selecting_rois or self.current_roi is None:
            return
        self.canvas.coords(self.current_roi, self.roi_start[0], self.roi_start[1], event.x, event.y)

    def end_roi(self, event):
        if not self.selecting_rois or self.current_roi is None:
            return

        x1, y1, x2, y2 = self.canvas.coords(self.current_roi)
        img_x1, img_y1 = self.canvas_to_base_coords(x1, y1)
        img_x2, img_y2 = self.canvas_to_base_coords(x2, y2)
        if img_x1 is None or img_x2 is None:
            return

        xmin, xmax = sorted([img_x1, img_x2])
        ymin, ymax = sorted([img_y1, img_y2])
        w = xmax - xmin
        h = ymax - ymin
        if w > 2 and h > 2:
            self.rois_temp.append((xmin, ymin, w, h))
        self.current_roi = None

    def finish_roi_selection(self, event=None):
        if not self.selecting_rois:
            return
        self.selecting_rois = False
        self.rois_search = self.rois_temp.copy()
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.root.unbind("<Return>")
        self.update_image_display()
        self.set_status(f"{len(self.rois_search)} ROI(s) enregistrée(s).")

    def select_fond(self):
        if self.image_display is None:
            messagebox.showerror("Erreur", "Image non prête.")
            return
        self.select_single_roi("fond")

    def select_mat(self):
        if self.image_display is None:
            messagebox.showerror("Erreur", "Image non prête.")
            return
        self.select_single_roi("mat")

    def select_single_roi(self, roi_type):
        self.selecting_single_roi = True
        self.roi_type = roi_type
        self.canvas.bind("<ButtonPress-1>", self.start_single_roi)
        self.canvas.bind("<B1-Motion>", self.draw_single_roi)
        self.canvas.bind("<ButtonRelease-1>", self.end_single_roi)
        self.root.bind("<Return>", self.finish_single_roi)
        label = "fond" if roi_type == "fond" else "matériau"
        self.set_status(f"Sélection de la zone {label} active. Trace la zone puis Entrée.")

    def start_single_roi(self, event):
        if not self.selecting_single_roi:
            return
        self.roi_start = (event.x, event.y)
        self.current_roi = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#4da6ff", width=2)

    def draw_single_roi(self, event):
        if not self.selecting_single_roi or self.current_roi is None:
            return
        self.canvas.coords(self.current_roi, self.roi_start[0], self.roi_start[1], event.x, event.y)

    def end_single_roi(self, event):
        return

    def finish_single_roi(self, event=None):
        if not self.selecting_single_roi or self.current_roi is None:
            return

        x1, y1, x2, y2 = self.canvas.coords(self.current_roi)
        img_x1, img_y1 = self.canvas_to_base_coords(x1, y1)
        img_x2, img_y2 = self.canvas_to_base_coords(x2, y2)
        if img_x1 is None or img_x2 is None:
            return

        xmin, xmax = sorted([img_x1, img_x2])
        ymin, ymax = sorted([img_y1, img_y2])
        w = xmax - xmin
        h = ymax - ymin

        if w > 2 and h > 2:
            if self.roi_type == "fond":
                self.roi_fond_coords = (xmin, ymin, w, h)
            elif self.roi_type == "mat":
                self.roi_mat_coords = (xmin, ymin, w, h)

        self.selecting_single_roi = False
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.root.unbind("<Return>")
        self.canvas.delete(self.current_roi)
        self.current_roi = None
        self.update_image_display()
        self.set_status("ROI fond / matériau enregistrée.")

    # ---------------------------
    # Bars detection / editing
    # ---------------------------
    def on_threshold_change(self, _val=None):
        self.threshold_value = int(round(self.threshold_slider.get()))
        self.min_area_value = int(round(self.min_area_slider.get()))
        self.erosion_value = int(round(self.erosion_slider.get()))
        self.threshold_value_label.configure(text=str(self.threshold_value))
        self.min_area_value_label.configure(text=str(self.min_area_value))
        self.erosion_value_label.configure(text=str(self.erosion_value))

        if self.current_step != 4:
            return

        if self.resegment_target_index is not None:
            self.preview_resegment_single_bar()
        else:
            self.detect_bars()
            self.update_bars_display()

    def detect_bars(self):
        if self.image_base_eq is None:
            return
        all_bars = []

        for roi_coords in self.rois_search:
            x, y, w, h = roi_coords
            roi = self.image_base_eq[y:y + h, x:x + w]
            if roi.size == 0:
                continue

            bars_local = detect_bar_rois(
                roi,
                threshold=self.threshold_value,
                min_area=self.min_area_value,
                erosion_iter=self.erosion_value,
            )
            shifted_bars = [cnt + np.array([[x, y]]) for cnt in bars_local]
            all_bars.extend(shifted_bars)

        self.bars = self.merge_connected_bars(all_bars, pad=2)
        self.numbers = list(range(1, len(self.bars) + 1))

    def display_bars_for_edit(self):
        if self.image_display is None:
            return
        self.update_bars_display()
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.initial_state = (self.bars.copy(), self.numbers.copy())

    def update_bars_display(self):
        if self.image_base_display is None:
            return
        img_full = draw_rois_with_numbers(self.image_base_display, self.bars, self.numbers)
        self.show_zoomed_overlay(img_full)

    def set_edit_mode(self, mode):
        self.edit_mode = mode
        texts = {
            "split": "Mode séparation verticale actif.",
            "roi": "Mode re-segmentation actif.",
            "number": "Mode modification des numéros actif.",
            "delete": "Mode suppression actif.",
        }
        self.set_status(texts.get(mode, f"Mode {mode}"))

    def on_canvas_click(self, event):
        x_click, y_click = self.canvas_to_base_coords(event.x, event.y)
        if x_click is None:
            return

        for i, cnt in enumerate(self.bars):
            x, y, w, h = cv2.boundingRect(cnt)
            if x <= x_click <= x + w and y <= y_click <= y + h:
                if self.edit_mode == "roi" and self.resegment_target_index == i:
                    self.history_stack.append((self.bars.copy(), self.numbers.copy()))
                    self.apply_resegment_single_bar(i)
                    self.resegment_target_index = None
                    self.update_bars_display()
                    return

                self.history_stack.append((self.bars.copy(), self.numbers.copy()))

                if self.edit_mode == "number":
                    self.modify_number(i)
                elif self.edit_mode == "roi":
                    self.resegment_roi(i)
                    return
                elif self.edit_mode == "split":
                    self.split_bar_vertical(i, x_click)
                elif self.edit_mode == "delete":
                    self.delete_roi(i)

                self.update_bars_display()
                return

    def modify_number(self, i):
        new_num = simpledialog.askinteger(
            "Modifier numéro",
            f"Nouveau numéro pour la barre {self.numbers[i]} :",
            initialvalue=self.numbers[i]
        )
        if new_num is not None:
            self.numbers[i] = int(new_num)
            self.set_status(f"Numéro modifié en {new_num}.")
        else:
            if self.history_stack:
                self.history_stack.pop()

    def resegment_roi(self, i):
        self.resegment_target_index = i
        self.set_status(f"Ajuste les curseurs puis reclique sur la ROI {self.numbers[i]} pour valider.")
        self.preview_resegment_single_bar()

    def preview_resegment_single_bar(self):
        if self.resegment_target_index is None:
            return
        if self.image_base_display is None or self.image_base_eq is None:
            return

        idx = self.resegment_target_index
        x, y, w, h = cv2.boundingRect(self.bars[idx])
        roi = self.image_base_eq[y:y + h, x:x + w]
        img_full = cv2.cvtColor(self.image_base_display.copy(), cv2.COLOR_GRAY2BGR)

        if roi.size != 0:
            bars_local = detect_bar_rois(
                roi,
                threshold=self.threshold_value,
                min_area=self.min_area_value,
                erosion_iter=self.erosion_value,
            )
            bars_local = [cnt + np.array([[x, y]]) for cnt in bars_local]
            for cnt in bars_local:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                cv2.rectangle(img_full, (bx, by), (bx + bw, by + bh), (255, 255, 0), 2)

        ox, oy, ow, oh = cv2.boundingRect(self.bars[idx])
        cv2.rectangle(img_full, (ox, oy), (ox + ow, oy + oh), (0, 0, 255), 1)
        self.show_zoomed_overlay(img_full)

    def apply_resegment_single_bar(self, i):
        x, y, w, h = cv2.boundingRect(self.bars[i])
        roi = self.image_base_eq[y:y + h, x:x + w]
        updated_bars = []

        if roi.size != 0:
            bars_local = detect_bar_rois(
                roi,
                threshold=self.threshold_value,
                min_area=self.min_area_value,
                erosion_iter=self.erosion_value,
            )
            updated_bars = [cnt + np.array([[x, y]]) for cnt in bars_local]

        if updated_bars:
            del self.bars[i]
            del self.numbers[i]
            for new_cnt in updated_bars:
                self.bars.append(new_cnt)
            self.numbers = list(range(1, len(self.bars) + 1))
            self.set_status(f"ROI re-segmentée avec {len(updated_bars)} box finale(s).")
        else:
            if self.history_stack:
                self.history_stack.pop()
            self.set_status("Aucune nouvelle barre détectée.")

    def delete_roi(self, i):
        del self.bars[i]
        del self.numbers[i]
        self.numbers = list(range(1, len(self.bars) + 1))
        self.set_status("ROI supprimée.")

    def split_bar_vertical(self, i, x_split):
        x, y, w, h = cv2.boundingRect(self.bars[i])
        if x_split <= x + 2 or x_split >= x + w - 2:
            self.set_status("Séparation trop proche du bord.")
            if self.history_stack:
                self.history_stack.pop()
            return

        local_split = x_split - x
        if local_split < 3 or local_split > w - 3:
            self.set_status("Séparation trop proche du bord.")
            if self.history_stack:
                self.history_stack.pop()
            return

        left_cnt = self.rect_to_contour(x, y, local_split, h)
        right_cnt = self.rect_to_contour(x + local_split, y, w - local_split, h)
        del self.bars[i]
        del self.numbers[i]
        self.bars.append(left_cnt)
        self.bars.append(right_cnt)
        self.numbers = list(range(1, len(self.bars) + 1))
        self.set_status("Séparation verticale appliquée.")

    def undo_edit(self):
        if self.history_stack:
            self.bars, self.numbers = self.history_stack.pop()
            self.set_status("Dernière modification annulée.")
            self.update_bars_display()

    def reset_edit(self):
        if self.initial_state is not None:
            self.bars = self.initial_state[0].copy()
            self.numbers = self.initial_state[1].copy()
            self.history_stack.clear()
            self.set_status("État initial restauré.")
            self.update_bars_display()

    def rects_overlap_or_close(self, r1, r2, pad=2):
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        ax1, ay1, ax2, ay2 = x1 - pad, y1 - pad, x1 + w1 + pad, y1 + h1 + pad
        bx1, by1, bx2, by2 = x2 - pad, y2 - pad, x2 + w2 + pad, y2 + h2 + pad
        return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

    def rect_to_contour(self, x, y, w, h):
        return np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], dtype=np.int32)

    def group_connected_bars(self, bars, pad=2):
        rects = [cv2.boundingRect(cnt) for cnt in bars]
        n = len(rects)
        visited = [False] * n
        groups = []

        for i in range(n):
            if visited[i]:
                continue
            stack = [i]
            visited[i] = True
            group = []
            while stack:
                cur = stack.pop()
                group.append(cur)
                for j in range(n):
                    if not visited[j] and self.rects_overlap_or_close(rects[cur], rects[j], pad=pad):
                        visited[j] = True
                        stack.append(j)
            groups.append(group)
        return groups

    def merge_connected_bars(self, bars, pad=2):
        if not bars:
            return []
        groups = self.group_connected_bars(bars, pad=pad)
        rects = [cv2.boundingRect(cnt) for cnt in bars]
        merged = []

        for g in groups:
            xs, ys, xe, ye = [], [], [], []
            for idx in g:
                x, y, w, h = rects[idx]
                xs.append(x)
                ys.append(y)
                xe.append(x + w)
                ye.append(y + h)
            x0, y0, x1, y1 = min(xs), min(ys), max(xe), max(ye)
            merged.append(self.rect_to_contour(x0, y0, x1 - x0, y1 - y0))
        return merged

    def sort_bars_reading_order(self, bars, order_mode="top_right_to_bottom_left", x_tol=12):
        rects = [cv2.boundingRect(cnt) for cnt in bars]
        items = [{"idx": i, "x": x, "y": y, "w": w, "h": h} for i, (x, y, w, h) in enumerate(rects)]
        items_sorted_x = sorted(items, key=lambda d: d["x"])
        columns = []

        for item in items_sorted_x:
            placed = False
            for col in columns:
                mean_x = np.mean([c["x"] for c in col])
                if abs(item["x"] - mean_x) <= x_tol:
                    col.append(item)
                    placed = True
                    break
            if not placed:
                columns.append([item])

        if order_mode in ("top_left_to_bottom_right", "bottom_left_to_top_right"):
            columns = sorted(columns, key=lambda col: np.mean([c["x"] for c in col]))
        else:
            columns = sorted(columns, key=lambda col: np.mean([c["x"] for c in col]), reverse=True)

        sorted_indices = []
        for col in columns:
            if order_mode in ("top_left_to_bottom_right", "top_right_to_bottom_left"):
                col_sorted = sorted(col, key=lambda d: d["y"])
            else:
                col_sorted = sorted(col, key=lambda d: d["y"], reverse=True)
            sorted_indices.extend([d["idx"] for d in col_sorted])

        return sorted_indices

    def renumber_current_bars(self, order_mode="top_right_to_bottom_left"):
        if not self.bars:
            self.numbers = []
            return
        sorted_indices = self.sort_bars_reading_order(self.bars, order_mode=order_mode)
        new_numbers = [0] * len(self.bars)
        for num, idx in enumerate(sorted_indices, start=1):
            new_numbers[idx] = num
        self.numbers = new_numbers

    def auto_number(self):
        if not self.bars:
            return
        self.history_stack.append((self.bars.copy(), self.numbers.copy()))
        self.renumber_current_bars(order_mode=self.auto_number_order)
        self.set_status("Numérotation automatique appliquée.")
        self.update_bars_display()

    def toggle_auto_order(self):
        orders = [
            "top_right_to_bottom_left",
            "top_left_to_bottom_right",
            "bottom_right_to_top_left",
            "bottom_left_to_top_right",
        ]
        labels = {
            "top_right_to_bottom_left": "Ordre auto : haut-droite → bas-gauche",
            "top_left_to_bottom_right": "Ordre auto : haut-gauche → bas-droite",
            "bottom_right_to_top_left": "Ordre auto : bas-droite → haut-gauche",
            "bottom_left_to_top_right": "Ordre auto : bas-gauche → haut-droite",
        }
        current_idx = orders.index(self.auto_number_order)
        next_idx = (current_idx + 1) % len(orders)
        self.auto_number_order = orders[next_idx]
        if self.bars:
            self.renumber_current_bars(order_mode=self.auto_number_order)
            self.update_bars_display()
        self.set_status(labels[self.auto_number_order])

    # ---------------------------
    # FTM / export
    # ---------------------------
    def compute_ftm(self):
        try:
            if self.image_base_rotated is None:
                raise ValueError("Image tournée absente.")
            if self.roi_fond_coords is None:
                raise ValueError("Zone fond non sélectionnée.")
            if self.roi_mat_coords is None:
                raise ValueError("Zone matériau non sélectionnée.")
            if len(self.bars) == 0:
                raise ValueError("Aucune barre détectée.")

            x, y, w, h = self.roi_fond_coords
            roi_fond = self.image_base_rotated[y:y + h, x:x + w]
            x, y, w, h = self.roi_mat_coords
            roi_mat = self.image_base_rotated[y:y + h, x:x + w]

            if roi_fond.size == 0 or roi_mat.size == 0:
                raise ValueError("ROI fond ou matériau vide.")

            sigma_fond_global = float(np.std(roi_fond))
            ct_espace_global = float(np.mean(roi_fond))
            ct_mat_global = float(np.mean(roi_mat))

            bar_rois = extract_bar_rois(self.image_base_rotated, self.bars)
            sorted_bar_info = sorted(zip(self.numbers, bar_rois), key=lambda x: x[0])
            sorted_numbers = [pair[0] for pair in sorted_bar_info]
            sorted_rois = [pair[1] for pair in sorted_bar_info]

            frequences_lpmm = [
                0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1,
                1.25, 1.4, 1.6, 1.8, 2.0, 2.24,
                2.5, 2.8, 3.15, 3.55, 4.0, 4.5,
                5.0, 5.6,
            ]

            if len(sorted_rois) == 0:
                raise ValueError("Aucune ROI barre exploitable.")

            frequences_detectees = np.array(frequences_lpmm[:len(sorted_rois)], dtype=np.float64)
            ftm_values = []
            for roi in sorted_rois:
                ftm_values.append(
                    compute_ftm_reglementaire(roi, sigma_fond_global, ct_espace_global, ct_mat_global)
                )

            ftm_values = np.array(ftm_values, dtype=np.float64)
            valid = np.isfinite(ftm_values) & np.isfinite(frequences_detectees)
            ftm_values = ftm_values[valid]
            frequences_detectees = frequences_detectees[valid]

            if len(ftm_values) < 2:
                raise ValueError("Pas assez de points pour ajuster la courbe.")

            params, _ = curve_fit(exp_func, frequences_detectees, ftm_values, p0=(1.0, -0.5), maxfev=10000)
            a_exp = float(params[0])
            b_exp = float(params[1])

            ftm50_freq = compute_ftm_threshold_frequency(a_exp, b_exp, 0.5)
            ftm10_freq = compute_ftm_threshold_frequency(a_exp, b_exp, 0.1)
            x_fit = np.linspace(float(np.min(frequences_detectees)), float(np.max(frequences_detectees)), 200)
            y_fit = exp_func(x_fit, *params)

            plt.figure(figsize=(10, 6))
            plt.plot(frequences_detectees, ftm_values, marker="o", linestyle="-", label="FTM mesurée")
            plt.plot(x_fit, y_fit, linestyle="--", label="Ajustement exponentiel")
            formula_str = f"$y = {params[0]:.3g} \\cdot e^{{{params[1]:.3g} x}}$"
            plt.text(0.95, 0.05, formula_str, transform=plt.gca().transAxes, fontsize=10,
                     verticalalignment="bottom", horizontalalignment="right",
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            plt.title("FTM en fonction de la fréquence spatiale (LP/mm)")
            plt.xlabel("Fréquence spatiale (LP/mm)")
            plt.ylabel("FTM réglementaire")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            folder_path = os.path.dirname(self.image_path) if self.image_path else "."
            graph_path = os.path.join(folder_path, "ftm_vs_frequence.png")
            xlsx_path = os.path.join(folder_path, "ftm_results.xlsx")
            annotated_path = os.path.join(folder_path, "tor18_segmented.png")

            plt.savefig(graph_path, dpi=200)
            plt.close()

            df = pd.DataFrame({
                "Groupe": sorted_numbers[:len(frequences_detectees)],
                "Résolution spatiale (lp/mm)": frequences_detectees,
                "FTM mesurée": ftm_values,
                "FTM exponentielle": exp_func(frequences_detectees, *params),
            })
            df.to_excel(xlsx_path, index=False)

            annotated_img = draw_rois_with_numbers(self.image_base_display, self.bars, self.numbers)
            annotated_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR) if annotated_img.ndim == 3 else annotated_img
            cv2.imwrite(annotated_path, annotated_bgr)

            self.last_results = {
                "folder_path": folder_path,
                "graph_image_path": graph_path,
                "xlsx_path": xlsx_path,
                "annotated_image_path": annotated_path,
                "image_name": self.image_path,
                "angle_deg": self.angle,
                "wl": self.wl,
                "ww": self.ww,
                "sigma_fond": sigma_fond_global,
                "ct_espace": ct_espace_global,
                "ct_mat": ct_mat_global,
                "exp_a": a_exp,
                "exp_b": b_exp,
                "ftm50": ftm50_freq,
                "ftm10": ftm10_freq,
            }
            self.results_ready = True
            self.show_results_preview()
            self.set_status("Calcul FTM terminé. Tu peux enregistrer le PDF.")
        except Exception as e:
            messagebox.showerror("Erreur calcul FTM", str(e))

    def show_results_preview(self):
        if not self.last_results:
            return
        graph_path = self.last_results["graph_image_path"]
        if os.path.exists(graph_path):
            img = cv2.imread(graph_path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.show_image_on_canvas(img)

    def save_pdf_report(self):
        if not self.results_ready or not self.last_results:
            messagebox.showerror("Erreur", "Aucun résultat disponible pour générer le PDF.")
            return

        folder_path = self.last_results["folder_path"]
        pdf_path = os.path.join(folder_path, DEFAULT_PDF_FILENAME)
        report_title = self.report_title_var.get().strip() or DEFAULT_REPORT_TITLE

        try:
            generate_pdf_report(
                pdf_path=pdf_path,
                annotated_image_path=self.last_results["annotated_image_path"],
                graph_image_path=self.last_results["graph_image_path"],
                image_name=self.last_results["image_name"],
                angle_deg=self.last_results["angle_deg"],
                wl=self.last_results["wl"],
                ww=self.last_results["ww"],
                sigma_fond=self.last_results["sigma_fond"],
                ct_espace=self.last_results["ct_espace"],
                ct_mat=self.last_results["ct_mat"],
                exp_a=self.last_results["exp_a"],
                exp_b=self.last_results["exp_b"],
                ftm50=self.last_results["ftm50"],
                ftm10=self.last_results["ftm10"],
                report_title=report_title,
            )
            self.set_status(f"PDF enregistré : {pdf_path}")
            messagebox.showinfo("PDF enregistré", f"Rapport PDF enregistré avec succès :\n{pdf_path}")
        except Exception as e:
            messagebox.showerror("Erreur PDF", str(e))

    def restart_analysis(self):
        self.current_step = 0
        self.image_path = None
        self.image = None
        self.image_rotated = None
        self.image_display = None
        self.image_eq = None
        self.image_base_rotated = None
        self.image_base_display = None
        self.image_base_eq = None

        self.zoom_factor = 1.0
        self.view_x0 = 0.0
        self.view_y0 = 0.0
        self.view_w = None
        self.view_h = None
        self.wl = 0
        self.ww = 1
        self.angle = 0

        self.rois_search = []
        self.roi_fond_coords = None
        self.roi_mat_coords = None
        self.rois_temp = []
        self.bars = []
        self.numbers = []
        self.threshold_value = 110
        self.min_area_value = 5
        self.erosion_value = 0
        self.resegment_target_index = None
        self.threshold_preview_mode = False
        self.history_stack = []
        self.initial_state = None
        self.edit_mode = "number"
        self.selecting_rois = False
        self.selecting_single_roi = False
        self.roi_type = None
        self.results_ready = False
        self.last_results = {}
        self.report_title_var.set(DEFAULT_REPORT_TITLE)

        self.threshold_slider.set(self.threshold_value)
        self.min_area_slider.set(self.min_area_value)
        self.erosion_slider.set(self.erosion_value)
        self.update_sliders()
        self.canvas.delete("all")
        self.update_step()
        self.set_status("Nouvelle analyse prête.")

    # ---------------------------
    # Navigation
    # ---------------------------
    def next_step(self):
        if self.current_step == 0:
            if self.image is None:
                messagebox.showerror("Erreur", "Sélectionne une image d'abord.")
                return
            self.rebuild_base_images()
            self.update_image_display()

        elif self.current_step == 1:
            self.rebuild_base_images()
            self.update_image_display()

        elif self.current_step == 2:
            if len(self.rois_search) == 0:
                messagebox.showerror("Erreur", "Sélectionne les ROIs de recherche.")
                return
            self.rebuild_base_images()
            self.image_display = self.build_working_image()
            self.detect_bars()
            if len(self.bars) == 0:
                messagebox.showwarning("Attention", "Aucune barre détectée dans les ROIs sélectionnées.")

        elif self.current_step == 3:
            if self.roi_fond_coords is None or self.roi_mat_coords is None:
                messagebox.showerror("Erreur", "Sélectionne les zones fond et matériau.")
                return

        self.current_step += 1
        if self.current_step < len(self.steps):
            self.update_step()


if __name__ == "__main__":
    root = ctk.CTk()
    app = MTFApplication(root)
    root.mainloop()
