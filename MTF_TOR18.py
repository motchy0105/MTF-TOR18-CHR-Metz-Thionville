import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd

def on_key(event):
    global edit_mode, bars, numbers, last_modified_text, history_stack, initial_state

    if event.key == 'm':
        edit_mode = "number"
        last_modified_text = "Mode : modification des numéros (appuyez sur une barre pour modifier)"
    elif event.key == 'r':
        edit_mode = "roi"
        last_modified_text = "Mode : re-segmentation des ROI (appuyez sur une barre pour re-segmenter)"
    elif event.key == 'd':  # Nouveau mode "delete"
        edit_mode = "delete"
        last_modified_text = "Mode : suppression (appuyez sur une barre pour la supprimer)"
    elif event.key == 'z':
        if history_stack:
            bars, numbers = history_stack.pop()
            last_modified_text = "✅ Dernière modification annulée."
            redraw()
        else:
            last_modified_text = "⚠️ Aucune modification à annuler."
    elif event.key == 'u':
        if initial_state:
            bars, numbers = initial_state[0].copy(), initial_state[1].copy()
            history_stack.clear()
            last_modified_text = "↩️ Toutes les modifications ont été annulées (état initial restauré)."
            redraw()
        else:
            last_modified_text = "⚠️ État initial non disponible (vérifie sa définition)."
    elif event.key == 'a':  # Nouvelle fonctionnalité pour réinitialiser les numéros
        numbers = list(range(1, len(bars)+1))
        history_stack.append((bars.copy(), numbers.copy()))
        last_modified_text = "🔢 Numérotation automatique réinitialisée (1 à N)."
        redraw()
    redraw()

def redraw():
    ax.clear()
    sorted_numbers, sorted_bars = numbers, bars  # garder l'ordre zigzag
    img_disp = draw_rois_with_numbers(image, sorted_bars, sorted_numbers)
    ax.imshow(img_disp, cmap='gray')
    title = "'m' = modification, 'r' = seuillage, 'd' = suppression, 'z'= retour, 'u' = état initial, 'a' = réinit numéros"
    if last_modified_text:
        title += f"\n{last_modified_text}"
    ax.set_title(title)
    ax.axis("off")
    status_text_obj.set_text(last_modified_text)
    fig.canvas.draw_idle()

def resegment_single_roi(image_eq, roi_coords):
    x, y, w, h = roi_coords
    roi = image_eq[y:y+h, x:x+w]

    window_name = "Redétection dans ROI"
    cv2.namedWindow(window_name)

    def nothing(x):
        pass

    cv2.createTrackbar('Seuil', window_name, 110, 255, nothing)

    while True:
        thresh_val = cv2.getTrackbarPos('Seuil', window_name)
        roi_disp = cv2.cvtColor(roi.copy(), cv2.COLOR_GRAY2BGR)
        bars_local = detect_bar_rois(roi, threshold=thresh_val)

        for cnt in bars_local:
            x1, y1, w1, h1 = cv2.boundingRect(cnt)
            cv2.rectangle(roi_disp, (x1, y1), (x1+w1, y1+h1), (0, 0, 255), 1)

        cv2.imshow(window_name, roi_disp)
        key = cv2.waitKey(30) & 0xFF
        if key == 13 or key == 27:
            break

    cv2.destroyWindow(window_name)

    # Décaler les nouveaux contours aux coordonnées absolues
    shifted_bars = [cnt + np.array([[x, y]]) for cnt in bars_local]
    return shifted_bars

def on_click(event):
    global last_modified_text, bars, numbers, history_stack
    if event.inaxes != ax:
        return
    x_click, y_click = int(event.xdata), int(event.ydata)

    for i, cnt in enumerate(bars):
        x, y, w, h = cv2.boundingRect(cnt)
        if x <= x_click <= x + w and y <= y_click <= y + h:
            # 🔁 Sauvegarde avant modif
            history_stack.append((bars.copy(), numbers.copy()))

            if edit_mode == "number":
                try:
                    new_num = input(f"Entrez le nouveau numéro pour la barre (actuel {numbers[i]}): ")
                    new_num_int = int(new_num)
                    numbers[i] = new_num_int
                    last_modified_text = f"Numéro de la barre {i+1} modifié en {new_num_int}"
                except ValueError:
                    last_modified_text = "Entrée invalide, numéro non modifié."
                    history_stack.pop()  # annule sauvegarde inutile
                redraw()
            elif edit_mode == "roi":
                print(f"🔍 Modification de la ROI {numbers[i]} sélectionnée.")
                roi_coords = (x, y, w, h)
                updated_bars = resegment_single_roi(image_eq, roi_coords)
                if updated_bars:
                    del bars[i]
                    del numbers[i]
                    for new_cnt in updated_bars:
                        bars.append(new_cnt)
                        numbers.append(max(numbers) + 1)
                    last_modified_text = f"ROI re-segmentée avec {len(updated_bars)} nouvelles barres."
                else:
                    last_modified_text = "Aucune nouvelle barre détectée."
                    history_stack.pop()
                redraw()
            elif edit_mode == "delete":  # Nouveau cas : suppression
                del bars[i]
                del numbers[i]
                last_modified_text = f"ROI {numbers[i]} supprimée."
                redraw()
            break

def exp_func(x, a, b):
    return a * np.exp(b * x)

def poly_to_latex(poly):
    coefs = poly.coefficients
    terms = []
    degree = len(coefs) - 1
    for i, c in enumerate(coefs):
        power = degree - i
        if abs(c) < 1e-8:
            continue
        # Format du coefficient (arrondi à 3 décimales)
        coef_str = f"{c:.3g}" if (abs(c - 1) > 1e-8 or power == 0) else ""
        # Signe
        sign = "+" if c > 0 else "-"
        # Terme variable
        if power > 1:
            term = f"{coef_str}x^{power}"
        elif power == 1:
            term = f"{coef_str}x"
        else:
            term = f"{coef_str}"
        terms.append(f" {sign} {term}")
    # Le premier terme ne doit pas avoir le signe + devant
    expr = "".join(terms).strip()
    if expr.startswith("+"):
        expr = expr[1:].strip()
    return "$y = " + expr + "$"
# 1. Charger une image PNG
def load_png_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint16:
        image = image.astype(np.uint16) *257




    return image

# 2. Égalisation de l’histogramme
def preprocess(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # Convertir en gris si nécessaire
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Vérifie le type, CLAHE accepte uint8 ou uint16 uniquement
    if image.dtype not in [np.uint8, np.uint16]:
        raise ValueError(f"Type d'image non supporté : {image.dtype}")
    return clahe.apply(image)
# 3. Détection des contours (barres)
def detect_bar_rois(image, threshold=110, min_area=300, erosion_iter=1):
    if image.dtype == np.uint16:
        thresh_16 = int(threshold * 256)
        _, binary_16 = cv2.threshold(image, thresh_16, 65535, cv2.THRESH_BINARY_INV)
        binary = (binary_16 / 256).astype(np.uint8)
    else:
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Erosion pour réduire la taille des contours (enlever bordures)
    kernel = np.ones((3,3), np.uint8)
    binary_eroded = cv2.erode(binary, kernel, iterations=erosion_iter)

    contours, _ = cv2.findContours(binary_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bars = [c for c in contours if cv2.contourArea(c) > min_area]
    bars = sorted(bars, key=lambda c: cv2.contourArea(c), reverse=True)
    return bars

# 4. Extraction des ROI barres
def extract_bar_rois(image, bars):
    rois = []
    for cnt in bars:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = image[y:y+h, x:x+w]
        rois.append(roi)
    return rois

# 5. Dessin des numéros sur les barres
def draw_rois_with_numbers(image, bars, numbers):
    # Normaliser l'image 16 bits en 8 bits
    if image.dtype == np.uint16:
        image_8bit = (image / 256).astype(np.uint8)  # Passage à 8 bits
    else:
        image_8bit = image

    image_rgb = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2BGR)
    font_scale = 0.4
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, cnt in enumerate(bars):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 0, 255), 1)
        text = str(numbers[i])
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cx = x + w // 2 - text_width // 2
        cy = y + h // 2 + text_height // 2
        cv2.putText(image_rgb, text, (cx, cy), font, font_scale, (0, 255, 0), thickness, lineType=cv2.LINE_AA)
    return image_rgb
# 6. FTM réglementaire mise à jour (utilise les valeurs globales du fond et du mat)
def compute_ftm_reglementaire(roi_groupe, sigma_fond, ct_espace, ct_mat):
    valid_pixels = roi_groupe[roi_groupe < 65535 ]
    sigma_groupe = np.std(valid_pixels)
    numerator = np.sqrt(np.abs(sigma_groupe**2 - sigma_fond**2))
    denominator = abs(ct_mat - ct_espace)
    if denominator == 0:
        return 0
    return (np.pi / np.sqrt(2)) * (numerator / denominator)

# 7. Interface pour ajuster dynamiquement le seuil
def interactive_threshold_selection(image_eq, rois_search):
    window_name = "Réglage du seuil de détection"
    cv2.namedWindow(window_name)

    def nothing(x):
        pass

    cv2.createTrackbar('Seuil', window_name, 110, 255, nothing)

    while True:
        thresh_val = cv2.getTrackbarPos('Seuil', window_name)
        display_img = cv2.cvtColor(image_eq.copy(), cv2.COLOR_GRAY2BGR)

        for roi_search in rois_search:
            roi = image_eq[int(roi_search[1]):int(roi_search[1]+roi_search[3]), int(roi_search[0]):int(roi_search[0]+roi_search[2])]
            bars_local = detect_bar_rois(roi, threshold=thresh_val)
            for cnt in bars_local:
                cnt_shifted = cnt + np.array([[roi_search[0], roi_search[1]]])
                x, y, w, h = cv2.boundingRect(cnt_shifted)
                cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 0, 255), 1)

        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(50) & 0xFF
        if key == 13 or key == 27:
            break

    cv2.destroyWindow(window_name)
    return thresh_val

# 8. Affichage zoomé de toutes les ROI barres avec leurs numéros
def show_cropped_rois_with_numbers(rois, numbers):
    pass  # Désactivé selon demande utilisateur
# Nouvelle fonction pour distribution en zigzag entre groupes
def update_numbers_zigzag(bar_counts_per_roi):
    max_len = max(bar_counts_per_roi)
    updated_numbers = []
    current_number = 1
    for i in range(max_len):
        for g in range(len(bar_counts_per_roi)):
            if i < bar_counts_per_roi[g]:
                updated_numbers.append(current_number)
                current_number += 1
    return updated_numbers

def halve_last_three_roi_widths():
    # Appliquer la réduction sur les 3 barres ayant les plus grands numéros
    if len(bars) < 3:
        return
    # Associer chaque barre à son numéro
    bar_info = list(zip(numbers, bars))
    # Trier par numéro croissant
    bar_info_sorted = sorted(bar_info, key=lambda x: x[0])
    # Réduire la largeur des 3 dernières (plus grands numéros)
    for i in range(-3, 0):
        num, cnt = bar_info_sorted[i]
        x, y, w, h = cv2.boundingRect(cnt)
        new_w = max(1, w // 2)
        x_center = x + w // 2
        x_new = x_center - new_w // 2
        new_cnt = np.array([[[x_new, y]], [[x_new + new_w, y]], [[x_new + new_w, y + h]], [[x_new, y + h]]])
        # Mettre à jour la barre dans la liste d'origine (par numéro)
        idx = numbers.index(num)
        bars[idx] = new_cnt

def on_click_wrapper(event):
    on_click(event)
    # Après modification, réduire la largeur des 3 dernières ROI
    halve_last_three_roi_widths()
    redraw()

def auto_crop_bars_to_min_width(debug_show=False):
    # Recadre chaque barre sur la zone utile (pixels noirs binarisés)
    global bars
    new_bars = []
    for idx, cnt in enumerate(bars):
        x, y, w, h = cv2.boundingRect(cnt)
        roi = image_eq[y:y+h, x:x+w]
        # Binarisation Otsu (barres noires sur fond blanc)
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Moyenne verticale pour robustesse
        proj = np.mean(binary, axis=0)
        # Seuil pour considérer une colonne comme "noire"
        min_val = np.min(proj)
        threshold = min_val + 0.1 * (np.max(proj) - min_val)
        black_cols = np.where(proj <= threshold)[0]
        if len(black_cols) > 0:
            x1 = black_cols[0]
            x2 = black_cols[-1]
            new_cnt = np.array([[[x + x1, y]], [[x + x2, y]], [[x + x2, y + h]], [[x + x1, y + h]]])
        else:
            new_cnt = cnt  # fallback : pas de recadrage
        new_bars.append(new_cnt)
    bars = new_bars

def sort_bars_spatially(bars, y_thresh=20):
    # Trie les barres d'abord par ligne (y), puis par colonne (x)
    # y_thresh : tolérance pour regrouper les barres sur la même ligne
    bar_centers = []
    for cnt in bars:
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2
        bar_centers.append((cnt, cx, cy))
    # Regrouper par lignes (y proche)
    bar_centers.sort(key=lambda tup: tup[2])  # tri par y
    lines = []
    for cnt, cx, cy in bar_centers:
        found = False
        for line in lines:
            if abs(line[0][2] - cy) < y_thresh:
                line.append((cnt, cx, cy))
                found = True
                break
        if not found:
            lines.append([(cnt, cx, cy)])
    # Trier chaque ligne par x
    for line in lines:
        line.sort(key=lambda tup: tup[1])
    # Aplatir
    sorted_bars = [tup[0] for line in lines for tup in line]
    return sorted_bars

def sort_bars_by_column(bars, x_thresh=50, debug=False):
    # Trie les barres par colonne (x), puis du haut vers le bas (y)
    bar_centers = []
    for cnt in bars:
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2
        bar_centers.append((cnt, cx, cy))
    # Regrouper par colonnes (x proche)
    bar_centers.sort(key=lambda tup: tup[1])  # tri par x
    columns = []
    for cnt, cx, cy in bar_centers:
        found = False
        for col in columns:
            if abs(col[0][1] - cx) < x_thresh:
                col.append((cnt, cx, cy))
                found = True
                break
        if not found:
            columns.append([(cnt, cx, cy)])
    if debug:
        print(f"Nombre de colonnes detectees : {len(columns)}")
        for i, col in enumerate(columns):
            print(f"Colonne {i+1} : {len(col)} barres")
    # Trier chaque colonne par y (haut vers bas)
    for col in columns:
        col.sort(key=lambda tup: tup[2])
    # Aplatir colonne par colonne
    sorted_bars = [tup[0] for col in columns for tup in col]
    return sorted_bars

def sort_bars_grid_columnwise(bars, y_thresh=30, debug=False):
    # 1. Regrouper en lignes (par y)
    bar_centers = []
    for cnt in bars:
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2
        bar_centers.append((cnt, cx, cy))
    bar_centers.sort(key=lambda tup: tup[2])  # tri par y
    lines = []
    for cnt, cx, cy in bar_centers:
        found = False
        for line in lines:
            if abs(line[0][2] - cy) < y_thresh:
                line.append((cnt, cx, cy))
                found = True
                break
        if not found:
            lines.append([(cnt, cx, cy)])
    # 2. Trier chaque ligne par x
    for line in lines:
        line.sort(key=lambda tup: tup[1])
    if debug:
        print(f"Nombre de lignes détectées : {len(lines)}")
        for i, line in enumerate(lines):
            print(f"Ligne {i+1} : {len(line)} barres")
    # 3. Reconstituer la grille (matrice [ligne][colonne])
    n_rows = len(lines)
    n_cols = max(len(line) for line in lines)
    # 4. Parcourir colonne par colonne
    sorted_bars = []
    for col in range(n_cols):
        for row in range(n_rows):
            if col < len(lines[row]):
                sorted_bars.append(lines[row][col][0])
    return sorted_bars

def sort_bars_grid_columnwise_mirrored(bars, y_thresh=30, debug=False):
    """Trie les barres de gauche à droite et de haut en bas"""
    bar_centers = []
    for cnt in bars:
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2
        bar_centers.append((cnt, cx, cy))

    # Étape 1 : grouper par colonnes (selon cx proche)
    bar_centers.sort(key=lambda tup: tup[1])  # tri par x croissant pour regroupement
    columns = []
    for cnt, cx, cy in bar_centers:
        found = False
        for col in columns:
            if abs(col[0][1] - cx) < y_thresh:  # même colonne si x proche
                col.append((cnt, cx, cy))
                found = True
                break
        if not found:
            columns.append([(cnt, cx, cy)])

    # Étape 2 : trier chaque colonne de haut en bas (y croissant)
    for col in columns:
        col.sort(key=lambda tup: tup[2])

    # Étape 3 : inverser l’ordre des colonnes (droite à gauche = miroir horizontal)
    columns = columns[::-1]

    if debug:
        print(f"Nombre de colonnes détectées : {len(columns)}")
        for i, col in enumerate(columns):
            print(f"Colonne {i+1}: {[ (b[1], b[2]) for b in col ]}")

    # Étape 4 : aplatir colonne par colonne (haut en bas)
    sorted_bars = []
    for col in columns:
        for tup in col:
            sorted_bars.append(tup[0])

    return sorted_bars





    # 2. Trier chaque ligne par x
    for line in lines:
        line.sort(key=lambda tup: tup[1])
    if debug:
        print(f"Nombre de lignes détectées : {len(lines)}")
        for i, line in enumerate(lines):
            print(f"Ligne {i+1} : {len(line)} barres")
    # 3. Reconstituer la grille (matrice [ligne][colonne])
    n_rows = len(lines)
    n_cols = max(len(line) for line in lines)
    # 4. Parcourir colonne par colonne, de droite à gauche (miroir)
    sorted_bars = []
    for col in reversed(range(n_cols)):
        for row in range(n_rows):
            if col < len(lines[row]):
                sorted_bars.append(lines[row][col][0])
    return sorted_bars

# 9. Chargement #MAIINNNNNN###################
root = tk.Tk()
root.withdraw()  # cache la fenêtre principale tkinter

# Ouvre une boîte de dialogue pour sélectionner une image (png, jpg...)
image_path = filedialog.askopenfilename(
    title="Sélectionnez une image",
    filetypes=[("Fichiers PNG", "*.png"), ("Fichiers JPG", "*.jpg;*.jpeg"), ("Tous fichiers", "*.*")]
)
if not image_path:
    print("Aucun fichier sélectionné. Fin du programme.")
    exit()

print(f"Image sélectionnée : {image_path}")
image = load_png_image(image_path)
image_eq = preprocess(image)

# Sélection manuelle de plusieurs régions de recherche des barres
print("Sélectionnez les RÉGIONS DE RECHERCHE DES BARRES (plusieurs possibles, appuyez sur Échap quand terminé)")
rois_search = cv2.selectROIs("Image : régions de recherche", image_eq)
cv2.destroyAllWindows()
#ajout d'une variable globale 
edit_mode = "number"  # ou "roi"
history_stack = []  # pour les annulations
initial_state = None  # pour le reset complet
# Choix interactif du seuil dans les régions sélectionnées
threshold_selected = interactive_threshold_selection(image_eq, rois_search)

# Sélection manuelle du fond
print("Sélectionnez la ZONE DU FOND")
roi_fond_coords = cv2.selectROI("Image : fond", image_eq)
roi_fond = image_eq[int(roi_fond_coords[1]):int(roi_fond_coords[1]+roi_fond_coords[3]), int(roi_fond_coords[0]):int(roi_fond_coords[0]+roi_fond_coords[2])]
cv2.destroyAllWindows()

# Sélection manuelle du matériau
print("Sélectionnez la ZONE DU MATERIAU")
roi_mat_coords = cv2.selectROI("Image : materiau", image_eq)
roi_mat = image_eq[int(roi_mat_coords[1]):int(roi_mat_coords[1]+roi_mat_coords[3]), int(roi_mat_coords[0]):int(roi_mat_coords[0]+roi_mat_coords[2])]
cv2.destroyAllWindows()

# Calcul des valeurs globales du fond et matériau
sigma_fond_global = np.std(roi_fond)
ct_espace_global = np.mean(roi_fond)
ct_mat_global = np.mean(roi_mat)

# Détection automatique des barres et numérotation zig-zag
bars = []
numbers = []
roi_index = 0

bar_counts_per_roi = []
bar_sets = []  # liste des barres par groupe

# 1. Détection dans chaque ROI
for roi_coords in rois_search:
    x, y, w, h = roi_coords
    roi = image_eq[y:y+h, x:x+w]
    bars_local = detect_bar_rois(roi, threshold=threshold_selected)
    shifted_bars = [cnt + np.array([[x, y]]) for cnt in bars_local]
    bar_sets.append(shifted_bars)

# 2. Récupération zigzagée des barres selon update_numbers_zigzag
bar_counts_per_roi = [len(group) for group in bar_sets]
numbers = update_numbers_zigzag(bar_counts_per_roi)

# 3. Réorganisation des barres selon ordre des numéros
bars = []
max_len = max(len(g) for g in bar_sets)
for i in range(max_len):
    for g in range(len(bar_sets)):
        if i < len(bar_sets[g]):
            bars.append(bar_sets[g][i])

numbers = update_numbers_zigzag(bar_counts_per_roi)

# Recadrage automatique de toutes les barres sur la zone utile (binarisation)
auto_crop_bars_to_min_width(debug_show=False)

# Tri grille colonne par colonne, effet miroir (haut-droite -> bas, puis colonne suivante à gauche)
bars = sort_bars_grid_columnwise_mirrored(bars, y_thresh=30, debug=True)
numbers = list(range(1, len(bars)+1))
# 🔁 Sauvegarde de l'état initial (pour annulation complète)
initial_state = (bars.copy(), numbers.copy())
# Interface utilisateur pour modifier les numéros
fig, ax = plt.subplots(figsize=(10, 10))
status_text_obj = ax.text(
    0.01, 0.99, "", transform=ax.transAxes,
    fontsize=10, verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
)
last_modified_text = ""

#####ùodification 
fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)  # Pour éviter conflits avec key handler
fig.canvas.mpl_connect('button_press_event', on_click)
redraw()
cid = fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key)   
plt.show()

# Extraction des barres et tri selon les numéros
bar_rois = extract_bar_rois(image_eq, bars)
ftm_values = []

sorted_bar_info = sorted(zip(numbers, bar_rois), key=lambda x: x[0])
sorted_numbers = [pair[0] for pair in sorted_bar_info]
sorted_rois = [pair[1] for pair in sorted_bar_info]

# Affichage global de toutes les barres recadrées sur l'image égalisée
fig, ax = plt.subplots(figsize=(10, 10))
img_disp = draw_rois_with_numbers(image_eq, bars, numbers)
ax.imshow(img_disp, cmap='gray')
ax.set_title("Toutes barres recadrées sur largeur utile (binarisation)")
ax.axis("off")
plt.show()

# show_cropped_rois_with_numbers(sorted_rois, sorted_numbers)  # Désactivé selon demande
# Liste des fréquences LP/mm du TOR 18 FG (à ajuster selon la mire réelle si besoin)
frequences_lpmm = [
    0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1,
    1.25, 1.4, 1.6, 1.8, 2.0, 2.24,
    2.5, 2.8, 3.15, 3.55, 4.0, 4.5,
    5.0, 5.6
]



for i, roi in enumerate(sorted_rois):
    ftm = compute_ftm_reglementaire(roi, sigma_fond_global, ct_espace_global, ct_mat_global)
    ftm_values.append(ftm)
    #print(f"FTM barre {sorted_numbers[i]} : {ftm:.4f}")
# Associer chaque barre triée à une fréquence
frequences_detectees = frequences_lpmm[:len(sorted_numbers)]

# ✅ Affichage du tableau FTM + fréquence + numéro
'''print("\n--- Résumé par groupe ---")
for num, freq, ftm in zip(sorted_numbers, frequences_detectees, ftm_values):
    print(f"Groupe {num} - Fréquence : {freq:.2f} LP/mm - FTM : {ftm:.4f}")'''

plt.figure(figsize=(10, 6))

# Ajustement expo sur la fréquence
params, _ = curve_fit(exp_func, frequences_detectees, ftm_values, p0=(1, -0.5))
x_fit = np.linspace(min(frequences_detectees), max(frequences_detectees), 200)
y_fit = exp_func(x_fit, *params)

plt.plot(frequences_detectees, ftm_values, marker='o', linestyle='-', label="FTM mesurée")
plt.plot(x_fit, y_fit, color='red', linestyle='--', label=f"Ajustement exponentiel")

# Afficher la fonction polynomiale en bas à droite (coordonnées relatives à l'axe)
formula_str = f"$y = {params[0]:.3g} \cdot e^{{{params[1]:.3g} x}}$"
plt.text(0.95, 0.05, formula_str, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='bottom', horizontalalignment='left',
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.title("FTM en fonction de la fréquence spatiale (LP/mm)")
plt.xlabel("Fréquence spatiale (LP/mm)")
plt.ylabel("FTM réglementaire")
plt.grid(True)
plt.legend()
plt.tight_layout()
# Sauvegarde de la figure
folder_path = os.path.dirname(image_path)  # dossier de l'image sélectionnée
image_filename = "ftm_vs_frequence.png"
save_path = os.path.join(folder_path, image_filename)

plt.savefig(save_path)
print(f"Figure sauvegardée sous : {save_path}")
plt.show()
# Écriture du fichier CSV
# Calcul des valeurs polynomiales
frequences_detectees = np.array(frequences_detectees)
ftm_poly_values = exp_func(frequences_detectees, *params)

# Récupération du dossier de l'image sélectionnée
folder_path = os.path.dirname(image_path)

# Création du DataFrame à partir des données calculées
data = {
    "Groupe": sorted_numbers[:len(frequences_detectees)],
    "Résolution spatiale (lp/mm)": frequences_detectees,
    "FTM mesurée": ftm_values,
    "FTM polynomiale": ftm_poly_values
}

df = pd.DataFrame(data)

# Nom et chemin du fichier Excel (dans le même dossier que l'image)
excel_filename = "ftm_results.xlsx"
excel_save_path = os.path.join(folder_path, excel_filename)

# Sauvegarde en Excel (sans index)
df.to_excel(excel_save_path, index=False)

print(f"Fichier Excel sauvegardé sous : {excel_save_path}")
