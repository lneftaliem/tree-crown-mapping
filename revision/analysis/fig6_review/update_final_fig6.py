"""
update_final_fig6.py -- puts the regenerated Figure 6 (selected windows; outputs/runs/v1b_figs300_fig6_selected) into
FINAL_FIGURES_300DPI. Nothing is deleted: the previous Fig6.{png,pdf} and contact sheet are moved to
FINAL_FIGURES_300DPI/superseded/. The manifest row for Fig6 is updated and the contact sheet rebuilt from the manifest
(same layout as assemble_final_figures.py).
"""
import os, shutil, pandas as pd
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
O, NEW, SUP = 'FINAL_FIGURES_300DPI', 'outputs/runs/v1b_figs300_fig6_selected/Fig6_example_tiles', 'FINAL_FIGURES_300DPI/superseded'
if not os.path.exists(SUP):  # (re-runs skip the moves; superseded files are never overwritten)
    os.makedirs(SUP)
    for ext in ('png', 'pdf'):
        shutil.move(f'{O}/1_Main/Fig6.{ext}', f'{SUP}/Fig6_previous_windows.{ext}')
        shutil.copy(f'{NEW}.{ext}', f'{O}/1_Main/Fig6.{ext}')
    shutil.move(f'{O}/CONTACT_SHEET_300DPI.pdf', f'{SUP}/CONTACT_SHEET_300DPI_previous.pdf')
M = pd.read_csv(f'{O}/figure_manifest.csv', dtype=str)
i = M.index[M.file == 'Fig6.png'][0]
im = Image.open(f'{O}/1_Main/Fig6.png'); w, h = im.size; dpi = round(im.info['dpi'][0])
M.loc[i, ['width_px', 'height_px', 'dpi_metadata', 'print_size_in', 'effective_dpi_at_180mm_width', 'status', 'source']] = [
    str(w), str(h), str(dpi), f'{w / dpi:.2f} x {h / dpi:.2f}', str(round(w / 7.09)),
    'REGENERATED from plotting script (v1b_ndvi02); windows selected by authors 2026-11-02 (NDVI only)', NEW]
M.to_csv(f'{O}/figure_manifest.csv', index=False)
rows = M.to_dict('records')
with PdfPages(f'{O}/CONTACT_SHEET_300DPI.pdf') as pdf:
    for k in range(0, len(rows), 6):
        fig, ax = plt.subplots(2, 3, figsize=(16.5, 11)); fig.suptitle('Final figures (300 DPI): overview' + (' (cont.)' if k else ''), fontsize=16, fontweight='bold')
        for a, r in zip(ax.flat, rows[k:k + 6] + [None] * 6):
            a.axis('off')
            if r:
                a.imshow(mpimg.imread(f'{O}/{r["folder"]}/{r["file"]}')); a.set_title(f'{r["file"][:-4]}\n{r["status"].split(" (")[0]}', fontsize=9)
        pdf.savefig(fig); plt.close(fig)
    for r in rows:
        fig = plt.figure(figsize=(16.5, 11))
        fig.suptitle(r['file'][:-4].replace('_RETAINED_original', '  (retained from original manuscript)'), fontsize=16, fontweight='bold', y=0.98)
        fig.text(0.5, 0.935, f'{r["width_px"]} x {r["height_px"]} px, {r["dpi_metadata"]} dpi | {r["status"]}', ha='center', fontsize=10, color='#444')
        a = fig.add_axes([0.03, 0.02, 0.94, 0.89]); a.imshow(mpimg.imread(f'{O}/{r["folder"]}/{r["file"]}')); a.axis('off')
        pdf.savefig(fig, dpi=150); plt.close(fig)
print(M.loc[i].to_dict())
