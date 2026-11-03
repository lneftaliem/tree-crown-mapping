"""
install_final_figure.py -- replaces one figure in FINAL_FIGURES_300DPI with a regenerated version. The replaced
png/pdf are moved, not deleted, to _ARCHIVE_superseded_20261102/tree_crowns_v1_20261101/FINAL_FIGURES_300DPI/<archive_dir>/
(with the previous contact sheet); the manifest row is updated and the contact sheet rebuilt (layout of
assemble_final_figures.py).
Usage: python install_final_figure.py <folder> <file stem> <new source stem> <status> <archive_dir>
"""
import os, sys, shutil, pandas as pd
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
D, NAME, NEW, STATUS, ARC = sys.argv[1:6]
O = 'FINAL_FIGURES_300DPI'
SUP = f'../_ARCHIVE_superseded_20261102/tree_crowns_v1_20261101/{O}/{ARC}'
assert not os.path.exists(SUP), 'refusing to overwrite archived files'
os.makedirs(SUP)
for ext in ('png', 'pdf'):
    if os.path.exists(f'{O}/{D}/{NAME}.{ext}'):
        shutil.move(f'{O}/{D}/{NAME}.{ext}', f'{SUP}/{NAME}.{ext}')
    if os.path.exists(f'{NEW}.{ext}'):
        shutil.copy(f'{NEW}.{ext}', f'{O}/{D}/{NAME}.{ext}')
shutil.copy(f'{O}/CONTACT_SHEET_300DPI.pdf', f'{SUP}/CONTACT_SHEET_300DPI_before.pdf')
M = pd.read_csv(f'{O}/figure_manifest.csv', dtype=str)
i = M.index[M.file == f'{NAME}.png'][0]
im = Image.open(f'{O}/{D}/{NAME}.png'); w, h = im.size; dpi = round(im.info['dpi'][0])
M.loc[i, ['vector_pdf', 'width_px', 'height_px', 'dpi_metadata', 'print_size_in', 'effective_dpi_at_180mm_width', 'status', 'source']] = [
    str(os.path.exists(f'{O}/{D}/{NAME}.pdf')), str(w), str(h), str(dpi), f'{w / dpi:.2f} x {h / dpi:.2f}', str(round(w / 7.09)), STATUS, NEW]
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
