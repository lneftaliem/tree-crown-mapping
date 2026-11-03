"""
assemble_final_figures.py -- copies the final regenerated figures (and the retained, non-tree-derived originals) into
FINAL_FIGURES_300DPI/{1_Main,2_ExtendedData,3_Supplementary}, writes figure_manifest.csv (true pixel size, DPI metadata,
print size, status, source) and CONTACT_SHEET_300DPI.pdf. Retained figures are copied as extracted: never resampled.
"""
import os, shutil, pandas as pd
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
R = 'outputs/runs/v1b_figs300/analysis_output'; V = '../threshold_v1_20261101/final_0.7_figfix'; PUB = 'manuscript_original/figures_extracted'
O = 'FINAL_FIGURES_300DPI'
assert not os.path.exists(O), 'refusing to overwrite an existing figure folder'
for d in ('1_Main', '2_ExtendedData', '3_Supplementary'):
    os.makedirs(f'{O}/{d}')
G = 'REGENERATED from plotting script (v1b_ndvi02)'
VV = 'REGENERATED from plotting script (model validation, final pipeline tau = 0.7; statistics unchanged)'
U = 'RETAINED from original manuscript (not tree-derived); extracted image, not upscaled'
items = [('1_Main', 'Fig1', f'{R}/identical_tree_density_map', G), ('1_Main', 'Fig2', f'{R}/equity_results/combined_income_zscore_density_scatter', G),
         ('1_Main', 'Fig3', f'{R}/equity_results/combined_income_zscore_canopy_cover_scatter_corrected', G),
         ('1_Main', 'Fig4', f'{R}/equity_results/combined_income_zscore_crown_area_scatter_corrected', G),
         ('1_Main', 'Fig5', f'{PUB}/MainText/image5', U), ('1_Main', 'Fig6', 'outputs/runs/v1b_figs300_fig6/Fig6_example_tiles', G),
         ('2_ExtendedData', 'ED_Fig1', f'{R}/extended_data/ED_Fig1_dedup_sensitivity', G), ('2_ExtendedData', 'ED_Fig2', f'{R}/crown_area_histogram', G),
         ('2_ExtendedData', 'ED_Fig3', f'{R}/extended_data/ED_Fig3_spatial_model_sensitivity', G),
         ('2_ExtendedData', 'ED_Fig4', f'{R}/extended_data/ED_Fig4_adjusted_income_coefficients', G),
         ('2_ExtendedData', 'ED_Table1', 'outputs/runs/v1b_figs300_tables/ED_Table1', G),
         ('3_Supplementary', 'FigS01', f'{V}/figure_d1_size_detection_delineation', VV), ('3_Supplementary', 'FigS02', f'{V}/figure_d2_merging_canopy_density', VV),
         ('3_Supplementary', 'FigS03', f'{V}/figure_d3_representative_visual_cases', VV), ('3_Supplementary', 'FigS04', f'{PUB}/SupplementaryInformation/image5', U),
         ('3_Supplementary', 'FigS05', f'{R}/inventory_correlation/inventory_correlation_4panel', G),
         ('3_Supplementary', 'FigS06', f'{R}/ripley/FigS6_ripley_L_by_income_quartile', G),
         ('3_Supplementary', 'FigS07', f'{PUB}/SupplementaryInformation/image8', U), ('3_Supplementary', 'FigS08', f'{PUB}/SupplementaryInformation/image9', U),
         ('3_Supplementary', 'FigS09', f'{R}/redlining_figures/figure_1_holc_austin_tree_distribution', G),
         ('3_Supplementary', 'FigS10', f'{R}/redlining_figures/figure_2_polygon_tree_metrics_by_grade', G),
         ('3_Supplementary', 'FigS11', f'{R}/redlining_figures/figure_3_influence_sensitivity_analysis', G),
         ('3_Supplementary', 'FigS12', f'{R}/surrey_reviewer2_analysis/surrey_reviewer2_sensitivity_figure', G)]
rows = []
for d, name, src, st in items:
    fn = name + ('_RETAINED_original' if st == U else '')
    shutil.copy(f'{src}.png', f'{O}/{d}/{fn}.png')
    if os.path.exists(f'{src}.pdf'):
        shutil.copy(f'{src}.pdf', f'{O}/{d}/{fn}.pdf')
    im = Image.open(f'{O}/{d}/{fn}.png'); w, h = im.size; dpi = im.info.get('dpi'); dpi = round(dpi[0]) if dpi else None
    rows.append(dict(folder=d, file=f'{fn}.png', vector_pdf=os.path.exists(f'{O}/{d}/{fn}.pdf'), width_px=w, height_px=h,
                     dpi_metadata=dpi if dpi else 'none', print_size_in=f'{w / dpi:.2f} x {h / dpi:.2f}' if dpi else 'n/a',
                     effective_dpi_at_180mm_width=round(w / 7.09), status=st, source=src))
M = pd.DataFrame(rows); M.to_csv(f'{O}/figure_manifest.csv', index=False)
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
pd.set_option('display.width', 200)
print(M[['file', 'width_px', 'height_px', 'dpi_metadata', 'print_size_in', 'effective_dpi_at_180mm_width']].to_string(index=False))
