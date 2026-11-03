"""
ed_table1_png.py -- renders Extended Data Table 1 (ED_Table1.csv of a run) as a 300 DPI PNG + PDF (7.0 in wide), with the
same rounding as ED_Table1.md, except tree counts, which are shown to 3 significant figures (e.g. 2,590,000). Usage: python ed_table1_png.py <ED_Table1.csv> <out_stem>
"""
import sys
import pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'pdf.fonttype': 42})
t = pd.read_csv(sys.argv[1])
cols = ['City', 'Trees identified', 'Municipal land area (km2)', 'Census units with detections', 'Median density (trees km-2)', 'Maximum density (trees km-2)']
head = ['City', 'Trees\nidentified', 'Municipal land\narea (km$^{2}$)', 'Census units\nwith detections', 'Median\ndensity\n(trees km$^{-2}$)', 'Maximum\ndensity\n(trees km$^{-2}$)']
fmt = [str, lambda v: f'{int(float(f"{v:.3g}")):,}',  # tree counts to 3 significant figures (author request 2026-11-02)
       lambda v: f'{v:.1f}', lambda v: f'{int(v):,}', lambda v: f'{v:,.0f}', lambda v: f'{v:,.0f}']
cells = [[f(r[c]) for f, c in zip(fmt, cols)] for _, r in t.iterrows()]
fig, ax = plt.subplots(figsize=(7.0, 2.2)); ax.axis('off')
tb = ax.table(cellText=cells, colLabels=head, loc='center', cellLoc='center', colLoc='center',
              colWidths=[0.2, 0.14, 0.17, 0.17, 0.16, 0.16])
tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.55)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor('#999999'); cell.set_linewidth(0.5)
    if r == 0:
        cell.set_text_props(fontweight='bold'); cell.set_facecolor('#eeeeee'); cell.set_height(cell.get_height() * 2.1)
    if c == 0 and r > 0:
        cell.set_text_props(ha='left'); cell.PAD = 0.06
for ext in ('png', 'pdf'):
    fig.savefig(f'{sys.argv[2]}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
