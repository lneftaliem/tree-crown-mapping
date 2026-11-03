"""
figaudit.py -- optional figure audit (active only when FIG_AUDIT_LOG is set; imported by sitecustomize.py).
Wraps matplotlib Figure.savefig and appends one JSON line per saved file: path, figure size (in), dpi argument, and the
visible text artists' font sizes (min / median) with the smallest texts, plus any text with raw 'e-'/'e+' exponents or
Unicode superscript digits/minus (which render as boxes in Arial). Does not change what is saved.
"""
import os, json, re
LOG = os.environ.get('FIG_AUDIT_LOG')
SUP = re.compile('[⁰-₟¹²³]')
RAW = re.compile(r'\d(\.\d+)?e[-+]\d')


def _install():
    import matplotlib.figure as mf
    from matplotlib.text import Text
    orig = mf.Figure.savefig

    def savefig(self, fname, *a, **k):
        try:
            texts = [t for t in self.findobj(Text) if t.get_visible() and t.get_text().strip()]
            fs = sorted((float(t.get_fontsize()), t.get_text()[:40]) for t in texts)
            bad = [t.get_text()[:60] for t in texts if SUP.search(t.get_text()) or RAW.search(t.get_text())]
            with open(LOG, 'a') as f:
                f.write(json.dumps(dict(file=os.path.abspath(str(fname)), figsize=list(self.get_size_inches()), dpi=k.get('dpi'),
                                        n_text=len(fs), min_font=fs[0][0] if fs else None,
                                        median_font=fs[len(fs) // 2][0] if fs else None, smallest=fs[:4], bad_notation=bad)) + '\n')
        except Exception as e:  # the audit must never break a figure
            with open(LOG, 'a') as f:
                f.write(json.dumps(dict(file=str(fname), audit_error=repr(e))) + '\n')
        return orig(self, fname, *a, **k)
    mf.Figure.savefig = savefig


if LOG:
    try:
        import matplotlib  # noqa: F401
        _install()
    except Exception:
        pass
