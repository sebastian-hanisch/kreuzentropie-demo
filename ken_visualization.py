"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go

import ken_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_trajectories(a):
    """Verlauf von q' über die Stufen: die ersten Lernläufe (dünn), die exakte Stufenfolge aus Erwartungswerten (gestrichelt) und das exakte Optimum (waagerecht)."""
    fig = go.Figure()
    for i, tr in enumerate(a.trajectories):
        fig.add_trace(go.Scatter(x=list(range(len(tr))), y=list(tr), mode="lines+markers", name=f"Lauf {i + 1}" if i == 0 else None, showlegend=(i == 0), legendgroup="lauf",
                                 line=dict(color=C.COLOR_CE, width=1), marker=dict(size=5), opacity=0.55))
    exact = [a.ladder[0][0]] + [x[3] for x in a.ladder]
    fig.add_trace(go.Scatter(x=list(range(len(exact))), y=exact, mode="lines+markers", name="exakte Stufenfolge", line=dict(color=C.COLOR_TRUTH, dash="dash")))
    if a.qp_opt == a.qp_opt:
        fig.add_hline(y=a.qp_opt, line=dict(color=C.COLOR_ORACLE, dash="dot"), annotation_text="optimales q'", annotation_position="top left")
    fig.update_xaxes(title_text="Stufe", dtick=1)
    fig.update_yaxes(title_text="gelernte Kippung q'", type="log")
    return _base(fig, 340)


def build_grid(exp):
    """Anteil festgefahrener Läufe je (N, ρ) als Heatmap, mit dem Verhältnis zum Optimum als Beschriftung."""
    ns = sorted({c["n"] for c in exp["cells"]})
    rhos = sorted({c["rho"] for c in exp["cells"]})
    z = [[next(c["stuck"] for c in exp["cells"] if c["n"] == n and c["rho"] == r) for r in rhos] for n in ns]
    text = [[("–" if next(c["ratio"] for c in exp["cells"] if c["n"] == n and c["rho"] == r) != next(c["ratio"] for c in exp["cells"] if c["n"] == n and c["rho"] == r)
              else f"{next(c['ratio'] for c in exp['cells'] if c['n'] == n and c['rho'] == r):.2f}".replace(".", ",")) for r in rhos] for n in ns]
    fig = go.Figure(go.Heatmap(z=z, x=[str(r).replace(".", ",") for r in rhos], y=[str(n) for n in ns], colorscale=[[0, "#2F6B65"], [0.5, "#E08E0B"], [1, "#C0392B"]], zmin=0, zmax=1, text=text,
                               texttemplate="%{text}", colorbar=dict(title="festgefahren")))
    fig.update_xaxes(title_text="Anteil der Elite ρ", type="category")
    fig.update_yaxes(title_text="Ziehungen je Stufe N", type="category")
    return _base(fig, 320)


def build_budget(exp):
    rows = exp["rows"]
    fig = go.Figure()
    colors = {"einfach": C.COLOR_PLAIN, "Orakel (festes Optimum)": C.COLOR_ORACLE, "Kreuzentropie": C.COLOR_CE, "adaptives Splitting": C.COLOR_SPLIT}
    fig.add_trace(go.Bar(x=[r["method"] for r in rows], y=[r["rmse"] for r in rows], marker_color=[colors[r["method"]] for r in rows], name="relativer Fehler (RMSE)"))
    fig.update_yaxes(title_text="relativer Fehler (RMSE)", type="log")
    fig = _base(fig, 320)
    fig.update_layout(showlegend=False)
    return fig


def build_prediction(exp):
    rows = exp["rows"]
    fig = go.Figure()
    xs = [str(r["budget"]) for r in rows]
    for key, label, color in (("plain", "einfach", C.COLOR_PLAIN), ("oracle", "Orakel", C.COLOR_ORACLE), ("ce", "Kreuzentropie", C.COLOR_CE), ("split", "adaptives Splitting", C.COLOR_SPLIT)):
        fig.add_trace(go.Scatter(x=xs, y=[r[key] for r in rows], mode="markers+lines", name=label, line=dict(color=color)))
    fig.update_xaxes(title_text="Gesamtbudget (Ziehungen)", type="category")
    fig.update_yaxes(title_text="vorhergesagter relativer Fehler", type="log")
    return _base(fig, 330)


def build_fixpoint(exp):
    fig = go.Figure()
    palette = ["#2F6B65", "#4c78a8", "#E08E0B", "#C0392B", "#8e6bb0"]
    for i, r in enumerate(exp["rows"]):
        ys = [r["stages"][0][0]] + [x[3] for x in r["stages"]]
        label = f"ρ = {str(r['rho']).replace('.', ',')}" + (" (erreicht)" if r["reached"] else " (Fixpunkt)" if r["fixpoint"] else " (steht)")
        fig.add_trace(go.Scatter(x=list(range(len(ys))), y=ys, mode="lines+markers", name=label, line=dict(color=palette[i % len(palette)])))
    fig.update_xaxes(title_text="Stufe", dtick=1)
    fig.update_yaxes(title_text="q' der exakten Stufenfolge", type="log")
    return _base(fig, 340)


def build_netz(netz):
    fig = go.Figure()
    xs, ys = [c[0] for c in netz.coords], [c[1] for c in netz.coords]
    ex, ey = [], []
    for u, v in netz.edges:
        ex += [xs[u], xs[v], None]
        ey += [ys[u], ys[v], None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#666", width=3), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=22, color="white", line=dict(color="#222", width=2)), showlegend=False, hoverinfo="skip"))
    for i, (x, y) in enumerate(zip(xs, ys)):
        fig.add_annotation(x=x, y=y, text=str(i), showarrow=False, font=dict(size=10, color="#222"))
    pad = 0.5
    fig.update_xaxes(visible=False, range=[min(xs) - pad, max(xs) + pad])
    fig.update_yaxes(visible=False, range=[min(ys) - pad, max(ys) + pad])
    fig = _base(fig, 280)
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    return fig
