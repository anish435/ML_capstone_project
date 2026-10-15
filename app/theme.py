"""Visual system for the editorial ML dashboard."""
import plotly.graph_objects as go

INK = "#26342F"
MUTED = "#68736D"
PAPER = "#F6F4EE"
SURFACE = "#FFFEFA"
LINE = "#D9DED6"
ACCENT = "#B35336"
PALETTE = ["#B35336", "#526F61", "#B58B50", "#657B91", "#936B5B", "#82927D", "#A6A193"]

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial, sans-serif", "color": INK, "size": 12},
        title={"font": {"family": "Georgia, serif", "size": 17, "color": INK}},
        colorway=PALETTE,
        xaxis={"gridcolor": LINE, "zerolinecolor": LINE, "linecolor": LINE, "ticks": "outside"},
        yaxis={"gridcolor": LINE, "zerolinecolor": LINE, "linecolor": LINE, "ticks": "outside"},
        hoverlabel={"bgcolor": SURFACE, "bordercolor": LINE, "font": {"family": "Arial, sans-serif", "color": INK}},
        margin={"l": 20, "r": 18, "t": 48, "b": 20},
    )
)

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap');
:root {{ --paper:{PAPER}; --surface:{SURFACE}; --ink:{INK}; --muted:{MUTED}; --line:{LINE}; --accent:{ACCENT}; }}
.stApp {{ background:var(--paper); color:var(--ink); font-family:'DM Sans',Arial,sans-serif; }}
[data-testid="stHeader"] {{ background:rgba(246,244,238,.94); }}
.block-container {{ max-width:1440px; padding-top:4.2rem; padding-bottom:4rem; }}
h1,h2,h3 {{ font-family:'Newsreader',Georgia,serif !important; font-weight:400 !important; color:var(--ink); }}
h1 {{ font-size:2.35rem !important; }} h2 {{ font-size:1.65rem !important; }} h3 {{ font-size:1.25rem !important; }}
.kicker {{ text-transform:uppercase; letter-spacing:.16em; font-size:.68rem; font-weight:700; color:var(--accent); margin:0 0 .45rem; }}
.lede {{ color:var(--muted); font-size:.95rem; max-width:780px; line-height:1.65; }}
.hairline {{ border-top:1px solid var(--line); margin:1.2rem 0 1.6rem; }}
.editorial-card {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:1.1rem 1.25rem; animation:rise .42s ease both; }}
.metric-label {{ color:var(--muted); text-transform:uppercase; letter-spacing:.12em; font-size:.65rem; }}
.metric-value {{ font-family:'Newsreader',Georgia,serif; font-size:1.8rem; color:var(--ink); margin-top:.35rem; }}
.model-chip {{ display:inline-block; padding:.3rem .55rem; border:1px solid var(--line); border-radius:999px; font-size:.72rem; margin:.15rem .25rem .15rem 0; color:var(--ink); }}
.pending-chip {{ border-style:dashed; color:var(--muted); }}
.stTabs [data-baseweb="tab-list"] {{ gap:.35rem; border-bottom:1px solid var(--line); }}
.stTabs [data-baseweb="tab"] {{ background:transparent; border:1px solid transparent; border-radius:999px; padding:.45rem .9rem; color:var(--muted); }}
.stTabs [aria-selected="true"] {{ color:var(--ink); background:var(--surface); border-color:var(--line); }}
div[role="radiogroup"] {{ gap:.35rem; }}
div[role="radiogroup"] label {{ background:var(--surface); border:1px solid var(--line); border-radius:999px; padding:.35rem .8rem; }}
.stButton button {{ border-radius:999px; border:1px solid var(--line); }}
@keyframes rise {{ from {{opacity:0; transform:translateY(7px)}} to {{opacity:1; transform:translateY(0)}} }}
@media(prefers-reduced-motion:reduce) {{ * {{ animation:none !important; }} }}
</style>
"""

def style_figure(fig):
    fig.update_layout(template=PLOTLY_TEMPLATE, transition={"duration": 320})
    return fig
