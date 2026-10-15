from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from core import REGISTRY, cache_summary, train_track
from theme import CSS, ACCENT, LINE, SURFACE, INK

st.set_page_config(page_title="Fieldnotes · Machine Learning",page_icon=None,layout="wide",initial_sidebar_state="collapsed")
st.markdown(CSS,unsafe_allow_html=True)
st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid {LINE};padding:.15rem 0 .8rem;margin-bottom:1.25rem">
 <div style="font-family:Georgia,serif;font-size:1.2rem;color:{INK}">Fieldnotes <span style="font-family:Arial,sans-serif;font-size:.68rem;letter-spacing:.12em;color:{ACCENT};text-transform:uppercase">/ Machine learning capstone</span></div>
 <div style="font-size:.72rem;color:#68736D">23CSE301 · Research dashboard</div>
</div>""",unsafe_allow_html=True)

sections=["Overview","Regression","Classification","Clustering"]
nav=st.radio("Project sections",sections,horizontal=True,label_visibility="collapsed",key="main_nav")
st.markdown('<div class="hairline"></div>',unsafe_allow_html=True)

if nav=="Overview":
    st.markdown('<div class="kicker">The capstone · three questions</div>',unsafe_allow_html=True)
    st.title("A working atlas of machine learning")
    st.markdown('<p class="lede">Three datasets, three different modelling questions. Explore the estimates, compare the held-out results, and inspect where each method succeeds or struggles.</p>',unsafe_allow_html=True)
    counts=[("Clean records",53772,"Regression · diamonds"),("Transactions",100000,"Classification · fraud"),("Catalog products",47921,"Clustering · products")]
    html=f'''<html><head><style>body{{margin:0;background:transparent;font-family:Arial,sans-serif}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.card{{background:{SURFACE};border:1px solid {LINE};border-radius:10px;padding:18px 20px;opacity:0;transform:translateY(8px);animation:in .45s ease forwards}}.card:nth-child(2){{animation-delay:.12s}}.card:nth-child(3){{animation-delay:.24s}}.label{{font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#68736D}}.num{{font:30px Georgia,serif;color:{INK};margin:8px 0 4px}}.sub{{font-size:12px;color:#68736D}}@keyframes in{{to{{opacity:1;transform:translateY(0)}}}}</style></head><body><div class="grid">'''
    for label,value,sub in counts:
        html+=f'<div class="card"><div class="label">{label}</div><div class="num" data-value="{value}">0</div><div class="sub">{sub}</div></div>'
    html+='''</div><script>document.querySelectorAll('.num').forEach((el,i)=>{const end=+el.dataset.value,start=performance.now()+i*120,duration=1100;function tick(now){const t=Math.min(1,Math.max(0,(now-start)/duration)),ease=1-Math.pow(1-t,3);el.textContent=Math.round(end*ease).toLocaleString('en-US');if(t<1)requestAnimationFrame(tick)}requestAnimationFrame(tick)})</script></body></html>'''
    components.html(html,height=125,scrolling=False)
    st.markdown("<div style='height:10px'></div>",unsafe_allow_html=True)
    cards=st.columns(3,gap="medium")
    metadata=[("Regression","Diamond valuation","Estimate price from physical dimensions and grading characteristics.","R² · RMSE · MAE", "regression"),("Classification","Fraud detection","Measure the detection trade-off in a strongly imbalanced dataset.","Recall · Precision · ROC-AUC","classification"),("Clustering","Product segments","Inspect catalogue structure without using category outcomes in fitting.","Silhouette · Davies–Bouldin","clustering")]
    cache=cache_summary()
    for col,(name,title,desc,metrics,key) in zip(cards,metadata):
        implemented=[m for m,(f,_) in REGISTRY[key].items() if f!="pending"]
        pending=[m for m,(f,_) in REGISTRY[key].items() if f=="pending"]
        chips="".join(f'<span class="model-chip">{m}</span>' for m in implemented[:3])+"".join(f'<span class="model-chip pending-chip">{m} · pending</span>' for m in pending[:2])
        col.markdown(f'<div class="editorial-card"><div class="kicker">{name}</div><h3>{title}</h3><p style="color:#68736D;min-height:48px">{desc}</p><div class="metric-label">Comparison measures</div><p>{metrics}</p><div style="margin-top:.8rem">{chips}</div></div>',unsafe_allow_html=True)
        col.caption("Model cache is ready." if cache[key] else "Models train on first visit; the initial load may take a few minutes.")
    st.markdown('<div class="hairline"></div>',unsafe_allow_html=True)
    st.caption("Counts are from the repository datasets after each notebook’s documented cleaning step. Selecting a section loads and fits only that track.")
else:
    module_name={"Regression":"pages_regression","Classification":"pages_classification","Clustering":"pages_clustering"}[nav]
    track=nav.lower()
    try:
        with st.spinner(f"Loading {track} data and fitted models…"):
            payload=train_track(track)
        import importlib
        module=importlib.import_module(module_name)
        module.render(payload)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Add the source dataset under `datasets/` and run `python app/build_models.py`.")

st.markdown('<div class="hairline"></div><div style="font-size:.72rem;color:#68736D">Reproducible seed: 42 · Notebook-aligned preprocessing · Local model cache</div>',unsafe_allow_html=True)
