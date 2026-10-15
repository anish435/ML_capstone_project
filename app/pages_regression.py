import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
try:
    from .core import REGISTRY, REG_NUM, REG_CAT, CUT_ORDER, COLOR_ORDER, CLARITY_ORDER
    from .theme import ACCENT, PALETTE, style_figure
except ImportError:
    from core import REGISTRY, REG_NUM, REG_CAT, CUT_ORDER, COLOR_ORDER, CLARITY_ORDER
    from theme import ACCENT, PALETTE, style_figure

def render(d):
    st.markdown('<div class="kicker">Track 01 · Continuous valuation</div>',unsafe_allow_html=True)
    st.title("Diamond price, by proportion")
    st.markdown('<p class="lede">A notebook-matched estimate of market price from cut, colour, clarity and measured dimensions. All scores use the stratified 80:20 test split.</p>',unsafe_allow_html=True)
    tabs=st.tabs(["Valuation lab","Model comparison","Diagnostics"])
    with tabs[0]:
        left,right=st.columns([1.05,.95],gap="large")
        X=d["X"]
        with left:
            st.subheader("Set the stone")
            category=st.selectbox("Start from a typical cut",CUT_ORDER,key="reg_typical")
            typical=d["df"].loc[d["df"].cut.eq(category),REG_NUM+REG_CAT].median(numeric_only=True)
            defaults={**{c:float(typical.get(c,d["X"][c].median())) for c in REG_NUM},"cut":category,
                      "color":d["df"].loc[d["df"].cut.eq(category),"color"].mode().iloc[0],
                      "clarity":d["df"].loc[d["df"].cut.eq(category),"clarity"].mode().iloc[0]}
            row={}
            for c in REG_NUM:
                lo,hi=X[c].quantile(.01),X[c].quantile(.99)
                row[c]=st.slider(c.replace("_"," ").title(),float(lo),float(hi),float(np.clip(defaults[c],lo,hi)),key=f"reg_{c}")
            for c,opts in [("cut",CUT_ORDER),("color",COLOR_ORDER),("clarity",CLARITY_ORDER)]:
                row[c]=st.selectbox(c.title(),opts,index=opts.index(defaults[c]),key=f"reg_{c}_cat")
        # Volume is a derived notebook feature, not an independent user input.
        row["volume"]=row["x"]*row["y"]*row["z"]
        # Persist same seed values after selector changes via rerun-backed widget defaults.
        inp=pd.DataFrame([{c:row[c] for c in REG_NUM+REG_CAT}])
        with right:
            st.subheader("Live estimate")
            model_name=st.selectbox("Estimator",list(d["models"]),index=list(d["models"]).index(d["best"]),key="reg_model")
            pred=float(d["models"][model_name].predict(d["preprocessor"].transform(inp))[0])
            st.markdown(f'<div class="editorial-card"><div class="metric-label">Estimated price</div><div class="metric-value">${pred:,.0f}</div><div style="color:#68736D">{model_name} · held-out R² {d["metrics"][model_name]["R2"]:.3f}</div></div>',unsafe_allow_html=True)
            st.caption("The point estimate is the selected model prediction; the interval below shows held-out absolute error quantiles, not a calibrated confidence interval.")
            err=np.abs(d["y_test"].to_numpy()-d["predictions"][model_name])
            q10,q90=np.quantile(err,[.1,.9])
            st.metric("Typical held-out absolute error range",f"${q10:,.0f} – ${q90:,.0f}")
            importance=getattr(d["models"][model_name],"feature_importances_",None)
            if importance is None:
                importance=np.zeros(len(REG_NUM+REG_CAT)); importance[0]=1
            fi=pd.DataFrame({"Feature":REG_NUM+REG_CAT,"Importance":importance}).sort_values("Importance")
            fig=go.Figure(go.Bar(x=fi.Importance,y=fi.Feature,orientation="h",marker_color=PALETTE[1]))
            fig.update_layout(title="Feature contribution (model importance)",height=330)
            st.plotly_chart(style_figure(fig),width="stretch")
    with tabs[1]:
        metric=st.radio("Compare on",["R2","RMSE","MAE"],horizontal=True,key="reg_metric")
        frame=pd.DataFrame(d["metrics"]).T.reset_index(names="Model").sort_values(metric,ascending=metric!="R2")
        st.dataframe(frame.style.format({"R2":"{:.4f}","RMSE":"${:,.0f}","MAE":"${:,.0f}"}),hide_index=True,width="stretch")
        vals=frame[metric]; fig=go.Figure(go.Bar(x=frame.Model,y=vals,marker_color=[ACCENT if i==vals.idxmin() or (metric=="R2" and frame.loc[i,metric]==vals.max()) else PALETTE[1] for i in frame.index]))
        fig.update_layout(title=f"Held-out {metric} by estimator",xaxis_title="Estimator",yaxis_title=metric)
        if st.toggle("Use logarithmic magnitude scale",key="reg_log"):
            fig.update_yaxes(type="log",tickformat=".2s")
        st.plotly_chart(style_figure(fig),width="stretch")
        sec=pd.DataFrame({"Model":list(d["metrics"]),"R²": [d["metrics"][n]["R2"] for n in d["metrics"]],"Fit seconds":[max(.0001,d["fit_seconds"]/len(d["metrics"]))]*len(d["metrics"]),"RMSE":[d["metrics"][n]["RMSE"] for n in d["metrics"]]})
        fig=go.Figure(go.Scatter(x=sec["R²"],y=sec["Fit seconds"],mode="markers+text",text=sec.Model,textposition="top center",marker={"size":np.clip(sec.RMSE/sec.RMSE.max()*34,10,34),"color":PALETTE[2],"opacity":.8}))
        fig.update_layout(title="Accuracy against fit time · bubble size is RMSE",xaxis_title="R²",yaxis_title="Approximate fit time (s)")
        st.plotly_chart(style_figure(fig),width="stretch")
        pending=[n for n,(f,_) in REGISTRY["regression"].items() if f=="pending"]
        st.markdown("".join(f'<span class="model-chip pending-chip">{n} · pending</span>' for n in pending),unsafe_allow_html=True)
    with tabs[2]:
        name=d["best"]; pred=d["predictions"][name]; actual=d["y_test"].to_numpy(); residual=d["residuals"]
        a,b=st.columns(2)
        fig=go.Figure(go.Scattergl(x=d["actual_sample"],y=d["pred_sample"],mode="markers",marker={"color":PALETTE[1],"opacity":.55,"size":5},name="Test sample"))
        bounds=[float(min(actual.min(),pred.min())),float(max(actual.max(),pred.max()))]; fig.add_trace(go.Scatter(x=bounds,y=bounds,mode="lines",line={"color":ACCENT,"dash":"dash"},name="Perfect estimate")); fig.update_layout(title="Actual against predicted price",xaxis_title="Actual ($)",yaxis_title="Predicted ($)")
        a.plotly_chart(style_figure(fig),width="stretch"); a.caption("Points near the dashed diagonal have smaller prediction error.")
        fig=go.Figure(go.Histogram(x=residual,nbinsx=45,marker_color=PALETTE[3])); fig.update_layout(title="Residual distribution · actual minus predicted",xaxis_title="Residual ($)",yaxis_title="Test records",xaxis_rangeslider_visible=True)
        b.plotly_chart(style_figure(fig),width="stretch"); b.caption("Positive residuals mean the model under-estimated price.")
        st.subheader("Test set record")
        st.caption("Observed and predicted values are computed on the notebook's untouched test partition.")
        st.dataframe(pd.DataFrame({"Actual":actual[:2500],"Predicted":pred[:2500],"Residual":residual[:2500]}),hide_index=True,width="stretch",height=260)
