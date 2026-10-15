import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
try:
    from .core import REGISTRY, CLF_FEATURES, CLF_NUM, CLF_CAT, CLF_BIN
    from .theme import ACCENT, PALETTE, style_figure
except ImportError:
    from core import REGISTRY, CLF_FEATURES, CLF_NUM, CLF_CAT, CLF_BIN
    from theme import ACCENT, PALETTE, style_figure

def render(d):
    st.markdown('<div class="kicker">Track 02 · Financial decisioning</div>',unsafe_allow_html=True)
    st.title("Fraud signals in context")
    st.markdown('<p class="lede">A class-imbalanced fraud study. Read recall and ROC-AUC alongside accuracy; a high accuracy alone can hide missed fraud.</p>',unsafe_allow_html=True)
    tabs=st.tabs(["Transaction lab","Model comparison","Diagnostics"])
    with tabs[0]:
        col,view=st.columns([1,1],gap="large")
        with col:
            st.subheader("Describe a transaction")
            X=d["X"]; row={}
            preset=st.selectbox("Start from typical class",["Overall median","Legitimate (0)","Fraudulent (1)"],key="cl_preset")
            cls=0 if preset.startswith("Legitimate") else 1
            sample=d["df"] if preset=="Overall median" else d["df"].loc[d["df"].risk_label.eq(cls)]
            numeric_defaults=sample[CLF_NUM].median(numeric_only=True)
            for c in CLF_NUM:
                q1,q99=X[c].quantile([.01,.99]); default=float(np.clip(numeric_defaults[c],q1,q99))
                row[c]=st.slider(c.replace("_"," ").title(),float(q1),float(q99),default,key=f"clf_{c}")
            for c in CLF_CAT:
                options=sorted(X[c].dropna().astype(str).unique().tolist())
                mode=sample[c].mode().iloc[0] if not sample[c].mode().empty else options[0]
                row[c]=st.selectbox(c.replace("_"," ").title(),options,index=options.index(str(mode)) if str(mode) in options else 0,key=f"clf_{c}")
            for c in CLF_BIN:
                if c not in row: row[c]=int(sample[c].median()>=.5) if c in sample else 0
            row["amount_ratio"]=row["transaction_amount"]/(row["avg_transaction_amount_30d_customer"]+1)
        with view:
            st.subheader("Risk estimate")
            model_name=st.selectbox("Estimator",list(d["models"]),index=list(d["models"]).index(d["best"]),key="clf_model")
            raw=pd.DataFrame([{c:row[c] for c in CLF_FEATURES}])
            z=d["preprocessor"].transform(raw); model=d["models"][model_name]
            probabilities=model.predict_proba(z)[0]; classes=list(model.classes_); fraud=float(probabilities[classes.index(1)])
            pred=int(model.predict(z)[0]); label="Fraud signal" if pred==1 else "Legitimate signal"
            st.markdown(f'<div class="editorial-card"><div class="metric-label">Predicted class</div><div class="metric-value">{label}</div><div style="color:#68736D">Fraud probability · {fraud:.1%}</div></div>',unsafe_allow_html=True)
            fig=go.Figure(go.Bar(x=["Legitimate","Fraud"],y=[float(probabilities[classes.index(0)]),fraud],marker_color=[PALETTE[1],ACCENT]))
            frames=[go.Frame(data=[go.Bar(x=["Legitimate","Fraud"],y=[float(probabilities[classes.index(0)])*k/12,fraud*k/12])],name=str(k)) for k in range(1,13)]
            fig.frames=frames; fig.update_layout(title="Class probabilities",yaxis={"range":[0,1],"tickformat":".0%"},updatemenus=[{"type":"buttons","x":0,"y":1.15,"buttons":[{"label":"Replay","method":"animate","args":[None,{"frame":{"duration":45,"redraw":True},"fromcurrent":False}]},{"label":"Pause","method":"animate","args":[[None],{"mode":"immediate","frame":{"duration":0},"transition":{"duration":0}}]}]}])
            st.plotly_chart(style_figure(fig),width="stretch")
            st.caption("Bars show this model's class probabilities for the current transaction inputs.")
            allprob={n:float(m.predict_proba(z)[0][list(m.classes_).index(1)]) for n,m in d["models"].items() if hasattr(m,"predict_proba")}
            hm=go.Figure(go.Heatmap(z=[list(allprob.values())],x=list(allprob),y=["Fraud probability"],zmin=0,zmax=1,colorscale=[[0,"#E9EDE7"],[1,ACCENT]],text=[[f"{x:.3f}" for x in allprob.values()]],texttemplate="%{text}"))
            hm.update_layout(title="Every implemented model on this input",height=190)
            st.plotly_chart(style_figure(hm),width="stretch")
            st.caption("This heatmap compares fraud probabilities across the implemented estimators on one shared input.")
    with tabs[1]:
        metric=st.radio("Compare on",["ROC-AUC","Recall","Precision","Weighted F1","Accuracy"],horizontal=True,key="clf_metric")
        frame=pd.DataFrame(d["metrics"]).T.reset_index(names="Model").sort_values(metric,ascending=False)
        st.dataframe(frame.style.format({c:"{:.4f}" for c in frame.columns if c!="Model"}),hide_index=True,width="stretch")
        vals=frame[metric]; fig=go.Figure(go.Bar(x=frame.Model,y=vals,marker_color=[ACCENT if v==vals.max() else PALETTE[1] for v in vals]))
        fig.update_layout(title=f"Held-out {metric} by estimator",yaxis_title=metric)
        if st.toggle("Zoom to observed range",key="clf_zoom") and len(vals): fig.update_yaxes(range=[max(0,float(vals.min())-.04),min(1,float(vals.max())+.02)])
        st.plotly_chart(style_figure(fig),width="stretch")
        times=np.arange(1,len(frame)+1)
        bubble=go.Figure(go.Scatter(x=frame["ROC-AUC"],y=frame["Recall"],mode="markers+text",text=frame.Model,textposition="top center",marker={"size":np.clip(frame["Weighted F1"]*34+8,8,34),"color":PALETTE[2],"opacity":.8}))
        bubble.update_layout(title="Recall against ROC-AUC · bubble size is weighted F1",xaxis_title="ROC-AUC",yaxis_title="Recall")
        st.plotly_chart(style_figure(bubble),width="stretch")
        pending=[n for n,(f,_) in REGISTRY["classification"].items() if f=="pending"]
        st.markdown("".join(f'<span class="model-chip pending-chip">{n} · pending</span>' for n in pending),unsafe_allow_html=True)
    with tabs[2]:
        name=st.selectbox("Diagnostic estimator",list(d["models"]),index=list(d["models"]).index(d["best"]),key="clf_diag")
        cm=d["confusion"][name].copy()
        if st.toggle("Show row percentages",key="clf_pct"): cm=cm/cm.sum(axis=1,keepdims=True)*100
        fig=go.Figure(go.Heatmap(z=cm,x=["Predicted legitimate","Predicted fraud"],y=["Actual legitimate","Actual fraud"],colorscale=[[0,"#EEF0E9"],[1,ACCENT]],text=np.round(cm,1),texttemplate="%{text}"))
        fig.update_layout(title=f"Confusion matrix · {name}",height=390)
        st.plotly_chart(style_figure(fig),width="stretch"); st.caption("Rows are actual classes and columns are predicted classes; use the toggle to normalize each actual class.")
        fpr,tpr,_=d["roc"][name]
        fig=go.Figure(go.Scatter(x=fpr,y=tpr,mode="lines",name=f"{name} · AUC {d['metrics'][name]['ROC-AUC']:.3f}",line={"color":ACCENT,"width":2})); fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line={"dash":"dash","color":PALETTE[6]},name="Chance")); fig.update_layout(title="One-vs-rest fraud ROC curve",xaxis_title="False positive rate",yaxis_title="True positive rate")
        st.plotly_chart(style_figure(fig),width="stretch"); st.caption("The ROC curve shows the trade-off between detecting fraud and raising false alarms across thresholds.")
        model=d["models"][name]; imp=getattr(model,"feature_importances_",None)
        if imp is None and hasattr(model,"coef_"): imp=np.mean(np.abs(model.coef_),axis=0)
        if imp is not None:
            names=d["preprocessor"].get_feature_names_out(); n=st.slider("Features to show",5,min(30,len(names)),12,key="clf_topn")
            fi=pd.DataFrame({"Feature":names,"Importance":imp}).nlargest(n,"Importance").sort_values("Importance")
            fig=go.Figure(go.Bar(x=fi.Importance,y=fi.Feature,orientation="h",marker_color=PALETTE[1])); fig.update_layout(title=f"Top {n} model importance values",height=430)
            st.plotly_chart(style_figure(fig),width="stretch"); st.caption("Importance is an estimator-specific ranking and does not establish causal effect.")
