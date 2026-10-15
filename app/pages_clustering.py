import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st
from sklearn.cluster import KMeans
try:
    from .core import REGISTRY, CLUST_FEATURES
    from .theme import ACCENT, PALETTE, style_figure
except ImportError:
    from core import REGISTRY, CLUST_FEATURES
    from theme import ACCENT, PALETTE, style_figure

def render(d):
    st.markdown('<div class="kicker">Track 03 · Catalog structure</div>',unsafe_allow_html=True)
    st.title("A map of the product catalogue")
    st.markdown('<p class="lede">Unsupervised K-Means groups products by engineered price, rating, review, velocity and discount features. Clustering does not predict product outcomes.</p>',unsafe_allow_html=True)
    tabs=st.tabs(["Segment explorer","Model comparison","Diagnostics"])
    with tabs[0]:
        reveal=st.toggle("Reveal the outcome label",key="cluster_reveal")
        st.caption("Clusters are formed without category labels; clustering does not predict outcomes.")
        idx=d["sample_idx"]; coords=d["coords"][idx]; labels=d["labels"][idx]
        fig=go.Figure()
        for k in sorted(np.unique(labels)):
            mask=labels==k
            fig.add_trace(go.Scattergl(x=coords[mask,0],y=coords[mask,1],mode="markers",name=f"Cluster {k}",marker={"size":5,"color":PALETTE[k%len(PALETTE)],"opacity":.58},customdata=idx[mask],hovertemplate="PC1 %{x:.2f}<br>PC2 %{y:.2f}<extra>Cluster "+str(k)+"</extra>"))
        fig.update_layout(title="Products in PCA space · reproducible 6,000 row sample",xaxis_title="Principal component 1",yaxis_title="Principal component 2",height=520)
        st.plotly_chart(style_figure(fig),width="stretch")
        st.caption("Each point is a product projected into two principal components and coloured by its unsupervised cluster.")
        if reveal:
            label_col="category_name" if "category_name" in d["df"] else ("category_id" if "category_id" in d["df"] else None)
            if label_col:
                st.markdown("**Outcome label revealed for interpretation only.**")
                frame=d["df"].copy(); frame["Cluster"]=d["labels"]
                st.dataframe(frame.groupby("Cluster")[label_col].agg(lambda x:x.mode().iloc[0] if not x.mode().empty else "—").rename("Most common category"),width="stretch")
            else: st.info("No outcome/category label is present in this dataset.")
        sizes=pd.Series(d["labels"]).value_counts().sort_index()
        cols=st.columns(len(sizes))
        for i,(k,v) in enumerate(sizes.items()): cols[i].markdown(f'<div class="editorial-card"><div class="metric-label">Cluster {k}</div><div class="metric-value">{v:,}</div><div>{v/len(d["labels"]):.1%} of clean products</div></div>',unsafe_allow_html=True)
        st.subheader("Try your own product")
        raw=d["df"]; row={}
        colspec=[("price","Price"),("stars_imputed","Stars"),("has_rating","Has rating"),("reviews","Reviews"),("has_reviews","Has reviews"),("boughtInLastMonth","Bought last month"),("has_list_price","Has list price"),("has_discount","Has discount")]
        c1,c2=st.columns(2)
        for i,(key,label) in enumerate(colspec):
            target=key
            if key=="reviews": target="log_reviews"
            elif key=="boughtInLastMonth": target="log_bought"
            elif key=="price": target="log_price"
            lo,hi=d["X"][target].quantile([.01,.99]); med=d["X"][target].median()
            col=c1 if i%2==0 else c2
            if key in {"has_rating","has_reviews","has_list_price","has_discount"}:
                val=col.selectbox(label,[0,1],index=int(round(float(med))),key=f"cluster_input_{key}")
            else:
                val=col.slider(label,float(lo),float(hi),float(med),key=f"cluster_input_{key}")
            row[target]=float(np.log1p(val)) if key in ("price","reviews","boughtInLastMonth") else val
        x=np.array([[row[f] for f in CLUST_FEATURES]])
        z=d["scaler"].transform(pd.DataFrame(x,columns=CLUST_FEATURES)); cluster=int(d["model"].predict(z)[0]); dist=np.linalg.norm(d["model"].cluster_centers_-z,axis=1)
        st.markdown(f'<div class="editorial-card"><div class="metric-label">Nearest segment</div><div class="metric-value">Cluster {cluster}</div><div>Distance to its centre: {dist[cluster]:.2f} standardised units</div></div>',unsafe_allow_html=True)
        fig=go.Figure(go.Bar(x=[f"Cluster {k}" for k in range(len(dist))],y=dist,marker_color=[ACCENT if k==cluster else PALETTE[1] for k in range(len(dist))])); fig.update_layout(title="Distance from this product to each cluster centre",yaxis_title="Euclidean distance")
        st.plotly_chart(style_figure(fig),width="stretch"); st.caption("The nearest centre determines the assigned cluster in the fitted K-Means feature space.")
    with tabs[1]:
        frame=pd.DataFrame([{"Metric":k,"Value":v} for k,v in d["metrics"].items()]); st.dataframe(frame,hide_index=True,width="stretch")
        metric=st.radio("Sweep metric",["inertia","silhouette"],horizontal=True,key="clust_metric")
        ks=list(d["sweep"]); values=[d["sweep"][k][metric] for k in ks]
        fig=go.Figure(go.Scatter(x=ks,y=values,mode="lines+markers",line={"color":PALETTE[1]},marker={"size":8})); fig.update_layout(title=f"K-Means {metric} across cluster counts",xaxis_title="Number of clusters (k)",yaxis_title=metric)
        chosen=st.slider("Sweep frame · cluster count",2,8,4,key="clust_k")
        fig.add_trace(go.Scatter(x=[chosen],y=[d["sweep"][chosen][metric]],mode="markers",marker={"size":15,"color":ACCENT},name=f"k = {chosen}"))
        st.plotly_chart(style_figure(fig),width="stretch"); st.caption("The curve compares real fitted K-Means partitions for k=2 through 8; the marker follows the slider.")
        bubble=go.Figure(go.Scatter(x=[d["metrics"]["Silhouette"]],y=[d["fit_seconds"]],mode="markers+text",text=["K-Means"],textposition="top center",marker={"size":28,"color":ACCENT})); bubble.update_layout(title="Silhouette against fit time",xaxis_title="Silhouette",yaxis_title="Fit time (seconds)")
        st.plotly_chart(style_figure(bubble),width="stretch")
        st.markdown('<span class="model-chip">K-Means · fitted</span><span class="model-chip pending-chip">Agglomerative Ward · pending</span>',unsafe_allow_html=True)
    with tabs[2]:
        n=st.slider("Features for profile",3,len(CLUST_FEATURES),len(CLUST_FEATURES),key="clust_features")
        means=d["X"].assign(Cluster=d["labels"]).groupby("Cluster")[CLUST_FEATURES[:n]].mean()
        fig=go.Figure()
        for k,row in means.iterrows(): fig.add_trace(go.Scatterpolar(r=row.values,theta=row.index,fill="toself",name=f"Cluster {k}",line={"color":PALETTE[k%len(PALETTE)]},opacity=.55))
        fig.update_layout(title="Standardised feature profile by segment",polar={"radialaxis":{"visible":True}},height=480)
        st.plotly_chart(style_figure(fig),width="stretch"); st.caption("Each radar line compares cluster-average engineered features on their original scale.")
        pca=d["pca"]
        st.metric("PCA variance shown",f"{pca.explained_variance_ratio_.sum():.1%}")
        if st.toggle("Explore hierarchical cut",key="dendro_on"):
            # Reproducible 120-row subset keeps the interactive diagnostic responsive.
            from scipy.cluster.hierarchy import linkage, fcluster
            sample=d["sample_idx"][:120]; Z=linkage(d["scaled"][sample],method="ward")
            height=float(st.slider("Dendrogram cut height",float(Z[:,2].min()),float(Z[:,2].max()),float(np.quantile(Z[:,2],.72)),key="dendro_height"))
            fig=ff.create_dendrogram(d["scaled"][sample],orientation="left",labels=[str(i) for i in sample],color_threshold=height)
            fig.update_layout(title="Ward dendrogram · 120 product sample",height=700,showlegend=False)
            fig.add_vline(x=height,line_dash="dash",line_color=ACCENT)
            st.plotly_chart(style_figure(fig),width="stretch"); st.caption("Branches join at their Ward linkage distance; the slider changes the displayed cut height.")
        # Cluster sweep animation uses Plotly frames; controls remain usable in static HTML fallback.
        if st.toggle("Animate cluster-count sweep",key="clust_anim"):
            idx=d["sample_idx"][:1200]; frames=[]
            for k in range(2,9):
                km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(d["scaled"][idx])
                frames.append(go.Frame(data=[go.Scattergl(x=d["coords"][idx,0],y=d["coords"][idx,1],mode="markers",marker={"color":km.labels_,"colorscale":"Viridis","size":5})],name=str(k)))
            fig=go.Figure(data=[go.Scattergl(x=d["coords"][idx,0],y=d["coords"][idx,1],mode="markers",marker={"color":d["labels"][idx],"colorscale":"Viridis","size":5})],frames=frames)
            fig.update_layout(title="Re-fit clusters recoloured in PCA space",updatemenus=[{"type":"buttons","buttons":[{"label":"Play","method":"animate","args":[None]},{"label":"Pause","method":"animate","args":[[None],{"mode":"immediate","frame":{"duration":0}}]}]}],sliders=[{"steps":[{"label":str(k),"method":"animate","args":[[str(k)],{"mode":"immediate","frame":{"duration":0}}]} for k in range(2,9)]}])
            st.plotly_chart(style_figure(fig),width="stretch"); st.caption("Each frame refits K-Means at a different k and recolours the same product sample.")
