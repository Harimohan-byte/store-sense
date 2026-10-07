"""StoreSense dashboard. Run: streamlit run app.py"""
import os
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="StoreSense", layout="wide")
st.title("StoreSense")
st.caption("Store video to process log to zone KPIs. One camera, no faces, "
           "no cross-camera identification.")

need = ["output/events.csv", "output/report.csv", "output/heatmap.png"]
missing = [p for p in need if not os.path.exists(p)]
if missing:
    st.error("Missing: " + ", ".join(missing) +
             ". Run events.py, metrics.py and heatmap.py first.")
    st.stop()

rep = pd.read_csv("output/report.csv")
ev = pd.read_csv("output/events.csv").sort_values(["case_id", "start"])

c1, c2, c3 = st.columns(3)
c1.metric("Cases (customers tracked)", ev.case_id.nunique())
c2.metric("Zone visits logged", len(ev))
c3.metric("Zones flagged", int((rep.flag != "OK").sum()))

left, right = st.columns([1, 1])
with left:
    st.subheader("Foot-traffic heat map")
    st.image("output/heatmap.png", use_container_width=True)
with right:
    st.subheader("Zone KPIs vs expected profile")
    st.dataframe(rep.drop(columns="advice"), hide_index=True,
                 use_container_width=True)
    st.subheader("Recommendations")
    for r in rep.itertuples():
        text = f"**{r.zone}** ({r.flag}): {r.advice}"
        if r.flag == "OK":
            st.success(text)
        else:
            st.warning(text)
    if (rep.confidence != "ok").any():
        st.info("Some zones have fewer than 10 visitors, so z-scores are "
                "indicative only.")

st.subheader("Customer journeys (one row per case)")
ev["case"] = "Case " + ev.case_id.astype(str)
fig = px.bar(ev, base="start", x="dur", y="case", color="activity",
             orientation="h", hover_data=["start", "end"])
fig.update_layout(xaxis_title="Time in clip (s)", yaxis_title="",
                  yaxis=dict(autorange="reversed"), height=380)
st.plotly_chart(fig, use_container_width=True)

a, b = st.columns(2)
with a:
    st.subheader("Flow between zones")
    ev["next"] = ev.groupby("case_id").activity.shift(-1)
    flow = (ev.dropna(subset=["next"]).groupby(["activity", "next"])
            .size().reset_index(name="count"))
    flow.columns = ["from", "to", "count"]
    st.dataframe(flow, hide_index=True, use_container_width=True)
with b:
    st.subheader("Journey variants")
    var = ev.groupby("case_id").activity.apply(" > ".join).value_counts()
    var = var.reset_index()
    var.columns = ["journey variant", "cases"]
    st.dataframe(var, hide_index=True, use_container_width=True)

st.subheader("Dwell time per zone visit")
fig2 = px.box(ev, x="activity", y="dur", points="all",
              labels={"dur": "Dwell (s)", "activity": "Zone"})
st.plotly_chart(fig2, use_container_width=True)
