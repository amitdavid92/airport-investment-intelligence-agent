"""Streamlit chat UI.

Two panes on purpose. The chat is the agent; the sidebar queries the scoring
engine directly over HTTP with no model involved. Running the same question
through both is the fastest way to show that the numbers come from the engine.
"""

from __future__ import annotations

import json
import os

import httpx
import streamlit as st

API_BASE = os.getenv("AIRPORT_IQ_API", "http://localhost:8000")
REQUEST_TIMEOUT = 180.0

st.set_page_config(page_title="Airport Investment Intelligence", page_icon="✈️", layout="wide")

# Single-hue bars: the five components are one measure (points contributed)
# across five named categories, so they share a colour and need no legend.
# Values are direct-labelled rather than axis-read.
st.markdown(
    """
    <style>
      .aiq-bar-row { margin: 0.45rem 0; }
      .aiq-bar-label {
        display: flex; justify-content: space-between;
        font-size: 0.78rem; margin-bottom: 3px;
        color: var(--aiq-text-secondary, #52514e);
      }
      .aiq-bar-track {
        background: var(--aiq-track, #ecebe8);
        border-radius: 4px; height: 9px; width: 100%;
      }
      .aiq-bar-fill {
        background: var(--aiq-series, #2a78d6);
        border-radius: 4px; height: 9px;
      }
      .aiq-bar-value { font-variant-numeric: tabular-nums; font-weight: 600; }
      .aiq-missing { color: var(--aiq-text-muted, #8a8983); font-style: italic; }
      @media (prefers-color-scheme: dark) {
        .aiq-bar-label { color: var(--aiq-text-secondary, #c3c2b7); }
        .aiq-bar-track { background: #2f2f2c; }
        .aiq-bar-fill  { background: #3987e5; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str, **params):
    try:
        response = httpx.get(f"{API_BASE}{path}", params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        return {"error": f"{exc.response.status_code}: {exc.response.text[:300]}"}
    except httpx.HTTPError as exc:
        return {"error": f"Cannot reach the API at {API_BASE} ({exc})"}


def component_bars(components: list[dict]) -> None:
    """Horizontal bars for the five score components, largest contribution first."""
    for entry in components:
        name = entry["component"].replace("_", " ").title()
        if not entry.get("available"):
            st.markdown(
                f'<div class="aiq-bar-row"><div class="aiq-bar-label">'
                f'<span>{name}</span><span class="aiq-missing">no data</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            continue
        score = entry["score_0_100"] or 0.0
        contributed = entry.get("points_contributed") or 0.0
        weight = entry["weight_in_profile"]
        st.markdown(
            f'<div class="aiq-bar-row">'
            f'<div class="aiq-bar-label"><span>{name} '
            f'<span style="opacity:.6">· weight {weight:.0%}</span></span>'
            f'<span class="aiq-bar-value">{score:.0f}'
            f'<span style="opacity:.6; font-weight:400"> → {contributed:.1f} pts</span>'
            f'</span></div>'
            f'<div class="aiq-bar-track"><div class="aiq-bar-fill" '
            f'style="width:{min(max(score,0),100):.1f}%"></div></div>'
            f"</div>",
            unsafe_allow_html=True,
        )


def render_breakdown(profile: dict) -> None:
    if "error" in profile or "detail" in profile:
        st.warning(profile.get("error") or profile.get("detail"))
        return

    st.markdown(f"**{profile['iata']} — {profile['name']}**")
    st.caption(
        f"{profile.get('city') or '—'}, {profile.get('state') or '—'} · "
        f"{profile['hub_class'].replace('_', ' ')}"
    )

    left, right = st.columns(2)
    left.metric("Investment score", f"{profile['investment_score']:.1f}")
    right.metric("National rank", f"#{profile['rank_overall']} / {profile['of_airports']}")

    confidence = profile.get("confidence", {})
    band = confidence.get("band", "unknown")
    icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(band, "⚪")
    st.caption(
        f"{icon} Confidence: **{band}** · profile: `{profile['weight_profile']}` · "
        f"ranked {profile['normalized_within']}"
    )

    st.markdown("###### Score components")
    component_bars(profile["components"])

    for note in confidence.get("notes", []):
        st.info(note, icon="ℹ️")

    with st.expander("Underlying figures"):
        figures = profile["raw_kpis"]
        st.dataframe(
            {
                "Metric": [k.replace("_", " ").title() for k in figures],
                "Value": [
                    f"{v:,.0f}" if isinstance(v, (int, float)) and v and abs(v) >= 1000
                    else ("—" if v is None else v)
                    for v in figures.values()
                ],
            },
            hide_index=True,
            width="stretch",
        )
        st.caption(profile["data_vintage"]["caveat"])


# --------------------------------------------------------------------------
# Sidebar: the scoring engine, queried directly. No model in this path.
# --------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Scoring engine")
    st.caption("Queries `/rank` directly — no language model in this path.")

    health = api_get("/health")
    if "error" in health:
        st.error(health["error"])
        st.stop()

    options = api_get("/options")
    regions = options.get("regions", {})

    region_key = st.selectbox(
        "Region",
        ["(all US)"] + list(regions),
        format_func=lambda k: regions.get(k, "All US airports"),
    )
    profile_key = st.selectbox("Weight profile", options.get("weight_profiles", ["balanced"]))
    top_n = st.slider("Results", 3, 20, 5)

    params = {"weight_profile": profile_key, "top_n": top_n}
    if region_key != "(all US)":
        params["region"] = region_key
    ranking = api_get("/rank", **params)

    if "error" in ranking:
        st.error(ranking["error"])
    else:
        st.dataframe(
            {
                "#": [r["position_in_result"] for r in ranking["results"]],
                "Airport": [r["iata"] for r in ranking["results"]],
                "Score": [f"{r['investment_score']:.1f}" for r in ranking["results"]],
            },
            hide_index=True,
            width="stretch",
        )
        for warning in ranking.get("warnings", []):
            st.caption(f"⚠️ {warning}")

        selected = st.selectbox(
            "Inspect breakdown",
            [r["iata"] for r in ranking["results"]],
            key="inspect",
        )
        if selected:
            st.divider()
            render_breakdown(api_get(f"/airport/{selected}", weight_profile=profile_key))

    st.divider()
    st.caption(
        f"{health['airports_scored']} airports scored · "
        f"long-haul: `{health['long_haul_provider']}`"
    )

# --------------------------------------------------------------------------
# Main pane: the agent.
# --------------------------------------------------------------------------

st.title("✈️ Airport Investment Intelligence")
st.caption(
    "Ask about US airport modernization and expansion candidates. "
    "Every figure comes from the deterministic scoring engine — the model "
    "selects tools and explains results, it never computes a number."
)

if not health.get("gemini_key_configured"):
    st.warning(
        "`GEMINI_API_KEY` is not set, so the chat is disabled. The scoring "
        "engine in the sidebar works without it.",
        icon="🔑",
    )

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.session_id = None

EXAMPLES = [
    "Which airports in New England are strong candidates for terminal expansion?",
    "Compare LA and Santa Ana airport congestion levels.",
    "What is the percentage of long haul flights out of Anchorage airport?",
    "What is the unmet flight demand in SFO airport and why?",
]

if not st.session_state.history:
    st.markdown("###### Try one of these")
    columns = st.columns(2)
    for index, example in enumerate(EXAMPLES):
        if columns[index % 2].button(example, key=f"ex{index}", width="stretch"):
            st.session_state.pending = example
            st.rerun()

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        if message.get("tools"):
            st.caption("🔧 " + " · ".join(message["tools"]))
        st.markdown(message["content"])

prompt = st.chat_input("Ask about airport investment potential…")
if "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    if not health.get("gemini_key_configured"):
        st.error("Set GEMINI_API_KEY and restart the API to use the chat.")
        st.stop()

    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        tools_used: list[str] = []
        tool_slot = st.empty()
        text_slot = st.empty()
        answer = ""

        payload = {"message": prompt, "session_id": st.session_state.session_id}
        try:
            with httpx.stream(
                "POST", f"{API_BASE}/chat", json=payload, timeout=REQUEST_TIMEOUT
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    kind = event.get("type")

                    if kind == "session":
                        st.session_state.session_id = event["session_id"]
                    elif kind == "tool_call":
                        tools_used.append(event["name"])
                        tool_slot.caption("🔧 " + " · ".join(tools_used))
                    elif kind == "text":
                        answer += event["text"]
                        text_slot.markdown(answer)
                    elif kind == "error":
                        st.error(event["message"])
        except httpx.HTTPError as exc:
            st.error(f"Request failed: {exc}")

        if answer:
            st.session_state.history.append(
                {"role": "assistant", "content": answer, "tools": tools_used}
            )

if st.session_state.history and st.button("Clear conversation"):
    if st.session_state.session_id:
        httpx.request("DELETE", f"{API_BASE}/chat/{st.session_state.session_id}", timeout=30)
    st.session_state.history = []
    st.session_state.session_id = None
    st.rerun()
