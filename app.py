"""
Krishi Rakshak: Satellite Edition 🛰️🌾
──────────────────────────────────────
A production-ready Streamlit app that lets farmers draw a field boundary
on an interactive map, fetches Sentinel-2 L2A imagery from the Microsoft
Planetary Computer STAC API, computes NDVI crop-health maps, and renders
a "Growth Chart" showing vegetation health over the last 6 months.
"""

# ── Standard library ────────────────────────────────────────────────
import datetime
import warnings

# ── Third-party ─────────────────────────────────────────────────────
import altair as alt
import folium
import folium.plugins
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import streamlit as st
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Polygon, shape
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Silence noisy rioxarray / GDAL deprecation warnings
warnings.filterwarnings("ignore")

# ── Page configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="Krishi Rakshak — Satellite Edition",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ───────────────────────────────────────────────────────
DEFAULT_CENTER = [17.8, 79.0]  # Telangana, India
DEFAULT_ZOOM = 7
STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
MAX_CLOUD_COVER = 10          # percent
LOOKBACK_MONTHS = 6
NDVI_CMAP = LinearSegmentedColormap.from_list(
    "crop_health", ["#d73027", "#fee08b", "#1a9850"]  # Red → Yellow → Green
)

# ── Multilingual support ────────────────────────────────────────────
LANGUAGES = ["English", "Hindi", "Telugu", "Tamil", "Marathi", "Punjabi", "Bengali"]

TITLE_MAP = {
    "English": "🌾 Krishi Rakshak — Satellite Edition 🛰️",
    "Hindi": "🌾 कृषि रक्षक — सैटेलाइट संस्करण 🛰️",
    "Telugu": "🌾 కృషి రక్షక్ — శాటిలైట్ ఎడిషన్ 🛰️",
    "Tamil": "🌾 கிருஷி ரக்ஷக் — செயற்கைக்கோள் பதிப்பு 🛰️",
    "Marathi": "🌾 कृषी रक्षक — उपग्रह आवृत्ती 🛰️",
    "Punjabi": "🌾 ਕ੍ਰਿਸ਼ੀ ਰਕਸ਼ਕ — ਸੈਟੇਲਾਈਟ ਐਡੀਸ਼ਨ 🛰️",
    "Bengali": "🌾 কৃষি রক্ষক — স্যাটেলাইট সংস্করণ 🛰️",
}

SUBTITLE_MAP = {
    "English": "Draw your farm boundary → Get instant crop-health insights from space",
    "Hindi": "अपने खेत की सीमा बनाएं → अंतरिक्ष से फसल स्वास्थ्य की जानकारी पाएं",
    "Telugu": "మీ పొలం సరిహద్దును గీయండి → అంతరిక్షం నుండి పంట ఆరోగ్య సమాచారం పొందండి",
    "Tamil": "உங்கள் பண்ணை எல்லையை வரையுங்கள் → விண்வெளியிலிருந்து பயிர் ஆரோக்கிய நுண்ணறிவுகளைப் பெறுங்கள்",
    "Marathi": "तुमच्या शेताची सीमा काढा → अवकाशातून पीक आरोग्य माहिती मिळवा",
    "Punjabi": "ਆਪਣੇ ਖੇਤ ਦੀ ਸੀਮਾ ਖਿੱਚੋ → ਪੁਲਾੜ ਤੋਂ ਫ਼ਸਲ ਸਿਹਤ ਜਾਣਕਾਰੀ ਪ੍ਰਾਪਤ ਕਰੋ",
    "Bengali": "আপনার খামারের সীমানা আঁকুন → মহাকাশ থেকে ফসল স্বাস্থ্য তথ্য পান",
}


# ═══════════════════════════════════════════════════════════════════
# DATA FETCHING (cached)
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="🔍 Searching satellite catalogue…", ttl=3600)
def search_sentinel_items(geojson_geometry: dict, date_range: str):
    """Search MS Planetary Computer for Sentinel-2 L2A items."""
    try:
        catalog = pystac_client.Client.open(
            STAC_API_URL,
            modifier=planetary_computer.sign_inplace,
        )
        search = catalog.search(
            collections=[COLLECTION],
            intersects=geojson_geometry,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
            max_items=60,
        )
        items = list(search.items())
        # Sort newest → oldest
        items.sort(key=lambda i: i.datetime, reverse=True)
        return items
    except Exception as exc:
        st.error(f"🛰️ Satellite connecting… please wait.\n\nDetails: {exc}")
        return []


@st.cache_data(show_spinner="📡 Downloading satellite bands…", ttl=3600)
def load_bands(item_href: str, bbox: tuple, resolution: int = 10):
    """Load Red (B04) & NIR (B08) bands for a single STAC item via odc-stac."""
    import odc.stac  # lazy import – heavy dependency

    # Re-open the item from its self link (pystac Items aren't picklable)
    catalog = pystac_client.Client.open(
        STAC_API_URL,
        modifier=planetary_computer.sign_inplace,
    )
    # search by the item id
    item_id = item_href  # we'll pass the ID string
    search = catalog.search(
        collections=[COLLECTION],
        ids=[item_id],
    )
    items = list(search.items())
    if not items:
        return None, None, None, None

    item = items[0]

    data = odc.stac.load(
        [item],
        bands=["B04", "B08", "B02", "B03"],
        bbox=bbox,
        resolution=resolution,
        groupby="solar_day",
    )

    red = data["B04"].isel(time=0).values.astype(float)
    nir = data["B08"].isel(time=0).values.astype(float)
    blue = data["B02"].isel(time=0).values.astype(float)
    green = data["B03"].isel(time=0).values.astype(float)

    return red, nir, blue, green


@st.cache_data(show_spinner="📈 Building growth chart data…", ttl=3600)
def compute_ndvi_timeseries(item_ids: list, bbox: tuple, resolution: int = 20):
    """Compute mean NDVI for each item id → returns a DataFrame."""
    import odc.stac

    records = []
    catalog = pystac_client.Client.open(
        STAC_API_URL,
        modifier=planetary_computer.sign_inplace,
    )

    progress = st.progress(0, text="Loading satellite images…")
    total = len(item_ids)

    for idx, iid in enumerate(item_ids):
        try:
            search = catalog.search(collections=[COLLECTION], ids=[iid])
            items = list(search.items())
            if not items:
                continue
            item = items[0]
            data = odc.stac.load(
                [item],
                bands=["B04", "B08"],
                bbox=bbox,
                resolution=resolution,
                groupby="solar_day",
            )
            red = data["B04"].isel(time=0).values.astype(float)
            nir = data["B08"].isel(time=0).values.astype(float)
            denom = nir + red
            ndvi = np.where(denom > 0, (nir - red) / denom, 0)
            mean_ndvi = float(np.nanmean(ndvi))
            records.append({"date": item.datetime.strftime("%Y-%m-%d"), "ndvi": mean_ndvi})
        except Exception:
            pass  # skip problematic scenes
        progress.progress((idx + 1) / total, text=f"Image {idx+1}/{total}")

    progress.empty()
    if not records:
        return pd.DataFrame(columns=["date", "ndvi"])
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════
# AI ANALYSIS (Gemini)
# ═══════════════════════════════════════════════════════════════════

def generate_ai_advice(
    mean_ndvi: float,
    max_ndvi: float,
    min_ndvi: float,
    trend_delta: float | None,
    selected_language: str,
    api_key: str,
) -> str:
    """Send satellite-derived stats to Gemini and get farmer-friendly advice."""
    if not GEMINI_AVAILABLE:
        return "⚠️ `google-generativeai` package not installed. Run: `pip install google-generativeai`"

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    health = "Good" if mean_ndvi > 0.4 else ("Moderate" if mean_ndvi > 0.2 else "Poor")
    trend_text = ""
    if trend_delta is not None:
        if trend_delta > 0.05:
            trend_text = f"The crop health has IMPROVED over the last 6 months (NDVI rose by {trend_delta:+.3f})."
        elif trend_delta < -0.05:
            trend_text = f"The crop health has DECLINED over the last 6 months (NDVI dropped by {trend_delta:+.3f})."
        else:
            trend_text = f"The crop health has been STABLE over the last 6 months (NDVI change: {trend_delta:+.3f})."

    prompt = f"""You are an Indian Agricultural Expert (Kisan Mitra).
Analyze the satellite data below and give practical farming advice.
CRITICAL: Output your ENTIRE response in {selected_language}.
Use simple, non-technical words that a farmer understands.

Satellite Data Summary:
- Average NDVI (Vegetation Index): {mean_ndvi:.3f}
- Maximum NDVI: {max_ndvi:.3f}
- Minimum NDVI: {min_ndvi:.3f}
- Overall Health Rating: {health}
- 6-Month Trend: {trend_text if trend_text else 'Not available'}

Please provide:
1. A simple explanation of what these numbers mean for the farmer's crops.
2. Specific actionable advice (irrigation, fertilizer, pest watch, etc.).
3. Any warnings or things to watch out for.
4. Encouraging words for the farmer.
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as exc:
        return f"⚠️ AI analysis failed: {exc}"


# ═══════════════════════════════════════════════════════════════════
# NDVI HELPERS
# ═══════════════════════════════════════════════════════════════════

def calc_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denom = nir + red
    ndvi = np.where(denom > 0, (nir - red) / denom, np.nan)
    return ndvi


def make_true_color(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """Stack and scale to 0-1 for display."""
    rgb = np.dstack([red, green, blue]).astype(float)
    # Sentinel-2 L2A reflectance in [0, 10 000]. Clip & normalise.
    p2, p98 = np.nanpercentile(rgb, [2, 98])
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-10), 0, 1)
    return rgb


# ═══════════════════════════════════════════════════════════════════
# UI — SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/satellite-signal.png",
        width=60,
    )
    st.markdown("## 🛰️ Krishi Rakshak")
    st.markdown("**Satellite Edition**")
    st.divider()

    # ── Language selector ───────────────────────────────────────────
    selected_language = st.selectbox(
        "🌐 Language / भाषा",
        options=LANGUAGES,
        index=0,
        help="All AI-generated advice will be in this language.",
    )
    st.divider()

    # ── Gemini API key ─────────────────────────────────────────────
    gemini_api_key = st.text_input(
        "🔑 Gemini API Key (optional)",
        type="password",
        help="Needed for AI crop advice. Get a free key at https://aistudio.google.com/app/apikey",
    )
    st.divider()

    st.markdown(
        "1. **Draw** a polygon over your farm on the map.\n"
        "2. The app fetches **Sentinel-2** imagery (last 6 months).\n"
        "3. View **True-Color** image and **NDVI Crop-Health** heatmap.\n"
        "4. Track crop health over time with the **Growth Chart**.\n"
        "5. Get **AI advice** from Kisan Mitra (requires Gemini key)."
    )
    st.divider()
    resolution = st.select_slider(
        "Spatial resolution (m/px)",
        options=[10, 20, 30, 60],
        value=10,
        help="Lower = sharper but slower to download.",
    )
    max_images_for_chart = st.slider(
        "Max images for Growth Chart",
        min_value=3,
        max_value=30,
        value=10,
        help="More images = richer chart but slower.",
    )
    st.divider()
    st.caption("Data: Microsoft Planetary Computer · Sentinel-2 L2A (ESA/Copernicus)")


# ═══════════════════════════════════════════════════════════════════
# GPS SESSION STATE
# ═══════════════════════════════════════════════════════════════════
if "map_center" not in st.session_state:
    st.session_state.map_center = DEFAULT_CENTER
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = DEFAULT_ZOOM

# ═══════════════════════════════════════════════════════════════════
# UI — MAIN AREA
# ═══════════════════════════════════════════════════════════════════

dynamic_title = TITLE_MAP.get(selected_language, TITLE_MAP["English"])
dynamic_subtitle = SUBTITLE_MAP.get(selected_language, SUBTITLE_MAP["English"])
st.markdown(
    f"<h1 style='text-align:center;'>{dynamic_title}</h1>"
    f"<p style='text-align:center;color:gray;'>{dynamic_subtitle}</p>",
    unsafe_allow_html=True,
)

# ── GPS "Find My Farm" ─────────────────────────────────────────────
gps_col1, gps_col2 = st.columns([1, 3])
with gps_col1:
    find_farm = st.button("📍 Find My Farm", use_container_width=True)
with gps_col2:
    gps_placeholder = st.empty()

if find_farm:
    with st.spinner("Requesting GPS location from your browser…"):
        loc = get_geolocation()
    if loc and isinstance(loc, dict) and "coords" in loc:
        lat = loc["coords"]["latitude"]
        lon = loc["coords"]["longitude"]
        st.session_state.map_center = [lat, lon]
        st.session_state.map_zoom = 16
        gps_placeholder.success(f"📍 Located! Lat: {lat:.5f}, Lon: {lon:.5f} — Map centered on your position.")
    else:
        gps_placeholder.warning(
            "Could not get GPS location. Make sure you allow location access "
            "in your browser and are using HTTPS."
        )

# ── Interactive map ─────────────────────────────────────────────────
m = folium.Map(
    location=st.session_state.map_center,
    zoom_start=st.session_state.map_zoom,
    tiles="OpenStreetMap",
)

# If GPS was used, add a marker at the user's location
if st.session_state.map_center != DEFAULT_CENTER:
    folium.Marker(
        location=st.session_state.map_center,
        popup="📍 Your Location",
        icon=folium.Icon(color="red", icon="user", prefix="fa"),
    ).add_to(m)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Satellite Basemap",
).add_to(m)
folium.LayerControl().add_to(m)

draw = folium.plugins.Draw(
    draw_options={
        "polyline": False,
        "rectangle": True,
        "circle": False,
        "circlemarker": False,
        "marker": False,
        "polygon": True,
    },
    edit_options={"edit": True, "remove": True},
)
draw.add_to(m)

col_map, col_info = st.columns([3, 1])
with col_map:
    map_data = st_folium(m, width=None, height=520, returned_objects=["all_drawings"])

with col_info:
    st.markdown("### 📍 How to use")
    st.markdown(
        "- Click the **polygon** or **rectangle** tool on the map's left toolbar.\n"
        "- Draw a shape over the farm area you want to analyse.\n"
        "- Results appear below automatically."
    )

# ── Extract drawn polygon ──────────────────────────────────────────
polygon_geojson = None
drawn_features = []

if map_data and map_data.get("all_drawings"):
    drawn_features = map_data["all_drawings"]

if not drawn_features:
    st.warning("⚠️ Please draw a farm boundary on the map to begin.")
    st.stop()

# Take the last drawn polygon
last_feature = drawn_features[-1]
geom = last_feature.get("geometry")
if geom is None or geom["type"] not in ("Polygon", "MultiPolygon"):
    st.warning("⚠️ Please draw a valid polygon on the map.")
    st.stop()

polygon_geojson = geom
poly_shape = shape(polygon_geojson)
bbox = poly_shape.bounds  # (minx, miny, maxx, maxy)

st.success(f"✅ Farm boundary captured — Area ≈ {poly_shape.area * 12321:.1f} hectares (approx)")

# ── Date range (last 6 months) ─────────────────────────────────────
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=LOOKBACK_MONTHS * 30)
date_range = f"{start_date.isoformat()}/{end_date.isoformat()}"

# ═══════════════════════════════════════════════════════════════════
# STEP 1 — Search for satellite imagery
# ═══════════════════════════════════════════════════════════════════
st.divider()
st.markdown("## 🔭 Satellite Image Search")

items = search_sentinel_items(polygon_geojson, date_range)

if not items:
    st.info(
        "No cloud-free Sentinel-2 images found for this area in the last 6 months. "
        "Try drawing a larger area or increasing cloud-cover tolerance."
    )
    st.stop()

st.markdown(f"Found **{len(items)}** cloud-free scenes (< {MAX_CLOUD_COVER}% clouds) from the last {LOOKBACK_MONTHS} months.")

# Show a table of found scenes
scene_df = pd.DataFrame(
    {
        "Date": [i.datetime.strftime("%Y-%m-%d %H:%M") for i in items],
        "Cloud %": [round(i.properties.get("eo:cloud_cover", 0), 1) for i in items],
        "ID": [i.id for i in items],
    }
)
with st.expander("📋 View all found scenes", expanded=False):
    st.dataframe(scene_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# STEP 2 — Load & analyse the clearest recent image
# ═══════════════════════════════════════════════════════════════════
st.divider()
st.markdown("## 🖼️ Crop Health Analysis (Latest Clear Image)")

best_item = items[0]  # already sorted newest-first, <10% cloud
st.markdown(
    f"**Selected scene:** `{best_item.id}` — "
    f"*{best_item.datetime.strftime('%d %B %Y')}* — "
    f"Cloud cover: {best_item.properties.get('eo:cloud_cover', '?')}%"
)

with st.spinner("Downloading satellite bands (Red, NIR, Blue, Green)…"):
    red, nir, blue, green = load_bands(best_item.id, bbox, resolution)

if red is None:
    st.error("Failed to load image data. Please try a different area or retry later.")
    st.stop()

ndvi = calc_ndvi(red, nir)
true_color = make_true_color(red, green, blue)

# ── Side-by-side visualisation ──────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📸 True-Color Image")
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.imshow(true_color)
    ax1.set_title("What the eye sees", fontsize=11)
    ax1.axis("off")
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

with col2:
    st.markdown("### 🌿 Crop-Health Heatmap (NDVI)")
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    im = ax2.imshow(ndvi, cmap=NDVI_CMAP, vmin=-0.1, vmax=0.9)
    ax2.set_title("NDVI — Red (stressed) → Green (healthy)", fontsize=11)
    ax2.axis("off")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="NDVI")
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

# ── Quick NDVI stats ────────────────────────────────────────────────
valid_ndvi = ndvi[~np.isnan(ndvi)]
if valid_ndvi.size > 0:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean NDVI", f"{np.mean(valid_ndvi):.3f}")
    c2.metric("Max NDVI", f"{np.max(valid_ndvi):.3f}")
    c3.metric("Min NDVI", f"{np.min(valid_ndvi):.3f}")
    c4.metric(
        "Health Rating",
        "🟢 Good" if np.mean(valid_ndvi) > 0.4 else ("🟡 Moderate" if np.mean(valid_ndvi) > 0.2 else "🔴 Poor"),
    )

# ═══════════════════════════════════════════════════════════════════
# STEP 2B — AI Crop Advice (Gemini)
# ═══════════════════════════════════════════════════════════════════
st.divider()
st.markdown("## 🤖 Kisan Mitra — AI Crop Advice")

if not gemini_api_key:
    st.info(
        "🔑 Enter a **Gemini API Key** in the sidebar to unlock AI-powered crop advice. "
        "Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey)."
    )
else:
    if valid_ndvi.size > 0:
        if st.button("🧠 Get AI Advice", use_container_width=True):
            with st.spinner("Kisan Mitra is analysing your field…"):
                advice = generate_ai_advice(
                    mean_ndvi=float(np.mean(valid_ndvi)),
                    max_ndvi=float(np.max(valid_ndvi)),
                    min_ndvi=float(np.min(valid_ndvi)),
                    trend_delta=None,  # will be updated after growth chart
                    selected_language=selected_language,
                    api_key=gemini_api_key,
                )
            st.markdown(advice)
    else:
        st.warning("No valid NDVI data to analyse.")

# ═══════════════════════════════════════════════════════════════════
# STEP 3 — Growth Chart (NDVI over time)
# ═══════════════════════════════════════════════════════════════════
st.divider()
st.markdown("## 📈 Growth Chart — Crop Health vs. Time")
st.caption(
    f"Computing average NDVI for up to {max_images_for_chart} scenes. "
    "This may take a minute…"
)

chart_item_ids = [i.id for i in items[:max_images_for_chart]]
ts_df = compute_ndvi_timeseries(chart_item_ids, bbox, resolution=20)

if ts_df.empty:
    st.info("Not enough data to build a growth chart. Try a larger area.")
else:
    # Altair chart
    chart = (
        alt.Chart(ts_df)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(format="%d %b %Y")),
            y=alt.Y("ndvi:Q", title="Average NDVI", scale=alt.Scale(domain=[-0.1, 0.9])),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
                alt.Tooltip("ndvi:Q", title="NDVI", format=".3f"),
            ],
        )
        .properties(
            title="🌱 Vegetation Health Trend (Last 6 Months)",
            height=360,
        )
        .configure_title(fontSize=16)
        .interactive()
    )

    # Colour band overlay (background zones)
    zones = pd.DataFrame(
        {
            "y": [0, 0.2, 0.4],
            "y2": [0.2, 0.4, 0.9],
            "label": ["Poor", "Moderate", "Healthy"],
            "color": ["#fddbc7", "#fff4c4", "#c7e9c0"],
        }
    )
    bands = (
        alt.Chart(zones)
        .mark_rect(opacity=0.25)
        .encode(
            y=alt.Y("y:Q"),
            y2="y2:Q",
            color=alt.Color("color:N", scale=None),
        )
    )

    st.altair_chart(bands + chart, use_container_width=True)

    # Trend interpretation
    if len(ts_df) >= 2:
        first_ndvi = ts_df.iloc[0]["ndvi"]
        last_ndvi = ts_df.iloc[-1]["ndvi"]
        delta = last_ndvi - first_ndvi
        if delta > 0.05:
            st.success(f"📈 Positive trend detected: NDVI rose by **{delta:+.3f}** — crop health is **improving**.")
        elif delta < -0.05:
            st.error(f"📉 Negative trend detected: NDVI dropped by **{delta:+.3f}** — crop health may be **declining**.")
        else:
            st.info(f"➡️ Stable crop health over this period (Δ NDVI = {delta:+.3f}).")

# ── Footer ──────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.85em;'>"
    "Krishi Rakshak — Satellite Edition · Powered by "
    "<a href='https://planetarycomputer.microsoft.com/'>Microsoft Planetary Computer</a> "
    "& Sentinel-2 (ESA/Copernicus)"
    "</div>",
    unsafe_allow_html=True,
)
