"""
Pixel Mosaic — Gallery Edition

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import time

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from pixel_mosaic.config import MosaicConfig
from pixel_mosaic.image_io import compute_target_size
from pixel_mosaic.palette import ANCHOR_PALETTES, generate_palette
from pixel_mosaic.solver_hungarian import solve_hungarian

# -- Page config -------------------------------------------------------
st.set_page_config(
    page_title="Pixel Mosaic",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

_DEFAULTS = MosaicConfig()
_SEED = 42

# -- CSS ---------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,300;1,400&family=Inter:wght@200;300;400;500&display=swap');

    .stApp {
        background-color: #faf9f6;
        color: #2a2a2a;
        font-family: 'Inter', 'Helvetica Neue', sans-serif;
    }
    .block-container {
        max-width: 1000px;
        padding-top: 3.5rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 300;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #2a2a2a;
    }

    .gallery-title {
        font-family: 'Cormorant Garamond', 'Georgia', serif;
        font-size: 2.8rem;
        font-weight: 300;
        letter-spacing: 0.06em;
        text-transform: none;
        text-align: center;
        color: #1a1a1a;
        border-bottom: 1px solid #1a1a1a;
        padding-bottom: 0.6rem;
        margin-bottom: 0.5rem;
    }
    .gallery-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 300;
        color: #1a1a1a;
        letter-spacing: 0.04em;
        line-height: 1.8;
        margin-bottom: 3.5rem;
    }
    .slider-desc {
        font-family: 'Cormorant Garamond', 'Georgia', serif;
        font-size: 0.95rem;
        font-weight: 300;
        font-style: italic;
        color: #6a6a64;
        line-height: 1.6;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }

    .label-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.6rem;
        font-weight: 400;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: #2a2a2a;
        margin-bottom: 0.2rem;
    }
    .label-detail {
        font-family: 'Cormorant Garamond', 'Georgia', serif;
        font-size: 0.85rem;
        font-weight: 300;
        font-style: italic;
        color: #a0a09a;
        margin-bottom: 0.8rem;
        line-height: 1.5;
        text-align: center;
    }

    .catalogue-line {
        font-family: 'Cormorant Garamond', 'Georgia', serif;
        font-size: 1.1rem;
        font-weight: 400;
        font-style: italic;
        color: #2a2a2a;
        text-align: center;
        margin-top: 0.8rem;
        margin-bottom: 0.15rem;
    }
    .catalogue-detail {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 400;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        text-align: center;
        color: #2a2a2a;
        margin-top: -1.8rem;
        margin-bottom: 2rem;
    }

    .passepartout {
        background: #faf9f6;
        border: 1px solid #e0ded8;
        padding: 16px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.04);
        margin-bottom: 0.6rem;
    }
    .passepartout img { display: block; width: 100%; }

    /* Remove all border-radius */
    .stButton > button, .stDownloadButton > button,
    .stFileUploader > div, .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    div[data-testid="stFileUploader"],
    div[data-testid="stFileUploader"] > div,
    div[data-testid="stFileUploader"] section,
    div[data-testid="stFileUploader"] button,
    .stAlert {
        border-radius: 0px !important;
    }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        background-color: transparent !important;
        color: #2a2a2a !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 0px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 400 !important;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        font-size: 0.6rem;
        padding: 0.8rem 1.5rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #2a2a2a !important;
        color: #faf9f6 !important;
    }
    /* Download button: black bg, white text (like COMPOSE) */
    .stDownloadButton > button {
        background-color: #2a2a2a !important;
        color: #faf9f6 !important;
    }
    .stDownloadButton > button:hover {
        background-color: transparent !important;
        color: #2a2a2a !important;
    }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background-color: #2a2a2a !important;
        color: #faf9f6 !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background-color: transparent !important;
        color: #2a2a2a !important;
    }

    /* File uploader */
    div[data-testid="stFileUploader"] {
        border: 1px solid #e0ded8;
        padding: 1.5rem;
        background: transparent;
    }
    div[data-testid="stFileUploader"] small {
        color: #a0a09a;
        font-size: 0.65rem;
    }
    /* Center the "Select artwork" label */
    .stFileUploader label {
        text-align: center !important;
        display: block !important;
    }
    /* File uploader buttons: same size, black bg, white text */
    div[data-testid="stFileUploader"] button {
        background-color: #2a2a2a !important;
        color: #faf9f6 !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 0px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 400 !important;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        font-size: 0.6rem !important;
        padding: 0.8rem 1.5rem !important;
        min-height: 2.4rem !important;
        min-width: 2.4rem !important;
        box-sizing: border-box !important;
        transition: all 0.2s ease;
    }
    div[data-testid="stFileUploader"] button:hover {
        background-color: transparent !important;
        color: #2a2a2a !important;
    }

    /* Labels */
    .stSelectbox label, .stNumberInput label,
    .stFileUploader label, .stCheckbox label,
    .stSlider label, .stTextInput label {
        font-size: 0.6rem !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #999 !important;
    }

    /* Sliders: all black, no orange */
    /* Thumb (the draggable point) */
    .stSlider [role="slider"],
    .stSlider [role="slider"]:focus,
    .stSlider [role="slider"]:hover {
        background-color: #2a2a2a !important;
        border-color: #2a2a2a !important;
        box-shadow: none !important;
    }
    /* Current value above thumb */
    .stSlider div[data-testid="stThumbValue"] {
        color: #2a2a2a !important;
        background: transparent !important;
    }
    /* Track background (full width behind the fill) */
    .stSlider div[data-baseweb="slider"] > div > div:first-child {
        background-color: #d0cec8 !important;
    }
    /* Active track fill (from min to thumb — follows the point) */
    .stSlider div[data-baseweb="slider"] > div > div:nth-child(2) {
        background-color: #2a2a2a !important;
    }
    /* Tick bar: plain text, no boxes at all */
    .stSlider [data-testid="stTickBar"],
    .stSlider [data-testid="stTickBar"] * {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    .stSlider [data-testid="stTickBar"] > div {
        color: #a0a09a !important;
        font-size: 0.6rem !important;
        padding: 0 !important;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: transparent;
        border: 1px solid #e0ded8;
        padding: 1rem 1.2rem;
    }
    div[data-testid="stMetric"] label {
        font-family: 'Inter', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.5rem;
        color: #a0a09a;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-family: 'Cormorant Garamond', serif !important;
        font-weight: 300;
        font-size: 1.4rem;
        color: #2a2a2a;
    }

    hr { border: none; border-top: 1px solid #e0ded8; margin: 2.5rem 0; }

    .wall-text {
        font-family: 'Cormorant Garamond', 'Georgia', serif;
        font-size: 0.95rem;
        font-weight: 300;
        font-style: italic;
        line-height: 1.8;
        color: #a0a09a;
        max-width: 560px;
        margin: 4rem auto 0;
        padding-top: 2rem;
        border-top: 1px solid #e0ded8;
        text-align: center;
    }

    .processing-text {
        font-family: 'Cormorant Garamond', serif;
        font-size: 1rem;
        font-weight: 300;
        font-style: italic;
        color: #a0a09a;
        padding: 1.5rem 0;
    }

    /* Palette cards: ColorHunt-style */
    .palette-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, 90px);
        gap: 1rem;
        margin: 0.8rem 0 1.5rem;
    }
    .palette-card {
        width: 90px;
        cursor: pointer;
        border: 1px solid #e0ded8;
        transition: outline 0.2s ease;
        outline: 1px solid transparent;
        outline-offset: 3px;
    }
    .palette-card:hover {
        outline: 1px solid #a0a09a;
    }
    .palette-card-selected {
        outline: 3px solid #2a2a2a !important;
    }
    .palette-card-color {
        width: 100%;
        height: 22px;
    }

    /* Palette card click overlay: transparent button pulled over card */
    [class*="st-key-pal_"] {
        cursor: pointer;
    }
    [class*="st-key-pal_"] .stButton {
        margin-top: -90px !important;
        position: relative !important;
        z-index: 10 !important;
        pointer-events: auto !important;
    }
    [class*="st-key-pal_"] .stButton > button {
        width: 100% !important;
        height: 90px !important;
        background: transparent !important;
        color: transparent !important;
        border: none !important;
        padding: 0 !important;
        min-height: 0 !important;
        cursor: pointer !important;
        box-shadow: none !important;
        outline: none !important;
        pointer-events: auto !important;
    }
    [class*="st-key-pal_"] .stButton > button:hover,
    [class*="st-key-pal_"] .stButton > button:focus,
    [class*="st-key-pal_"] .stButton > button:active {
        background: transparent !important;
        color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* Hide streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# -- Helpers -----------------------------------------------------------

def _add_passepartout(img: Image.Image, border: int = 20) -> Image.Image:
    w, h = img.size
    bg = (250, 249, 246)
    canvas = Image.new("RGB", (w + border * 2, h + border * 2), bg)
    canvas.paste(img, (border, border))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        [border - 1, border - 1, border + w, border + h],
        outline=(224, 222, 216), width=1,
    )
    return canvas


# -- Title -------------------------------------------------------------
st.markdown(
    '<div class="gallery-title">Mosaic Creator</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="gallery-subtitle">'
    "Upload any image and this app will recreate it using only a fixed set of "
    "randomly generated colours \u2014 one unique colour per pixel, no duplicates. "
    "You choose a curated colour palette, and the implemented Hungarian method "
    "(also Kuhn\u2013Munkres algorithm) finds the mathematically optimal placement "
    "for every single colour to approximate your image as closely as possible. "
    "The result is a pixel mosaic that looks like your original photograph "
    "reinterpreted through a restricted palette, somewhere between pixel art "
    "and pointillism. Export the final piece as a high-resolution PNG."
    "</div>",
    unsafe_allow_html=True,
)

# -- Controls: just two sliders ----------------------------------------
ctrl1, ctrl2 = st.columns(2)
with ctrl1:
    max_side = st.slider("Max side (px)", 4, 100, _DEFAULTS.max_side)
    st.markdown(
        '<div class="slider-desc">'
        "Controls the resolution of the mosaic by setting the longest edge in "
        "pixels. Higher values produce more detailed mosaics but take "
        "significantly longer to compute; lower values create a coarser, "
        "more abstract look."
        "</div>",
        unsafe_allow_html=True,
    )
with ctrl2:
    upscale = st.slider("Upscale", 4, 20, _DEFAULTS.pixel_upscale)
    st.markdown(
        '<div class="slider-desc">'
        "Enlarges the output image by rendering each mosaic pixel as a block "
        "of this size. Higher values give you a larger export file with crisp "
        "pixel edges; lower values produce a smaller image."
        "</div>",
        unsafe_allow_html=True,
    )

# -- Palette selection via clickable cards -----------------------------
st.markdown(
    '<div class="label-title" style="text-align:center;font-size:0.9rem;margin-top:1.8rem;">'
    "Choose a colour palette</div>",
    unsafe_allow_html=True,
)

if "selected_anchor" not in st.session_state:
    st.session_state.selected_anchor = "ember"

anchor_names = list(ANCHOR_PALETTES.keys())
cols_per_row = 8

for row_start in range(0, len(anchor_names), cols_per_row):
    row_names = anchor_names[row_start : row_start + cols_per_row]
    cols = st.columns(cols_per_row)
    for i, aname in enumerate(row_names):
        hexes = ANCHOR_PALETTES[aname][0]
        is_selected = st.session_state.selected_anchor == aname
        sel_class = " palette-card-selected" if is_selected else ""
        card_html = (
            f'<div class="palette-card{sel_class}">'
            + "".join(
                f'<div class="palette-card-color" style="background:{c};"></div>'
                for c in hexes
            )
            + "</div>"
        )
        with cols[i]:
            with st.container(key=f"pal_{aname}"):
                st.markdown(card_html, unsafe_allow_html=True)
                if st.button(" ", key=f"a_{aname}", use_container_width=True):
                    st.session_state.selected_anchor = aname
                    st.rerun()

palette_mode = st.session_state.selected_anchor
st.markdown('<div style="margin-top:1.4rem;"></div>', unsafe_allow_html=True)
spread = st.slider("Spread", 1.0, 40.0, ANCHOR_PALETTES[palette_mode][1], step=1.0)
st.markdown(
    '<div class="slider-desc">'
    "Controls how far the generated colours deviate from the chosen anchor "
    "palette in perceptual colour space. Higher values introduce more colour "
    "variety and unexpected tones; lower values keep colours tightly clustered "
    "around the original anchor palette."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# -- Upload ------------------------------------------------------------
uploaded = st.file_uploader(
    "Select artwork", type=["jpg", "jpeg", "png", "webp", "bmp", "jfif"],
)

# Persist upload in session state so palette changes don't clear it
if uploaded is not None:
    st.session_state.uploaded_data = uploaded.getvalue()
elif "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None

if st.session_state.uploaded_data is not None:
    original = Image.open(io.BytesIO(st.session_state.uploaded_data)).convert("RGB")
    w, h = compute_target_size(original.width, original.height, max_side)
    num_pixels = w * h

    target_img = original.resize((w, h), Image.LANCZOS)
    target = np.array(target_img, dtype=np.uint8)

    palette = generate_palette(
        num_pixels, mode=palette_mode, seed=_SEED, spread=spread,
    )

    # Compose
    if st.button("COMPOSE", type="primary", use_container_width=True):
        target_flat = target.reshape(-1, 3)

        progress = st.empty()
        progress.markdown(
            '<div class="processing-text">Composing ...</div>',
            unsafe_allow_html=True,
        )

        t0 = time.perf_counter()
        mosaic_flat = solve_hungarian(palette, target_flat, color_space="lab")
        elapsed = time.perf_counter() - t0
        progress.empty()

        mosaic = mosaic_flat.reshape(h, w, 3)

        t_f = target.reshape(-1, 3).astype(np.float64)
        m_f = mosaic.reshape(-1, 3).astype(np.float64)
        error = float(np.mean(np.sqrt(np.sum((t_f - m_f) ** 2, axis=1))))

        st.markdown("---")

        # Main artwork
        mosaic_display = Image.fromarray(mosaic).resize(
            (w * upscale, h * upscale), Image.NEAREST,
        )
        framed = _add_passepartout(mosaic_display, border=28)
        st.image(framed, use_container_width=True)

        st.markdown(
            f'<div class="catalogue-detail">'
            f"{w} &times; {h}, {palette_mode} palette, "
            f"{num_pixels:,} colours, 2026"
            f"</div>",
            unsafe_allow_html=True,
        )

        buf = io.BytesIO()
        mosaic_display.save(buf, format="PNG")
        _, dl_col, _ = st.columns([1, 2, 1])
        with dl_col:
            st.download_button(
                "SAVE ART",
                data=buf.getvalue(),
                file_name="pixel_mosaic.png",
                mime="image/png",
                use_container_width=True,
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Resolution", f"{w} \u00d7 {h}")
        m2.metric("Colours", f"{num_pixels:,}")
        m3.metric("Time", f"{elapsed:.1f} s")
        m4.metric("Avg Error", f"{error:.1f}")

        # Input Images
        st.markdown("")
        st.markdown("")
        st.markdown(
            '<div class="label-title" style="text-align:center;font-size:0.9rem;">Input Images</div>',
            unsafe_allow_html=True,
        )

        doc1, doc2, doc3 = st.columns(3)
        with doc1:
            st.image(
                _add_passepartout(
                    original.resize((w * upscale, h * upscale), Image.LANCZOS),
                    border=12,
                ),
                use_container_width=True,
            )
            st.markdown(
                '<div class="label-detail">Source</div>',
                unsafe_allow_html=True,
            )
        with doc2:
            st.image(
                _add_passepartout(
                    target_img.resize((w * upscale, h * upscale), Image.NEAREST),
                    border=12,
                ),
                use_container_width=True,
            )
            st.markdown(
                f'<div class="label-detail">{w} &times; {h}</div>',
                unsafe_allow_html=True,
            )
        with doc3:
            st.image(
                _add_passepartout(
                    Image.fromarray(palette.reshape(h, w, 3)).resize(
                        (w * upscale, h * upscale), Image.NEAREST,
                    ),
                    border=12,
                ),
                use_container_width=True,
            )
            st.markdown(
                f'<div class="label-detail">{palette_mode}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

    else:
        # Preview
        st.markdown(
            '<div class="label-title" style="margin-top:0.5rem;text-align:center;font-size:0.9rem;">Input Images</div>',
            unsafe_allow_html=True,
        )
        prev1, prev2, prev3 = st.columns(3)
        with prev1:
            st.image(original, use_container_width=True)
            st.markdown(
                '<div class="label-detail">Source</div>',
                unsafe_allow_html=True,
            )
        with prev2:
            st.image(
                target_img.resize((w * upscale, h * upscale), Image.NEAREST),
                use_container_width=True,
            )
            st.markdown(
                f'<div class="label-detail">{w} &times; {h}</div>',
                unsafe_allow_html=True,
            )
        with prev3:
            st.image(
                Image.fromarray(palette.reshape(h, w, 3)).resize(
                    (w * upscale, h * upscale), Image.NEAREST,
                ),
                use_container_width=True,
            )
            st.markdown(
                f'<div class="label-detail">{palette_mode}</div>',
                unsafe_allow_html=True,
            )

else:
    st.markdown(
        '<p style="font-family: Cormorant Garamond, Georgia, serif; '
        "color: #bbb; font-size: 1rem; font-weight: 300; "
        'font-style: italic; margin-top: 2rem;">'
        "Select an artwork to begin.</p>",
        unsafe_allow_html=True,
    )

# Wall text
st.markdown(
    '<div class="wall-text">'
    "Each mosaic is constructed from a fixed set of randomly generated colours, "
    "one per pixel. No colours are added or modified \u2014 only their positions "
    "change. The arrangement is computed using the Hungarian algorithm, which "
    "finds the mathematically optimal assignment by minimising perceptual "
    "colour distance in CIELAB space."
    "</div>",
    unsafe_allow_html=True,
)
