"""
FaceSim – Celebrity Look-Alike Web App
Run with:  streamlit run src/app.py
"""

import sys
import os
import io
import tempfile
import requests
import urllib.parse
import base64
from google import genai
from pathlib import Path
from dotenv import load_dotenv
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

import streamlit as st
from PIL import Image
from src.interface import get_matches

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceSim – Find Your Celebrity Twin",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS  – purple / lavender cinematic theme
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@700;800&display=swap');

/* ── Root & Background ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #f0eeff;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { display: none; }
footer { visibility: hidden; }

/* ── Nav bar ── */
.nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 48px;
    background: rgba(255,255,255,0.6);
    backdrop-filter: blur(12px);
    border-radius: 0 0 24px 24px;
    margin-bottom: 32px;
    box-shadow: 0 2px 24px rgba(110,86,207,0.07);
}
.nav-logo {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 800;
    color: #4f32c8;
    letter-spacing: -0.5px;
}
.nav-links { display: flex; gap: 32px; }
.nav-links a {
    text-decoration: none;
    color: #6b7280;
    font-size: 0.875rem;
    font-weight: 500;
    transition: color .2s;
}
.nav-links a:hover, .nav-links a.active { color: #4f32c8; }

/* ── Hero ── */
.hero { text-align: center; padding: 8px 16px 32px; }
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.4rem, 5vw, 3.6rem);
    font-weight: 800;
    color: #4f32c8;
    margin: 0 0 12px;
    letter-spacing: -1px;
}
.hero p {
    color: #6b7280;
    font-size: 1.05rem;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Upload zone ── */
.upload-card {
    background: white;
    border-radius: 24px;
    padding: 40px;
    box-shadow: 0 4px 32px rgba(110,86,207,0.10);
    border: 2px dashed #c4b5fd;
    text-align: center;
    transition: border-color .3s, box-shadow .3s;
}
.upload-card:hover {
    border-color: #7c3aed;
    box-shadow: 0 8px 40px rgba(110,86,207,0.18);
}

/* ── Section header ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #1f1f2e;
    margin: 0 0 4px;
}
.section-sub {
    color: #9ca3af;
    font-size: 0.875rem;
    margin-bottom: 20px;
}

/* ── Match card ── */
.match-card {
    background: white;
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(110,86,207,0.12);
    position: relative;
}
.match-card img { width: 100%; display: block; border-radius: 20px 20px 0 0; }
.match-label {
    position: absolute;
    top: 16px;
    left: 16px;
    background: rgba(79,50,200,0.90);
    backdrop-filter: blur(6px);
    color: white;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.5px;
}
.match-info {
    padding: 14px 16px;
}
.match-name {
    font-weight: 700;
    font-size: 1.05rem;
    color: #1f1f2e;
}
.match-role {
    color: #9ca3af;
    font-size: 0.8rem;
    margin-top: 2px;
}

/* ── Score ring ── */
.score-ring-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
}
.score-ring {
    width: 120px; height: 120px;
    border-radius: 50%;
    border: 5px solid #7c3aed;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    background: white;
    box-shadow: 0 0 0 8px #ede9fe;
}
.score-number {
    font-size: 2rem;
    font-weight: 800;
    color: #4f32c8;
    line-height: 1;
}
.score-pct { font-size: 0.85rem; font-weight: 600; color: #7c3aed; }
.score-label {
    font-size: 0.7rem;
    letter-spacing: 1.5px;
    color: #9ca3af;
    font-weight: 600;
    text-transform: uppercase;
}
.vs-arrow {
    font-size: 1.4rem;
    color: #c4b5fd;
    animation: pulse 1.8s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ── Gallery grid ── */
.gallery-img {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(110,86,207,0.13);
    transition: transform .25s, box-shadow .25s;
}
.gallery-img:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 32px rgba(110,86,207,0.22);
}

/* ── Secondary match chip ── */
.secondary-card {
    background: white;
    border-radius: 16px;
    padding: 14px 16px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 14px;
    box-shadow: 0 3px 16px rgba(110,86,207,0.09);
    transition: transform .2s, box-shadow .2s;
}
.secondary-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 28px rgba(110,86,207,0.16);
}
.secondary-badge {
    background: linear-gradient(135deg, #7c3aed, #a78bfa);
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 20px;
    white-space: nowrap;
}
.secondary-name { font-weight: 700; font-size: 0.95rem; color: #1f1f2e; margin-bottom: 4px; }

/* ── Helper ── */
.divider { height:1px; background:#e5e7eb; margin: 36px 0; border:none; }
.tag {
    display: inline-block;
    background: #ede9fe;
    color: #7c3aed;
    font-size: 0.72rem;
    font-weight: 700;
    border-radius: 20px;
    padding: 3px 10px;
    letter-spacing: 0.3px;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Nav bar
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav-bar">
    <div class="nav-logo">✨ FaceSim</div>
    <div class="nav-links">
        <a href="#" class="active">Discover</a>
        <a href="#">Gallery</a>
        <a href="#">About</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Hero
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Find Your Celebrity Twin.</h1>
    <p>Upload your photo and our AI will find which famous person you look most like — powered by FaceNet deep embeddings.</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def get_image_base64(path: str) -> str:
    """Read a local image and return a base64 data URI."""
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded_string}"


@st.cache_data(show_spinner=False)
def batch_generate_summaries(names: list[str]) -> dict:
    """Generate multiple biographies in a single Vertex AI JSON request."""
    try:
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        if project:
            client = genai.Client(vertexai=True, project=project, location=location)
        else:
            client = genai.Client(vertexai=True)
            
        prompt = f"""
        You are an assistant. Provide data in strict JSON format. Do not use markdown wrappers.
        Return a JSON object where each key is a celebrity name from the list.
        For EVERY name in the list, provide a very short engaging description (max 15 words) highlighting their fame and what they do (e.g. "Famous actor known for superhero roles") in the "short_bio" field.
        Names to process: {json.dumps(names)}
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        
        # Defensive JSON extraction
        raw = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        return {}


def render_top_match_card(name: str, img_b64: str, bio_text: str) -> str:
    return f"""
        <div class="match-card">
            <img src="{img_b64}">
            <div class="match-label">✔ Top Match</div>
            <div class="match-info">
                <div class="match-name">{name}</div>
                <div class="match-role">Celebrity · Dataset match</div>
                <div style="margin-top: 12px; font-size: 0.85rem; color: #4b5563; line-height: 1.5; padding-top: 12px; border-top: 1px dashed #e5e7eb;">
                    {bio_text}
                </div>
            </div>
        </div>"""


def render_sec_card(name: str, dist: str, img_b64: str, bio_text: str) -> str:
    bio_html = f'<div style="font-size: 0.72rem; color: #6b7280; margin-top: 8px; line-height: 1.3;">{bio_text}</div>' if bio_text else ""
    return f"""
        <div class="secondary-card" style="align-items: flex-start;">
            <img src="{img_b64}" width="56" height="56" style="border-radius:10px; object-fit: cover; flex-shrink: 0;">
            <div style="flex-grow: 1;">
                <div class="secondary-name">{name}</div>
                <span class="secondary-badge">Dist: {dist}</span>
                {bio_html}
            </div>
        </div>"""


def google_image_urls(query: str, num: int = 5) -> list[str]:
    """
    Scrape Google Images for *query* and return up to *num* direct image URLs.
    Falls back to an empty list if scraping fails (e.g., blocked by Google).
    """
    search_url = (
        "https://www.google.com/search?tbm=isch&q="
        + urllib.parse.quote_plus(query)
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(search_url, headers=headers, timeout=8)
        resp.raise_for_status()
        # Pull raw image src= values via a simple regex (avoids heavy deps)
        import re
        # Google embeds thumbnails as base64 or as src URLs in <img> tags
        # Focus on the data-src / src of actual images (not icons / logos)
        urls = re.findall(r'"(https://[^"]+\.(?:jpg|jpeg|png|webp)[^"]*)"', resp.text)
        # Deduplicate while preserving order
        seen, clean = set(), []
        for u in urls:
            if u not in seen:
                seen.add(u)
                clean.append(u)
        return clean[:num]
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Upload section
# ──────────────────────────────────────────────────────────────────────────────
col_up, col_gap, col_hint = st.columns([1, 0.05, 1])

with col_up:
    uploaded_file = st.file_uploader(
        "Drop your photo here",
        type=["jpg", "jpeg", "png", "webp"],
        help="Best results with a clear, front-facing photo.",
        label_visibility="collapsed",
    )

with col_hint:
    if not uploaded_file:
        st.markdown("""
        <div style="background:white;border-radius:20px;padding:32px 28px;
                    box-shadow:0 4px 24px rgba(110,86,207,0.09);margin-top:4px">
            <div style="font-size:2.5rem;margin-bottom:12px">🎬</div>
            <div style="font-weight:700;font-size:1.1rem;color:#1f1f2e;margin-bottom:8px">
                How it works
            </div>
            <ol style="color:#6b7280;font-size:0.9rem;line-height:2;padding-left:18px">
                <li>Upload a clear face photo</li>
                <li>Our AI detects &amp; embeds your face</li>
                <li>We search thousands of celebrity faces</li>
                <li>See your match with similarity score!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────────────────────────
if uploaded_file:
    # Save upload to a temp file so PIL / MTCNN can read it
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    with st.spinner("🔍 Analysing your face… this may take a moment on first run."):
        matches = get_matches(tmp_path, k=13)

    os.unlink(tmp_path)   # clean up temp file

    # ── Error handling ────────────────────────────────────────────────────────
    if isinstance(matches, dict) and "error" in matches:
        st.error(f"⚠️ {matches['error']}")
        st.stop()

    top_match = matches[0]
    other_matches = matches[1:]

    top_name   = top_match["Name"]
    top_score  = f"{top_match['Similarity Score']:.4f}"
    top_img    = top_match["Image Path"]

    # ── Section: It's a Match ─────────────────────────────────────────────────
    st.markdown("""
    <div class="hero" style="padding:0 0 24px">
        <h1 style="font-size:2.6rem">It's a Match.</h1>
        <p>Our AI has found your closest celebrity counterpart with high precision.</p>
    </div>
    """, unsafe_allow_html=True)

    col_you, col_score, col_celeb = st.columns([5, 3, 5])

    with col_you:
        uploaded_file.seek(0)
        user_img_b64 = base64.b64encode(uploaded_file.read()).decode()
        st.markdown(f"""
        <div class="match-card">
            <img src="data:image/jpeg;base64,{user_img_b64}">
            <div class="match-info">
                <div class="match-name">You</div>
                <div class="match-role">Your uploaded photo</div>
            </div>
        </div>""", unsafe_allow_html=True)

    with col_score:
        st.markdown(f"""
        <div class="score-ring-container" style="height:100%;justify-content:center;min-height:220px">
            <div class="vs-arrow">◀ ▶</div>
            <div class="score-ring">
                <div class="score-number" style="font-size: 1.5rem;">{top_score}</div>
                <div class="score-pct">L2</div>
            </div>
            <div class="score-label">Distance</div>
        </div>
        """, unsafe_allow_html=True)

    with col_celeb:
        display_name = top_name.replace("_", " ").title()
        top_img_b64 = get_image_base64(top_img)
        
        # We place an empty container to render the loaded state FIRST
        top_match_placeholder = st.empty()
        top_match_placeholder.markdown(
            render_top_match_card(display_name, top_img_b64, "<i>✨ Generating AI insights...</i>"),
            unsafe_allow_html=True
        )

    # ── Section: Celebrity Gallery (Google Images) ────────────────────────────
    st.markdown(f"""
    <div class="section-title">{display_name} Gallery</div>
    <div class="section-sub">Explore photos of your primary match from around the web.</div>
    """, unsafe_allow_html=True)

    gallery_query = f"{display_name} celebrity photos"
    gallery_urls  = google_image_urls(gallery_query, num=5)

    if gallery_urls:
        g_cols = st.columns(len(gallery_urls))
        for gcol, url in zip(g_cols, gallery_urls):
            with gcol:
                try:
                    img_data = requests.get(url, timeout=6, headers={
                        "User-Agent": "Mozilla/5.0"
                    }).content
                    gcol.image(Image.open(io.BytesIO(img_data)), use_container_width=True)
                except Exception:
                    gcol.markdown(f"[🔗 View photo]({url})")
    else:
        # Fallback: show a Google search link
        search_link = f"https://www.google.com/search?tbm=isch&q={urllib.parse.quote_plus(display_name)}"
        st.markdown(f"""
        <div style="background:white;border-radius:16px;padding:24px 28px;
                    box-shadow:0 3px 16px rgba(110,86,207,0.08);text-align:center">
            <p style="color:#6b7280;margin-bottom:12px">
                Live image fetching was blocked — open Google Images instead.
            </p>
            <a href="{search_link}" target="_blank"
               style="background:linear-gradient(135deg,#7c3aed,#a78bfa);color:white;
                      padding:10px 24px;border-radius:20px;text-decoration:none;
                      font-weight:600;font-size:0.9rem">
               🔍 View {display_name} on Google Images →
            </a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Section: Other similar celebrities ───────────────────────────────────
    if other_matches:
        st.markdown("""
        <div class="section-title">Secondary Alignments</div>
        <div class="section-sub">Other celebrities with high biometric correlation to your photo.</div>
        """, unsafe_allow_html=True)

        # Render secondary matches in a responsive grid of 3 columns
        sec_placeholders = []
        num_cols = 3
        for i in range(0, len(other_matches), num_cols):
            # We enforce exactly 3 columns per row (or fewer if we hit the end of the list)
            row_matches = other_matches[i:i+num_cols]
            cols = st.columns(num_cols)
            for col, sm in zip(cols, row_matches):
                sec_dist = f"{sm['Similarity Score']:.4f}"
                sec_name = sm["Name"].replace("_", " ").title()
                sec_img  = sm["Image Path"]
                with col:
                    img_b64 = get_image_base64(sec_img)
                    sec_p = st.empty()
                    sec_placeholders.append((sec_p, sm, img_b64))
                    
                    # Display loading state
                    sec_p.markdown(
                        render_sec_card(sec_name, sec_dist, img_b64, "<i>Loading...</i>"),
                        unsafe_allow_html=True
                    )

    # ── Section: Load Multi-AI Data ───────────────────────────────────────────
    with st.spinner("🤖 Vertex AI is generating celebrity insights..."):
        names_to_fetch = [display_name] + [sm["Name"].replace("_", " ").title() for sm in other_matches]
        ai_data = batch_generate_summaries(names_to_fetch)

    # Update Top Placeholder
    top_bio = ai_data.get(display_name, {}).get("short_bio", "Bio information unavailable.")
    top_match_placeholder.markdown(
        render_top_match_card(display_name, top_img_b64, top_bio),
        unsafe_allow_html=True
    )
    
    # Update Secondary Placeholders
    for sec_p, sm, sb_img in sec_placeholders:
        s_name = sm["Name"].replace("_", " ").title()
        s_dist = f"{sm['Similarity Score']:.4f}"
        s_bio = ai_data.get(s_name, {}).get("short_bio", "Information unavailable.")
        sec_p.markdown(render_sec_card(s_name, s_dist, sb_img, s_bio), unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:48px 0 24px;color:#9ca3af;font-size:0.8rem">
        Powered by FaceNet · Built with Streamlit · For entertainment purposes only
    </div>
    """, unsafe_allow_html=True)
