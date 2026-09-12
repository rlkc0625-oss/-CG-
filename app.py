import io
import time
import base64
import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="図面→内観CGメーカー", page_icon="🏠")

st.title("🏠 図面→内観CGメーカー")
st.caption("無料のAI Hordeを使って試作するVer.1")

uploaded = st.file_uploader(
    "① 図面をアップロード",
    type=["png", "jpg", "jpeg", "webp"]
)

view = st.selectbox(
    "② 視点",
    ["LDK全体", "リビング側からキッチンを見る",
     "キッチン側からリビングを見る", "ダイニング側からLDKを見る",
     "玄関ホール"]
)

style = st.selectbox(
    "③ 内装テイスト",
    ["ナチュラル", "グレージュモダン", "ホテルライク",
     "北欧", "和モダン", "シンプルモダン"]
)

lighting = st.selectbox("④ 光", ["昼", "夕方", "夜"])

extra = st.text_area(
    "⑤ 追加指示（任意）",
    placeholder="例：床は明るいオーク、壁は白、家具は少なめ"
)

if uploaded:
    st.image(uploaded, caption="アップロードした図面", use_container_width=True)

def submit_horde(image_bytes, prompt):
    # AI Horde img2img endpoint.
    # Anonymous key is intentionally used for the free prototype.
    url = "https://aihorde.net/api/v2/generate/async"
    headers = {
        "apikey": "0000000000",
        "Client-Agent": "floorplan-cg-app:1.0"
    }

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "prompt": prompt,
        "models": ["AlbedoBase XL (SDXL)"],
        "params": {
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "cfg_scale": 7,
            "n": 1,
            "denoising_strength": 0.45,
            "sampler_name": "k_euler_a"
        },
        "source_image": b64,
        "source_processing": "img2img",
        "nsfw": False,
        "censor_nsfw": True
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["id"]

def wait_for_result(request_id):
    status_url = f"https://aihorde.net/api/v2/generate/status/{request_id}"
    for _ in range(90):
        r = requests.get(status_url, headers={"Client-Agent": "floorplan-cg-app:1.0"}, timeout=30)
        r.raise_for_status()
        data = r.json()

        if data.get("done") and data.get("generations"):
            return data["generations"][0]["img"]

        time.sleep(4)

    raise TimeoutError("生成に時間がかかっています。もう一度お試しください。")

if st.button("✨ CGを生成", type="primary", disabled=uploaded is None):
    if not uploaded:
        st.warning("先に図面をアップロードしてください。")
        st.stop()

    source = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
    normalized = io.BytesIO()
    source.save(normalized, format="PNG")
    image_bytes = normalized.getvalue()

    prompt = f"""
photorealistic architectural interior visualization based on the uploaded floor plan.
Camera/view: {view}.
Interior style: {style}.
Lighting: {lighting}.
{extra}

CRITICAL FLOOR PLAN PRESERVATION:
Preserve the uploaded floor plan's walls, room arrangement, major openings,
doors, windows, kitchen position and circulation as faithfully as possible.
Do not invent new rooms, walls, windows or doors.
Do not change the basic layout.
Do not add logos, text, product names or watermarks.
Create a realistic residential interior view corresponding to the selected area.
Architectural visualization, realistic materials, natural proportions, high detail.
"""

    try:
        with st.spinner("AIに生成を依頼しています。無料版なので待ち時間が長い場合があります…"):
            request_id = submit_horde(image_bytes, prompt)
            image_b64 = wait_for_result(request_id)

        image_data = base64.b64decode(image_b64)
        result = Image.open(io.BytesIO(image_data)).convert("RGB")

        st.success("生成完了")
        st.image(result, caption="生成された内観CG", use_container_width=True)

        out = io.BytesIO()
        result.save(out, format="PNG")
        st.download_button(
            "📥 CGを保存",
            data=out.getvalue(),
            file_name="floorplan_cg.png",
            mime="image/png"
        )

    except requests.HTTPError as e:
        st.error(f"AI Hordeとの通信に失敗しました：{e}")
    except Exception as e:
        st.error(f"生成に失敗しました：{e}")

st.divider()
st.caption(
    "※これは無料検証用の試作版です。AI画像生成のため、図面の寸法・壁・窓・ドア等を完全に正確に保持するCAD/BIMではありません。"
)
