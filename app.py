import io
import time
import base64
import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="図面→内観CGメーカー", page_icon="🏠")

st.title("🏠 図面→内観CGメーカー")
st.caption("無料AI Horde版 Ver.3 — 匿名無料枠対応")

uploaded = st.file_uploader(
    "① 図面をアップロード",
    type=["png", "jpg", "jpeg", "webp"]
)

view = st.selectbox(
    "② 視点",
    ["LDK全体", "リビング側からキッチンを見る",
     "キッチン側からリビングを見る", "ダイニング側からLDKを見る",
     "玄関ホール"],
)

style = st.selectbox(
    "③ 内装テイスト",
    ["ナチュラル", "グレージュモダン", "ホテルライク",
     "北欧", "和モダン", "シンプルモダン"],
)

lighting = st.selectbox("④ 光", ["昼", "夕方", "夜"])

extra = st.text_area(
    "⑤ 追加指示（任意）",
    placeholder="例：床は明るいオーク、壁は白、家具は少なめ",
)

if uploaded:
    st.image(uploaded, caption="アップロードした図面", use_container_width=True)


def normalize_to_webp(uploaded_bytes):
    source = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")

    # Keep the uploaded drawing readable while preventing oversized phone photos.
    max_side = 1536
    if max(source.size) > max_side:
        ratio = max_side / max(source.size)
        source = source.resize(
            (max(64, int(source.width * ratio)),
             max(64, int(source.height * ratio))),
            Image.Resampling.LANCZOS,
        )

    out = io.BytesIO()
    source.save(out, format="WEBP", quality=90, method=6)
    return out.getvalue()


def submit_horde(image_bytes, prompt):
    url = "https://aihorde.net/api/v2/generate/async"
    headers = {
        "apikey": "0000000000",
        "Client-Agent": "floorplan-cg-app:3.0",
        "Content-Type": "application/json",
    }

    b64 = base64.b64encode(image_bytes).decode("ascii")

    # Anonymous AI Horde requests over 576x576 require Kudos upfront.
    # Use 576x576 so the free anonymous path can be tested without Kudos.
    payload = {
        "prompt": prompt,
        "params": {
            "width": 576,
            "height": 576,
            "steps": 15,
            "n": 1,
            "cfg_scale": 6.5,
            "denoising_strength": 0.35,
        },
        "nsfw": False,
        "censor_nsfw": True,
        "source_image": b64,
        "source_processing": "img2img",
        "r2": True,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)

    if not response.ok:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise RuntimeError(f"AI Horde {response.status_code}: {detail}")

    return response.json()["id"]


def wait_for_result(request_id):
    check_url = f"https://aihorde.net/api/v2/generate/check/{request_id}"
    status_url = f"https://aihorde.net/api/v2/generate/status/{request_id}"

    for _ in range(120):
        check = requests.get(
            check_url,
            headers={"Client-Agent": "floorplan-cg-app:3.0"},
            timeout=30,
        )
        check.raise_for_status()
        if check.json().get("done"):
            break
        time.sleep(5)
    else:
        raise TimeoutError("無料AIの待ち時間が長いためタイムアウトしました。もう一度お試しください。")

    status = requests.get(
        status_url,
        headers={"Client-Agent": "floorplan-cg-app:3.0"},
        timeout=30,
    )
    status.raise_for_status()

    generations = status.json().get("generations") or []
    if not generations:
        raise RuntimeError("生成は完了しましたが画像が返りませんでした。")

    image_value = generations[0].get("img")
    if not image_value:
        raise RuntimeError("生成画像データが見つかりませんでした。")

    if image_value.startswith(("http://", "https://")):
        r = requests.get(image_value, timeout=60)
        r.raise_for_status()
        return r.content

    return base64.b64decode(image_value)


if st.button("✨ CGを生成", type="primary", disabled=uploaded is None):
    if not uploaded:
        st.warning("先に図面をアップロードしてください。")
        st.stop()

    prompt = f"""
photorealistic architectural interior visualization based on the uploaded Japanese residential floor plan.
Camera/view: {view}.
Interior style: {style}.
Lighting: {lighting}.
Additional instructions: {extra}

Use the uploaded floor plan as the structural reference.
Preserve existing walls, room arrangement, major doors, windows/openings,
kitchen position and circulation as faithfully as possible.
Do not invent new rooms, walls, doors or windows.
Do not change the basic layout.
Do not add logos, text, product names or watermarks.
Create a realistic residential interior view of the selected area.
"""

    try:
        with st.spinner("無料AIに生成を依頼しています。混雑時は時間がかかります…"):
            webp_bytes = normalize_to_webp(uploaded.getvalue())
            request_id = submit_horde(webp_bytes, prompt)
            result_bytes = wait_for_result(request_id)

        result = Image.open(io.BytesIO(result_bytes)).convert("RGB")
        st.success("生成完了")
        st.image(result, caption="生成された内観CG", use_container_width=True)

        output = io.BytesIO()
        result.save(output, format="PNG")
        st.download_button(
            "📥 CGを保存",
            data=output.getvalue(),
            file_name="floorplan_cg.png",
            mime="image/png",
        )

    except Exception as e:
        st.error(f"生成に失敗しました：{e}")

st.divider()
st.caption(
    "※無料検証版です。AI画像生成のため、CAD/BIMのように寸法・壁・窓・ドアを完全一致させるものではありません。"
)
