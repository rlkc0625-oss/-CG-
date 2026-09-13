import os
import tempfile
from pathlib import Path

import streamlit as st
from gradio_client import Client, handle_file

st.set_page_config(page_title="図面→内観CG", page_icon="🏠", layout="centered")

SPACE = "InstantX/Qwen-Image-ControlNet"

st.title("🏠 図面 → 内観CG")
st.caption("無料Hugging Face / Qwen Image ControlNet版")

uploaded = st.file_uploader(
    "① 間取り図をアップロード",
    type=["png", "jpg", "jpeg", "webp"],
)

view = st.selectbox(
    "② 見たい場所",
    ["LDK", "リビング", "キッチン", "ダイニング", "玄関", "主寝室"],
)

style = st.selectbox(
    "③ インテリア",
    ["モダン和風", "ナチュラル", "グレージュモダン", "ホテルライク", "北欧"],
)

lighting = st.selectbox(
    "④ 光",
    ["昼・自然光", "夕方・暖色", "夜・間接照明"],
)

if uploaded:
    st.image(uploaded, caption="アップロードした間取り図", use_container_width=True)

if st.button("🚀 内観CGを生成", type="primary", use_container_width=True):
    if uploaded is None:
        st.error("先に間取り図をアップロードしてください。")
        st.stop()

    prompt = f"""
Photorealistic architectural interior visualization of a Japanese detached house.
Create an eye-level interior view of the {view}.
Interior style: {style}.
Lighting: {lighting}.
Use the uploaded floor plan as the structural reference.
Preserve the wall layout, room boundaries, doors, windows, openings, and kitchen position
as accurately as possible. Do not redesign or move architectural elements.
Show realistic Japanese residential materials, realistic furniture, accurate perspective,
natural proportions, premium architectural photography, highly realistic materials,
soft realistic shadows, professional interior photography.
Do not show the floor plan itself in the final image.
"""

    negative_prompt = (
        "floor plan, blueprint, top view, aerial view, sketch, line art, "
        "diagram, distorted walls, extra doors, missing windows, distorted furniture, "
        "text, watermark, logo, blurry, low quality"
    )

    hf_token = None
    try:
        hf_token = st.secrets.get("HF_TOKEN")
    except Exception:
        hf_token = None

    with tempfile.NamedTemporaryFile(
        suffix=Path(uploaded.name).suffix or ".png", delete=False
    ) as tmp:
        tmp.write(uploaded.getbuffer())
        image_path = tmp.name

    try:
        with st.spinner("無料AIで内観CGを生成しています…"):
            client = Client(SPACE, token=hf_token)

            result = client.predict(
                handle_file(image_path),
                prompt,
                "Canny",
                negative_prompt,
                42,
                True,
                1.0,
                5.0,
                30,
                False,
                api_name="/generate",
            )

        if isinstance(result, (list, tuple)) and len(result) >= 1:
            st.image(result[0], caption="生成された内観CG", use_container_width=True)
            if len(result) >= 2:
                with st.expander("ControlNetが読み取った線画像"):
                    st.image(result[1], use_container_width=True)
        else:
            st.write(result)

except Exception as e:
    import traceback

    st.error("生成に失敗しました。")
    st.write("エラー種類：", type(e).__name__)
    st.write("エラー内容：")
    st.code(repr(e))
    st.write("詳細ログ：")
    st.code(traceback.format_exc())
    finally:
        try:
            os.remove(image_path)
        except OSError:
            pass

st.divider()
st.caption(
    "※無料ZeroGPUには日次の利用枠があります。図面の形状を完全にCADのように固定することはできません。"
)
