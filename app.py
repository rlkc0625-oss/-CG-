import io
import time
import base64
import requests
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps

API = "https://aihorde.net/api/v2"
HEADERS = {
    "Content-Type": "application/json",
    "apikey": "0000000000",
    "Client-Agent": "FloorplanCG-Free/10.0"
}

st.set_page_config(page_title="図面→内観CGメーカー", page_icon="🏠", layout="centered")
st.title("🏠 図面→内観CGメーカー")
st.caption("無料AI Horde版 Ver.10 — まず『本物の住宅内観CG』を出す版")

uploaded = st.file_uploader(
    "① 図面をアップロード（参考用）",
    type=["png", "jpg", "jpeg", "webp"]
)

view = st.selectbox("② 視点", [
    "LDK全体",
    "リビング側からキッチンを見る",
    "キッチン側からリビングを見る",
    "ダイニング側からLDKを見る",
    "玄関ホール",
    "自由指定"
])

style = st.selectbox("③ インテリア", [
    "ナチュラル",
    "グレージュモダン",
    "ホテルライク",
    "北欧",
    "和モダン",
    "シンプルモダン",
    "自由指定"
])

lighting = st.selectbox("④ 時間帯", ["昼", "夕方", "夜"])

custom = st.text_area(
    "⑤ 追加指示（任意）",
    placeholder="例：明るいオーク床、白い壁、グレージュのキッチン、間接照明、高級感"
)

if uploaded:
    src = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
    src = ImageOps.exif_transpose(src)

    max_side = 1536
    scale = min(1.0, max_side / max(src.size))
    if scale < 1:
        src = src.resize(
            (int(src.width * scale), int(src.height * scale)),
            Image.Resampling.LANCZOS
        )

    src = ImageEnhance.Contrast(src).enhance(1.15)
    src = ImageEnhance.Sharpness(src).enhance(1.10)

    st.image(src, caption="アップロードした図面", use_container_width=True)

    if st.button("✨ CGを生成", type="primary", use_container_width=True):

        style_text = {
            "ナチュラル":
                "premium Japanese natural modern home, light oak floor, warm white walls, refined wood cabinetry",
            "グレージュモダン":
                "luxury greige modern Japanese home, warm gray walls, oak wood, stone accents",
            "ホテルライク":
                "high-end Japanese residence with hotel-inspired luxury, natural stone, wood, elegant indirect lighting",
            "北欧":
                "bright refined Scandinavian-inspired Japanese home, pale oak, soft white walls, elegant simple furniture",
            "和モダン":
                "premium contemporary Japanese modern home, natural wood, stone, calm neutral palette",
            "シンプルモダン":
                "premium minimalist modern Japanese home, clean architectural lines, sophisticated neutral materials",
            "自由指定":
                custom or "premium contemporary Japanese residential interior"
        }[style]

        view_text = {
            "LDK全体":
                "wide eye-level view of a finished open-plan LDK",
            "リビング側からキッチンを見る":
                "standing in the living room looking toward the finished kitchen",
            "キッチン側からリビングを見る":
                "standing near the kitchen looking toward the finished living room",
            "ダイニング側からLDKを見る":
                "standing at the dining area looking across the finished LDK",
            "玄関ホール":
                "standing in a finished Japanese entrance hall looking toward the interior",
            "自由指定":
                custom or "natural eye-level architectural interior view"
        }[view]

        light_text = {
            "昼": "soft natural daylight from windows, bright and realistic",
            "夕方": "warm late-afternoon sunlight with subtle interior lighting",
            "夜": "warm sophisticated evening lighting with downlights and indirect lighting"
        }[lighting]

        # 重要：今回は図面画像を画像生成APIへ渡さない。
        # 図面の線がそのまま出る原因を完全に切り離し、
        # まず「普通の完成住宅CG」を安定して出す。
        prompt = f"""
Create a finished FULL-COLOR PHOTOREALISTIC architectural visualization
of a premium Japanese residential interior.

This is a completed real home interior, photographed with a professional
architectural camera.

CAMERA:
{view_text}
Eye-level camera approximately 1.5 meters above the finished floor.
Natural wide-angle perspective.
Realistic room depth and believable proportions.

DESIGN:
{style_text}

LIGHTING:
{light_text}

SCENE:
A beautiful finished residential interior with a realistic floor,
finished walls, finished ceiling, windows, doors, kitchen cabinetry,
built-in storage, sofa, dining table, chairs, rugs and tasteful decor.
Use realistic architectural materials and construction details.

QUALITY:
photorealistic, high-end residential photography,
physically believable materials, realistic wood grain,
natural stone texture, realistic glass, metal and fabric,
soft global illumination, realistic contact shadows,
subtle reflections, natural exposure, premium Japanese custom-home CG.

IMPORTANT:
The uploaded floor plan is only a reference for the user's house project.
Do not reproduce, trace, redraw or display the floor plan.
The result must be a perspective interior photograph.
The result must be a normal, clean residential interior image.

USER DESIGN REQUEST:
{custom}
"""

        payload = {
            "prompt": prompt,
            "params": {
                "width": 576,
                "height": 576,
                "steps": 20,
                "cfg_scale": 7.0
            },
            "nsfw": False
        }

        with st.spinner("AIがカラーの住宅内観CGを生成しています…"):
            try:
                r = requests.post(
                    f"{API}/generate/async",
                    json=payload,
                    headers=HEADERS,
                    timeout=60
                )

                if r.status_code >= 400:
                    st.error(f"AI Hordeエラー {r.status_code}: {r.text}")
                    st.stop()

                job_id = r.json().get("id")
                if not job_id:
                    st.error(f"生成IDを取得できませんでした: {r.text}")
                    st.stop()

                img_url = None

                for _ in range(150):
                    time.sleep(2)

                    check = requests.get(
                        f"{API}/generate/check/{job_id}",
                        headers=HEADERS,
                        timeout=30
                    )

                    if check.status_code >= 400:
                        st.error(
                            f"生成確認エラー {check.status_code}: {check.text}"
                        )
                        st.stop()

                    status = check.json()

                    if status.get("done"):
                        result = requests.get(
                            f"{API}/generate/status/{job_id}",
                            headers=HEADERS,
                            timeout=60
                        )

                        if result.status_code >= 400:
                            st.error(
                                f"結果取得エラー {result.status_code}: {result.text}"
                            )
                            st.stop()

                        generations = result.json().get("generations") or []

                        if generations:
                            img_url = generations[0].get("img")
                        break

                if not img_url:
                    st.error("画像が返されませんでした。もう一度お試しください。")
                    st.stop()

                if img_url.startswith("data:image"):
                    image_bytes = base64.b64decode(img_url.split(",", 1)[1])
                elif img_url.startswith("http"):
                    image_bytes = requests.get(
                        img_url, timeout=60
                    ).content
                else:
                    image_bytes = base64.b64decode(img_url)

                output = Image.open(
                    io.BytesIO(image_bytes)
                ).convert("RGB")

                st.success("生成完了")
                st.image(
                    output,
                    caption="生成された内観CG",
                    use_container_width=True
                )

                out = io.BytesIO()
                output.save(out, format="PNG")

                st.download_button(
                    "📥 CGを保存",
                    data=out.getvalue(),
                    file_name="interior_cg.png",
                    mime="image/png"
                )

            except requests.RequestException as e:
                st.error(f"AI Hordeとの通信に失敗しました：{e}")
            except Exception as e:
                st.error(f"生成に失敗しました：{e}")

st.divider()
st.caption(
    "※無料検証版。Ver.10はまず『ちゃんとしたカラー住宅内観CGを生成する』ことを優先しています。"
)
