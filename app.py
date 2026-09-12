import io
import time
import base64
import requests
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps

st.set_page_config(page_title="図面→内観CGメーカー", page_icon="🏠", layout="centered")

st.title("🏠 図面→内観CGメーカー")
st.caption("無料AI Horde版 Ver.4 — 図面を『室内から見たCG』へ変換")

uploaded = st.file_uploader("① 図面をアップロード", type=["png", "jpg", "jpeg", "webp"])

view = st.selectbox("② 視点", [
    "LDK全体", "リビング側からキッチンを見る", "キッチン側からリビングを見る",
    "ダイニング側からLDKを見る", "玄関ホール", "自由指定"
])

style = st.selectbox("③ インテリア", [
    "ナチュラル", "グレージュモダン", "ホテルライク", "北欧",
    "和モダン", "シンプルモダン", "自由指定"
])

lighting = st.selectbox("④ 時間帯", ["昼", "夕方", "夜"])
custom = st.text_area("⑤ 追加指示（任意）",
    placeholder="例：床は明るいオーク、壁は白、間接照明を入れる。高級感のある住宅CGにする。")

if uploaded:
    src = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
    src = ImageOps.exif_transpose(src)

    max_side = 1536
    scale = min(1.0, max_side / max(src.size))
    if scale < 1:
        src = src.resize((int(src.width * scale), int(src.height * scale)),
                         Image.Resampling.LANCZOS)

    src = ImageEnhance.Contrast(src).enhance(1.18)
    src = ImageEnhance.Sharpness(src).enhance(1.15)

    buf = io.BytesIO()
    src.save(buf, format="WEBP", quality=88, method=6)
    source_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    st.image(src, caption="アップロードした図面", use_container_width=True)

    if st.button("✨ CGを生成", type="primary", use_container_width=True):
        style_text = {
            "ナチュラル": "明るいナチュラル住宅、木質感のある床、白〜淡色の壁",
            "グレージュモダン": "上品なグレージュ、木と石を組み合わせた現代的な住宅",
            "ホテルライク": "高級ホテルのような落ち着いた住宅、石・木・間接照明",
            "北欧": "明るい北欧住宅、淡い木、白い壁、柔らかな家具",
            "和モダン": "現代的な和モダン住宅、木、石、落ち着いた色",
            "シンプルモダン": "無駄を抑えた上質なシンプルモダン住宅",
            "自由指定": custom or "上質で自然な住宅インテリア"
        }[style]

        light_text = {
            "昼": "昼の自然光が大きな窓から入る明るい室内",
            "夕方": "夕方の柔らかな自然光と暖色の照明",
            "夜": "夜の落ち着いた室内、暖色の間接照明とダウンライト"
        }[lighting]

        view_text = {
            "LDK全体": "LDKの室内全体を見渡す広角のアイレベル視点",
            "リビング側からキッチンを見る": "リビングに立ってキッチン方向を見るアイレベル視点",
            "キッチン側からリビングを見る": "キッチン付近からリビング方向を見るアイレベル視点",
            "ダイニング側からLDKを見る": "ダイニング側からLDK全体を見るアイレベル視点",
            "玄関ホール": "玄関ホールに立って室内方向を見るアイレベル視点",
            "自由指定": custom or "自然で使いやすいアイレベルの室内視点"
        }[view]

        prompt = f"""
Create a photorealistic architectural interior visualization from the uploaded floor-plan image.

IMPORTANT: The uploaded image is a FLOOR PLAN REFERENCE, NOT the final image.
Do NOT return the floor plan, blueprint, top-down drawing, or photograph of paper.
Instead, reconstruct the likely INTERIOR SPACE and render it as a realistic eye-level architectural photograph.

VIEW: {view_text}
INTERIOR STYLE: {style_text}
LIGHTING: {light_text}

GEOMETRY PRESERVATION:
- Carefully inspect the floor plan first.
- Preserve the shown room arrangement and relative spatial relationships.
- Preserve wall locations, major openings, doors, windows, kitchen location, and circulation as much as possible.
- Do not invent additional rooms or move major walls.
- Do not arbitrarily move doors or windows.
- Do not change the basic LDK proportions.
- If something is unclear, make a conservative architectural inference rather than redesigning the plan.

OUTPUT:
- Photorealistic residential interior CG/photo.
- Eye-level camera around 1.5 m above floor.
- Natural perspective and believable room proportions.
- Show floor, walls, ceiling, lighting, kitchen/cabinetry and only compatible furniture.
- No floor-plan lines, handwritten marks, dimensions, labels, Japanese text, logos or watermarks.
- Do not show the source paper.

ADDITIONAL USER INSTRUCTIONS:
{custom}
"""

        payload = {
            "prompt": prompt,
            "params": {
                "width": 576, "height": 576, "steps": 15,
                "cfg_scale": 6.5, "denoising_strength": 0.65,
                "sampler_name": "k_euler", "n": 1
            },
            "source_image": source_b64,
            "source_processing": "img2img",
            "nsfw": False
        }

        # AI HordeのAPIキーはHTTPヘッダーで送ります
        horde_headers = {
            "Content-Type": "application/json",
            "apikey": "0000000000",
            "Client-Agent": "FloorplanCG-Free/4.1"
        }

        with st.spinner("AIが図面を解析して室内CGを生成しています…"):
            try:
                r = requests.post(
                    "https://aihorde.net/api/v2/generate/async",
                    json=payload, headers=horde_headers, timeout=60
                )
                if r.status_code >= 400:
                    st.error(f"AI Hordeエラー {r.status_code}: {r.text}")
                    st.stop()

                job_id = r.json().get("id")
                if not job_id:
                    st.error(f"生成IDを取得できませんでした: {r.text}")
                    st.stop()

                result = None
                for _ in range(90):
                    time.sleep(2)
                    check = requests.get(
                        f"https://aihorde.net/api/v2/generate/check/{job_id}",
                        headers=horde_headers,
                        timeout=30
                    )
                    if check.status_code >= 400:
                        st.error(f"生成確認エラー {check.status_code}: {check.text}")
                        st.stop()

                    status = check.json()
                    if status.get("done"):
                        result_r = requests.get(
                            f"https://aihorde.net/api/v2/generate/status/{job_id}",
                            headers=horde_headers,
                            timeout=60
                        )
                        if result_r.status_code >= 400:
                            st.error(f"結果取得エラー {result_r.status_code}: {result_r.text}")
                            st.stop()
                        result = result_r.json()
                        break

                if not result:
                    st.error("生成に時間がかかっています。もう一度お試しください。")
                    st.stop()

                generations = result.get("generations", [])
                if not generations:
                    st.error(f"画像が返されませんでした: {result}")
                    st.stop()

                img_url = generations[0].get("img")
                if not img_url:
                    st.error(f"画像URLを取得できませんでした: {generations[0]}")
                    st.stop()

                if img_url.startswith("data:image"):
                    _, encoded = img_url.split(",", 1)
                    image_bytes = base64.b64decode(encoded)
                elif img_url.startswith("http"):
                    image_bytes = requests.get(img_url, timeout=60).content
                else:
                    image_bytes = base64.b64decode(img_url)

                output = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                st.success("生成完了")
                st.image(output, caption="生成された内観CG", use_container_width=True)

                out = io.BytesIO()
                output.save(out, format="PNG")
                st.download_button(
                    "📥 CGを保存", data=out.getvalue(),
                    file_name="interior_cg.png", mime="image/png"
                )

            except requests.RequestException as e:
                st.error(f"AI Hordeとの通信に失敗しました：{e}")
            except Exception as e:
                st.error(f"生成に失敗しました：{e}")

st.divider()
st.caption("※無料検証版です。AI画像生成のため、寸法・壁・窓・ドアを完全一致させるCAD/BIMではありません。")
