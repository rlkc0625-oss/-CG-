import io
import time
import base64
import requests
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps

API = "https://aihorde.net/api/v2"
KEY = "0000000000"
HEADERS = {
    "Content-Type": "application/json",
    "apikey": KEY,
    "Client-Agent": "FloorplanCG-Free/8.0"
}

st.set_page_config(page_title="図面→内観CGメーカー", page_icon="🏠", layout="centered")
st.title("🏠 図面→内観CGメーカー")
st.caption("無料AI Horde版 Ver.9 — 安全な無人住宅内観CG")

uploaded = st.file_uploader("① 図面をアップロード", type=["png", "jpg", "jpeg", "webp"])

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
    placeholder="例：床は明るいオーク、壁は白、キッチンはグレージュ、高級感のある住宅CG"
)

def wait_interrogation(job_id):
    for _ in range(120):
        time.sleep(2)
        r = requests.get(f"{API}/interrogate/status/{job_id}",
                         headers=HEADERS, timeout=30)
        if r.status_code >= 400:
            raise RuntimeError(f"図面解析確認エラー {r.status_code}: {r.text}")
        data = r.json()
        state = data.get("state", "")
        if state == "done":
            forms = data.get("forms") or []
            for form in forms:
                result = form.get("result") or {}
                if result.get("caption"):
                    return result["caption"]
                if result.get("interrogation"):
                    return str(result["interrogation"])
            return "住宅の平面図。LDK、居室、水回り、収納などを含む住宅レイアウト。"
        if state in ("faulted", "cancelled"):
            raise RuntimeError(f"図面解析に失敗しました: {data}")
    raise RuntimeError("図面解析がタイムアウトしました。")

def wait_image(job_id):
    for _ in range(150):
        time.sleep(2)
        r = requests.get(f"{API}/generate/check/{job_id}",
                         headers=HEADERS, timeout=30)
        if r.status_code >= 400:
            raise RuntimeError(f"画像生成確認エラー {r.status_code}: {r.text}")
        check = r.json()
        if check.get("done"):
            r2 = requests.get(f"{API}/generate/status/{job_id}",
                              headers=HEADERS, timeout=60)
            if r2.status_code >= 400:
                raise RuntimeError(f"画像結果取得エラー {r2.status_code}: {r2.text}")
            data = r2.json()
            gens = data.get("generations") or []
            if not gens:
                raise RuntimeError(f"画像が返されませんでした: {data}")
            return gens[0].get("img")
    raise RuntimeError("画像生成がタイムアウトしました。")

def get_image_bytes(img):
    if not img:
        raise RuntimeError("画像URLを取得できませんでした。")
    if img.startswith("data:image"):
        return base64.b64decode(img.split(",", 1)[1])
    if img.startswith("http"):
        r = requests.get(img, timeout=60)
        r.raise_for_status()
        return r.content
    return base64.b64decode(img)

if uploaded:
    src = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
    src = ImageOps.exif_transpose(src)

    max_side = 1536
    scale = min(1.0, max_side / max(src.size))
    if scale < 1:
        src = src.resize((int(src.width * scale), int(src.height * scale)),
                         Image.Resampling.LANCZOS)

    src = ImageEnhance.Contrast(src).enhance(1.15)
    src = ImageEnhance.Sharpness(src).enhance(1.10)

    buf = io.BytesIO()
    src.save(buf, format="WEBP", quality=88, method=6)
    source_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    st.image(src, caption="アップロードした図面", use_container_width=True)

    if st.button("✨ CGを生成", type="primary", use_container_width=True):
        style_text = {
            "ナチュラル": "warm Japanese natural modern home, light oak flooring, warm white walls, refined wood cabinetry",
            "グレージュモダン": "luxury greige modern Japanese home, warm gray walls, oak wood, stone accents",
            "ホテルライク": "high-end luxury hotel-like Japanese residence, stone, wood, elegant indirect lighting",
            "北欧": "bright Scandinavian-inspired Japanese home, pale oak, soft white walls, simple elegant furniture",
            "和モダン": "premium contemporary Japanese modern home, natural wood, stone, calm neutral palette",
            "シンプルモダン": "premium minimalist modern Japanese home, clean lines, sophisticated neutral materials",
            "自由指定": custom or "premium contemporary Japanese residential interior"
        }[style]

        view_text = {
            "LDK全体": "wide eye-level view of the LDK interior",
            "リビング側からキッチンを見る": "standing in the living room looking toward the kitchen",
            "キッチン側からリビングを見る": "standing near the kitchen looking toward the living room",
            "ダイニング側からLDKを見る": "standing at the dining area looking across the LDK",
            "玄関ホール": "standing in the entrance hall looking into the house",
            "自由指定": custom or "natural eye-level interior architectural view"
        }[view]

        light_text = {
            "昼": "soft natural daylight",
            "夕方": "warm late-afternoon sunlight mixed with interior lighting",
            "夜": "warm sophisticated night lighting with downlights and indirect lighting"
        }[lighting]

        with st.spinner("① 図面をAIに解析しています…"):
            try:
                ir_payload = {
                    "source_image": source_b64,
                    "forms": [{"name": "caption"}]
                }
                ir = requests.post(f"{API}/interrogate/async",
                                   json=ir_payload, headers=HEADERS, timeout=60)
                if ir.status_code >= 400:
                    st.error(f"図面解析エラー {ir.status_code}: {ir.text}")
                    st.stop()

                ir_id = ir.json().get("id")
                if not ir_id:
                    st.error(f"図面解析IDを取得できませんでした: {ir.text}")
                    st.stop()

                caption = wait_interrogation(ir_id)
                st.info(f"AIの図面認識: {caption}")

            except Exception as e:
                st.warning(f"図面解析は利用できなかったため、選択した視点・内装条件でCGを生成します。\\n{e}")
                caption = "住宅の平面図を参考にした住宅内部"

        # ここでは元画像をimg2imgに渡さない。
        # これが「図面線画のまま返る」問題を避けるポイント。
        prompt = f"""
Create a finished, FULL-COLOR, PHOTOREALISTIC high-end Japanese residential
INTERIOR ARCHITECTURAL VISUALIZATION of an EMPTY HOME INTERIOR.

SAFETY / CONTENT:
- The image contains NO people, NO children, NO adults, NO human figures,
  NO silhouettes, NO mannequins, NO portraits, and NO human-like characters.
- Show only architecture, furniture, fixtures, materials and lighting.
- This is a completely ordinary family home interior with no sexual or
  suggestive content of any kind.


This is an interior photograph/CG of a completed house.
The final image MUST NOT look like a floor plan, drawing, sketch, blueprint,
diagram, paper document, or line art.

CAMERA:
{view_text}
Eye-level camera around 1.5 meters high, realistic wide-angle architectural
photography, natural perspective, realistic room proportions.

INTERIOR:
{style_text}

LIGHT:
{light_text}

FLOOR-PLAN REFERENCE DESCRIPTION:
{caption}

Use the reference description only to infer the type of space and plausible
relationship between rooms. The final image is a perspective interior view,
NOT a reproduction of the plan.

REALISM:
realistic wood grain, stone, tile, painted walls, glass, metal, fabric,
natural reflections, contact shadows, physically plausible lighting,
realistic furniture scale, professional architectural photography,
premium Japanese custom home.

The room must be clearly three-dimensional:
floor, walls, ceiling, windows, doors, kitchen/cabinetry and furniture.
No top-down view.

ABSOLUTELY FORBIDDEN:
floor plan, blueprint, sketch, pencil, ink, line drawing, black outlines,
wireframe, diagram, paper, document, dimensions, Japanese labels, handwritten
notes, plan symbols, grayscale, monochrome, cartoon, illustration, tracing,
architectural drawing, people, person, child, adult, human figure, silhouette,
mannequin, portrait, body, face.

Additional user instructions:
{custom}
"""

        # 図面の説明はあくまで建築情報として使用し、人物等は生成しない。
        payload = {
            "prompt": prompt,
            "params": {
                "width": 576,
                "height": 576,
                "steps": 20,
                "cfg_scale": 7.0,
                "negative_prompt": "floor plan, blueprint, sketch, line art, pencil drawing, ink drawing, architectural drawing, diagram, grayscale, monochrome, wireframe, paper, document, handwritten notes, dimensions, labels, black outlines, traced lines, cartoon, illustration, top-down view, people, person, child, adult, human figure, silhouette, mannequin, portrait, face, body, nudity, sexual content"
            },
            "nsfw": False
        }

        with st.spinner("② 本物の室内CGを生成しています…"):
            try:
                gen = requests.post(f"{API}/generate/async",
                                   json=payload, headers=HEADERS, timeout=60)
                if gen.status_code >= 400:
                    st.error(f"AI Horde画像生成エラー {gen.status_code}: {gen.text}")
                    st.stop()

                job_id = gen.json().get("id")
                if not job_id:
                    st.error(f"生成IDを取得できませんでした: {gen.text}")
                    st.stop()

                img = wait_image(job_id)
                output = Image.open(io.BytesIO(get_image_bytes(img))).convert("RGB")

                st.success("生成完了")
                st.image(output, caption="生成された内観CG", use_container_width=True)

                out = io.BytesIO()
                output.save(out, format="PNG")
                st.download_button(
                    "📥 CGを保存",
                    data=out.getvalue(),
                    file_name="interior_cg.png",
                    mime="image/png"
                )

            except Exception as e:
                st.error(f"生成に失敗しました：{e}")

st.divider()
st.caption("※無料検証版。今回は『まず本物のカラー内観CGを出す』ことを優先しています。寸法・壁・窓・ドアの完全一致は次段階の課題です。")
