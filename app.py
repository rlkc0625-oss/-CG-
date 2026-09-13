import gc
import hashlib
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st
import torch
import segmentation_models_pytorch as smp
from PIL import Image, ImageOps, ImageDraw
from safetensors.torch import load_file

st.set_page_config(page_title="図面→3D構造", page_icon="🏠", layout="wide")

# Official Yytsi/floorplan-to-3d-walls
MODEL_URL = "https://huggingface.co/Yytsi/floorplan-to-3d-walls/resolve/main/best.safetensors"
MODEL_SHA256 = "d7f6a0fd06e2931aecfc8c4849192c5e153701578026efc78d9a6246731a8d6c"
MODEL_PATH = Path("best.safetensors")

SIZE = 512
MAX_MB = 15
CLASSES = ("floor", "wall", "door", "window")
COLORS = {
    "floor": (238, 238, 234),
    "wall": (55, 55, 60),
    "door": (225, 125, 55),
    "window": (55, 150, 225),
}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model():
    if MODEL_PATH.exists():
        try:
            if MODEL_PATH.stat().st_size == 97851168 and file_sha256(MODEL_PATH) == MODEL_SHA256:
                return
        except OSError:
            pass
        try:
            MODEL_PATH.unlink()
        except OSError as e:
            raise RuntimeError("壊れたモデルファイルを削除できません。アプリを再起動してください。") from e

    tmp = MODEL_PATH.with_suffix(".download")
    last = None

    for attempt in range(3):
        try:
            if tmp.exists():
                tmp.unlink()
            with requests.get(
                MODEL_URL,
                stream=True,
                timeout=(20, 180),
                headers={"User-Agent": "floorplan-cg-app"},
            ) as r:
                if r.status_code == 429:
                    raise RuntimeError("Hugging Faceの一時的なアクセス制限です。")
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)

            if tmp.stat().st_size != 97851168:
                raise RuntimeError("AIモデルのダウンロードが途中で終了しました。")

            if file_sha256(tmp) != MODEL_SHA256:
                raise RuntimeError("AIモデルの整合性確認に失敗しました。")

            tmp.replace(MODEL_PATH)
            return
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))

    raise RuntimeError("AIモデルを取得できませんでした。") from last


@st.cache_resource(show_spinner=False)
def load_model():
    ensure_model()
    try:
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=4,
        )
        state = load_file(str(MODEL_PATH), device="cpu")
        model.load_state_dict(state, strict=True)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(
            "公式モデルの読み込みに失敗しました。依存ライブラリとモデル重みの組み合わせを確認してください。"
        ) from e


def read_image(uploaded):
    raw = uploaded.getvalue()
    if not raw:
        raise ValueError("画像ファイルが空です。")
    if len(raw) > MAX_MB * 1024 * 1024:
        raise ValueError(f"画像が大きすぎます。{MAX_MB}MB以下にしてください。")
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:
        raise ValueError("画像を読み込めません。PNG/JPG/JPEG/WEBPを使用してください。") from e
    if img.width < 100 or img.height < 100:
        raise ValueError("画像が小さすぎます。")
    return img


def preprocess(image):
    w, h = image.size
    scale = min(SIZE / w, SIZE / h)
    iw = max(1, round(w * scale))
    ih = max(1, round(h * scale))
    resized = image.resize((iw, ih), Image.Resampling.LANCZOS)

    # Official training preprocessing: ImageNet-mean fill + centered letterbox.
    fill = tuple(int(x * 255) for x in MEAN)
    canvas = Image.new("RGB", (SIZE, SIZE), fill)
    left = (SIZE - iw) // 2
    top = (SIZE - ih) // 2
    canvas.paste(resized, (left, top))

    arr = np.asarray(canvas).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    return canvas, tensor, (left, top, iw, ih)


def predict(image):
    canvas, tensor, rect = preprocess(image)
    model = load_model()

    try:
        with torch.inference_mode():
            logits = model(tensor.unsqueeze(0))
            mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    except RuntimeError as e:
        gc.collect()
        if "memory" in str(e).lower():
            raise RuntimeError("メモリ不足で解析できませんでした。画像を小さくして再試行してください。") from e
        raise RuntimeError("AI推論中にエラーが発生しました。") from e

    left, top, iw, ih = rect
    cleaned = np.zeros_like(mask)
    cleaned[top:top + ih, left:left + iw] = mask[top:top + ih, left:left + iw]
    return canvas, cleaned, rect


def color_mask(mask):
    out = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for i, name in enumerate(CLASSES):
        out[mask == i] = COLORS[name]
    return Image.fromarray(out)


def overlay(base, mask):
    out = np.asarray(base).copy().astype(np.float32)
    for i, name in enumerate(("wall", "door", "window"), 1):
        m = mask == i
        out[m] = out[m] * 0.4 + np.asarray(COLORS[name]) * 0.6
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def extract_polygons(mask, class_id):
    binary = (mask == class_id).astype(np.uint8)
    if binary.sum() == 0:
        return []

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(
        closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    holes_by_parent = {}
    for i, (_, _, _, parent) in enumerate(hierarchy):
        if parent != -1:
            holes_by_parent.setdefault(parent, []).append(i)

    result = []
    for i, (_, _, _, parent) in enumerate(hierarchy):
        if parent != -1 or cv2.contourArea(contours[i]) < 30:
            continue
        outer = cv2.approxPolyDP(contours[i], 1.5, True).reshape(-1, 2).astype(float).tolist()
        if len(outer) < 3:
            continue
        holes = []
        for j in holes_by_parent.get(i, []):
            if cv2.contourArea(contours[j]) < 30:
                continue
            ring = cv2.approxPolyDP(contours[j], 1.5, True).reshape(-1, 2).astype(float).tolist()
            if len(ring) >= 3:
                holes.append(ring)
        result.append({"outer": outer, "holes": holes})
    return result


def make_geometry(mask):
    return {
        name: extract_polygons(mask, i)
        for i, name in enumerate(CLASSES)
        if name != "floor"
    }


def make_obj(geometry):
    # No dimension line is available to the model, so this is an approximate
    # preview scale. Do not use this OBJ as a construction-accurate model.
    scale = 0.02
    heights = {"wall": 2.4, "door": 2.0, "window": 1.2}
    lines = ["# floorplan-to-3d preview", "# Approximate scale only"]
    vc = 0

    for name in ("wall", "door", "window"):
        for poly in geometry.get(name, []):
            pts = poly["outer"]
            if len(pts) < 3:
                continue
            base = vc + 1
            for x, y in pts:
                lines.append(f"v {x*scale:.4f} {(SIZE-y)*scale:.4f} 0")
            for x, y in pts:
                lines.append(f"v {x*scale:.4f} {(SIZE-y)*scale:.4f} {heights[name]:.4f}")
            n = len(pts)
            lines.append(f"g {name}")
            for i in range(n):
                j = (i + 1) % n
                lines.append(f"f {base+i} {base+j} {base+n+j} {base+n+i}")
            vc += n * 2

    return "\n".join(lines) + "\n"


st.title("🏠 図面 → 3D構造")
st.caption("壁・ドア・窓・床をAI解析し、3D用構造データを作成")

uploaded = st.file_uploader(
    "① 間取り図をアップロード",
    type=["png", "jpg", "jpeg", "webp"],
)

if uploaded:
    try:
        original = read_image(uploaded)
        st.image(original, caption="元の間取り図", use_container_width=True)
    except Exception as e:
        st.error(str(e))
        st.stop()

    if st.button("🚀 図面を解析", type="primary", use_container_width=True):
        try:
            with st.spinner("AIモデルを準備しています…"):
                load_model()

            with st.spinner("図面を解析しています…"):
                processed, mask, rect = predict(original)
                geometry = make_geometry(mask)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(processed, caption="AI入力", use_container_width=True)
            with c2:
                st.image(color_mask(mask), caption="AI判定", use_container_width=True)
            with c3:
                st.image(overlay(processed, mask), caption="構造確認", use_container_width=True)

            counts = {k: len(geometry[k]) for k in ("wall", "door", "window")}
            st.write(
                f"壁 {counts['wall']} / ドア {counts['door']} / 窓 {counts['window']}"
            )

            if sum(counts.values()) == 0:
                st.warning(
                    "構造物を認識できませんでした。正面から撮影した、文字や影の少ない図面で再試行してください。"
                )
            else:
                structure = {
                    "version": 1,
                    "canvas_size": [SIZE, SIZE],
                    "content_rect": list(rect),
                    "polygons": geometry,
                }

                st.download_button(
                    "📥 構造JSON",
                    json.dumps(structure, ensure_ascii=False, indent=2).encode(),
                    "floorplan_structure.json",
                    "application/json",
                    use_container_width=True,
                )

                st.download_button(
                    "📥 3D OBJ",
                    make_obj(geometry).encode(),
                    "floorplan_structure.obj",
                    "text/plain",
                    use_container_width=True,
                )

                st.success("図面の構造解析が完了しました。")

        except requests.exceptions.RequestException as e:
            st.error("AIモデルの取得中に通信エラーが発生しました。時間を置いて再試行してください。")
            with st.expander("詳細"):
                st.code(repr(e))
        except Exception as e:
            st.error("解析中にエラーが発生しました。")
            with st.expander("詳細"):
                st.exception(e)
        finally:
            gc.collect()

st.divider()
st.caption(
    "構造モデル: Yytsi/floorplan-to-3d-walls / MIT. "
    "公開モデルはCubiCasa5Kで学習されており、日本の実施設計図では誤認識する場合があります。"
)
