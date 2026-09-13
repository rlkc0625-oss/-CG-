import gc
import hashlib
import io
import json
import math
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
import torch
import segmentation_models_pytorch as smp
from PIL import Image, ImageOps
from safetensors.torch import load_file
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

st.set_page_config(page_title="図面→内観CGメーカー", page_icon="🏠", layout="wide")

MODEL_URL = "https://huggingface.co/Yytsi/floorplan-to-3d-walls/resolve/main/best.safetensors"
MODEL_SHA256 = "d7f6a0fd06e2931aecfc8c4849192c5e153701578026efc78d9a6246731a8d6c"
MODEL_SIZE = 97851168
MODEL_PATH = Path("best.safetensors")
SIZE = 512
MAX_MB = 15
CLASSES = ("floor", "wall", "door", "window")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

STYLE = {
    "ナチュラル": {"wall": "#F2EEE5", "floor": "#C9A77A", "wood": "#A97B50", "accent": "#8B6B4E"},
    "グレージュモダン": {"wall": "#D8D2C8", "floor": "#B9AFA2", "wood": "#8E7965", "accent": "#5F5B57"},
    "モダン和風": {"wall": "#E7E1D6", "floor": "#9B7656", "wood": "#6E4D38", "accent": "#403B36"},
    "ホテルライク": {"wall": "#D7D7D5", "floor": "#77736D", "wood": "#5E544A", "accent": "#2F3032"},
    "北欧": {"wall": "#F4F2EC", "floor": "#D1B18A", "wood": "#B28A62", "accent": "#66706A"},
}
LIGHT = {
    "昼・自然光": (1.0, 0.96, 0.88),
    "夕方・暖色": (1.0, 0.72, 0.42),
    "夜・間接照明": (0.72, 0.78, 1.0),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model():
    if MODEL_PATH.exists():
        try:
            if MODEL_PATH.stat().st_size == MODEL_SIZE and sha256(MODEL_PATH) == MODEL_SHA256:
                return
            MODEL_PATH.unlink()
        except OSError as e:
            raise RuntimeError("AIモデルを確認できません。アプリを再起動してください。") from e

    tmp = MODEL_PATH.with_suffix(".download")
    last = None
    for attempt in range(3):
        try:
            if tmp.exists():
                tmp.unlink()
            with requests.get(MODEL_URL, stream=True, timeout=(20, 180), headers={"User-Agent": "floorplan-cg-app"}) as r:
                if r.status_code == 429:
                    raise RuntimeError("Hugging Faceの一時的なアクセス制限です。少し時間を置いて再試行してください。")
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
            if tmp.stat().st_size != MODEL_SIZE:
                raise RuntimeError("AIモデルのダウンロードサイズが一致しません。")
            if sha256(tmp) != MODEL_SHA256:
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
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=4)
    state = load_file(str(MODEL_PATH), device="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def read_image(uploaded):
    raw = uploaded.getvalue()
    if not raw:
        raise ValueError("画像ファイルが空です。")
    if len(raw) > MAX_MB * 1024 * 1024:
        raise ValueError(f"画像は{MAX_MB}MB以下にしてください。")
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except Exception as e:
        raise ValueError("PNG/JPG/JPEG/WEBPの図面を使用してください。") from e
    if img.width < 100 or img.height < 100:
        raise ValueError("画像が小さすぎます。")
    return img


def preprocess(image):
    w, h = image.size
    scale = min(SIZE / w, SIZE / h)
    iw, ih = max(1, round(w * scale)), max(1, round(h * scale))
    resized = image.resize((iw, ih), Image.Resampling.LANCZOS)
    fill = tuple(int(x * 255) for x in MEAN)
    canvas = Image.new("RGB", (SIZE, SIZE), fill)
    left, top = (SIZE - iw) // 2, (SIZE - ih) // 2
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
            raise RuntimeError("メモリ不足です。画像を小さくして再試行してください。") from e
        raise RuntimeError("AI推論中にエラーが発生しました。") from e
    left, top, iw, ih = rect
    cleaned = np.zeros_like(mask)
    cleaned[top:top + ih, left:left + iw] = mask[top:top + ih, left:left + iw]
    return canvas, cleaned, rect


def extract_polygons(mask, class_id):
    binary = (mask == class_id).astype(np.uint8)
    if binary.sum() == 0:
        return []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    holes = {}
    for i, (_, _, _, parent) in enumerate(hierarchy):
        if parent != -1:
            holes.setdefault(parent, []).append(i)
    result = []
    for i, (_, _, _, parent) in enumerate(hierarchy):
        if parent != -1 or cv2.contourArea(contours[i]) < 30:
            continue
        outer = cv2.approxPolyDP(contours[i], 1.5, True).reshape(-1, 2).astype(float).tolist()
        if len(outer) < 3:
            continue
        result.append({"outer": outer, "holes": [cv2.approxPolyDP(contours[j], 1.5, True).reshape(-1, 2).astype(float).tolist() for j in holes.get(i, []) if cv2.contourArea(contours[j]) >= 30]})
    return result


def make_geometry(mask):
    return {name: extract_polygons(mask, i) for i, name in enumerate(CLASSES)}


def polygon_area(poly):
    p = np.asarray(poly, dtype=float)
    if len(p) < 3:
        return 0.0
    return abs(cv2.contourArea(p.astype(np.float32)))


def largest_floor(geometry):
    floors = geometry.get("floor", [])
    if not floors:
        return None
    return max(floors, key=lambda p: polygon_area(p["outer"]))


def build_preview_mesh(geometry, style_name):
    """Build a lightweight deterministic 3D preview from the segmentation mask.
    This is a visualization, not a construction-accurate BIM/CAD model.
    """
    scale = 0.025
    wall_h = 2.4
    floor_poly = largest_floor(geometry)
    if floor_poly is None:
        raise RuntimeError("床領域を認識できなかったため、3D化できませんでした。")

    pts = np.asarray(floor_poly["outer"], dtype=float)
    pts[:, 0] *= scale
    pts[:, 1] = (SIZE - pts[:, 1]) * scale
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
    if xmax - xmin < 2 or ymax - ymin < 2:
        raise RuntimeError("認識された床領域が小さすぎます。")

    palette = STYLE[style_name]
    return pts, (xmin, xmax, ymin, ymax), palette, wall_h


def cuboid(ax, x0, x1, y0, y1, z0, z1, color, alpha=1.0):
    verts = [
        [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],
        [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],
        [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],
        [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],
        [(x1,y1,z0),(x0,y1,z0),(x0,y1,z1),(x1,y1,z1)],
        [(x0,y1,z0),(x0,y0,z0),(x0,y0,z1),(x0,y1,z1)],
    ]
    ax.add_collection3d(Poly3DCollection(verts, facecolors=color, edgecolors="none", alpha=alpha))


def make_cg(geometry, style_name, lighting, view_name):
    pts, bounds, palette, wall_h = build_preview_mesh(geometry, style_name)
    xmin, xmax, ymin, ymax = bounds
    lx, ly, lz = LIGHT[lighting]
    fig = plt.figure(figsize=(10, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#EDEBE6")
    ax.set_facecolor("#EDEBE6")

    # Floor surface
    floor3 = [(x, y, 0) for x, y in pts]
    ax.add_collection3d(Poly3DCollection([floor3], facecolors=palette["floor"], edgecolors="none", alpha=1))

    # Boundary walls from floor bounding box. This intentionally creates a stable room shell
    # even when the segmentation returns fragmented wall contours.
    t = max(0.12, min((xmax-xmin), (ymax-ymin)) * 0.018)
    cuboid(ax, xmin, xmax, ymin, ymin+t, 0, wall_h, palette["wall"])
    cuboid(ax, xmin, xmax, ymax-t, ymax, 0, wall_h, palette["wall"])
    cuboid(ax, xmin, xmin+t, ymin, ymax, 0, wall_h, palette["wall"])
    cuboid(ax, xmax-t, xmax, ymin, ymax, 0, wall_h, palette["wall"])

    # Simple architectural furniture blocks, kept generic so no fake product/model names are introduced.
    room_w, room_d = xmax-xmin, ymax-ymin
    cx, cy = (xmin+xmax)/2, (ymin+ymax)/2
    sofa_w = min(room_w*0.42, 2.4)
    sofa_d = min(room_d*0.16, 0.9)
    cuboid(ax, cx-sofa_w/2, cx+sofa_w/2, ymin+room_d*0.20, ymin+room_d*0.20+sofa_d, 0.12, 0.48, palette["accent"])
    table_w = min(room_w*0.25, 1.5)
    table_d = min(room_d*0.16, 0.85)
    cuboid(ax, cx-table_w/2, cx+table_w/2, cy-table_d/2, cy+table_d/2, 0.55, 0.68, palette["wood"])
    # kitchen counter along the opposite side
    counter_d = min(room_d*0.12, 0.65)
    cuboid(ax, xmin+room_w*0.08, xmin+room_w*0.08+min(room_w*0.38,2.6), ymax-room_d*0.12, ymax-room_d*0.12+counter_d, 0.78, 0.92, palette["wood"])
    # dining table and chairs as compact blocks
    if view_name in ("ダイニング", "キッチン", "LDK"):
        dw, dd = min(room_w*0.20, 1.2), min(room_d*0.14, 0.75)
        cuboid(ax, xmax-room_w*0.30, xmax-room_w*0.30+dw, cy-dd/2, cy+dd/2, 0.68, 0.76, palette["wood"])

    # Lighting is represented by a subtle emissive ceiling volume.
    if lighting == "夜・間接照明":
        for frac in (0.28, 0.5, 0.72):
            x = xmin + room_w*frac
            cuboid(ax, x-0.08, x+0.08, ymax-0.18, ymax-0.08, wall_h-0.08, wall_h-0.02, "#FFF2C7", 0.9)
    elif lighting == "夕方・暖色":
        cuboid(ax, xmin+room_w*0.35, xmin+room_w*0.65, ymax-0.10, ymax-0.04, wall_h-0.05, wall_h, "#FFE2B0", 0.85)

    if view_name == "玄関":
        elev = 18
        azim = -65
    elif view_name == "キッチン":
        elev = 20
        azim = 25
    elif view_name == "ダイニング":
        elev = 22
        azim = -25
    else:
        elev = 21
        azim = -55
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(0, wall_h)
    ax.set_box_aspect((max(room_w,0.1), max(room_d,0.1), wall_h))
    ax.set_axis_off()
    # Stable framing and lighting-like tonal modulation.
    ax.set_title(f"{view_name} / {style_name} / {lighting}", pad=8, fontsize=12)
    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return Image.open(buf).convert("RGB")


def color_mask(mask):
    colors = [(238,238,234),(55,55,60),(225,125,55),(55,150,225)]
    out = np.zeros((SIZE,SIZE,3), dtype=np.uint8)
    for i, c in enumerate(colors):
        out[mask == i] = c
    return Image.fromarray(out)


def overlay(base, mask):
    out = np.asarray(base).copy().astype(np.float32)
    for i, c in ((1,(55,55,60)),(2,(225,125,55)),(3,(55,150,225))):
        m = mask == i
        out[m] = out[m]*0.4 + np.asarray(c)*0.6
    return Image.fromarray(np.clip(out,0,255).astype(np.uint8))


def structure_json(geometry, rect):
    return {"version": 2, "canvas_size": [SIZE,SIZE], "content_rect": list(rect), "classes": CLASSES, "polygons": geometry, "note": "3D preview is approximate and is not construction-accurate."}


def make_obj(geometry):
    scale=0.02; heights={"wall":2.4,"door":2.0,"window":1.2}
    lines=["# floorplan-to-3d preview","# Approximate scale only"]
    vc=0
    for name in ("wall","door","window"):
        for poly in geometry.get(name,[]):
            pts=poly["outer"]
            if len(pts)<3: continue
            base=vc+1
            for x,y in pts: lines.append(f"v {x*scale:.4f} {(SIZE-y)*scale:.4f} 0")
            for x,y in pts: lines.append(f"v {x*scale:.4f} {(SIZE-y)*scale:.4f} {heights[name]:.4f}")
            n=len(pts); lines.append(f"g {name}")
            for i in range(n):
                j=(i+1)%n; lines.append(f"f {base+i} {base+j} {base+n+j} {base+n+i}")
            vc += n*2
    return "\n".join(lines)+"\n"


st.title("🏠 図面 → 内観CGメーカー")
st.caption("間取り図を解析し、構造確認と3D内観プレビューまで自動生成します。")

uploaded = st.file_uploader("① 間取り図をアップロード", type=["png","jpg","jpeg","webp"])
style = st.selectbox("② 内装テイスト", list(STYLE.keys()))
lighting = st.selectbox("③ 照明", list(LIGHT.keys()))
view = st.selectbox("④ 視点", ["LDK","リビング","キッチン","ダイニング","玄関"])

if uploaded:
    try:
        original = read_image(uploaded)
        st.image(original, caption="元の間取り図", use_container_width=True)
    except Exception as e:
        st.error(str(e)); st.stop()

    if st.button("🚀 図面からCGを作成", type="primary", use_container_width=True):
        try:
            with st.spinner("AIモデルを準備しています…"):
                load_model()
            with st.spinner("図面を解析しています…"):
                processed, mask, rect = predict(original)
                geometry = make_geometry(mask)

            counts = {k: len(geometry[k]) for k in ("wall","door","window")}
            c1,c2,c3 = st.columns(3)
            with c1: st.image(processed, caption="AI入力", use_container_width=True)
            with c2: st.image(color_mask(mask), caption="AI判定", use_container_width=True)
            with c3: st.image(overlay(processed, mask), caption="構造確認", use_container_width=True)
            st.write(f"壁 {counts['wall']} / ドア {counts['door']} / 窓 {counts['window']}")

            if counts["wall"] + counts["door"] + counts["window"] == 0:
                st.error("構造を認識できませんでした。正面から撮影した図面で再試行してください。")
                st.stop()

            with st.spinner("3D内観CGを生成しています…"):
                cg = make_cg(geometry, style, lighting, view)

            st.success("3D内観CGの生成が完了しました。")
            st.image(cg, caption=f"3D内観プレビュー｜{view}｜{style}｜{lighting}", use_container_width=True)
            out=io.BytesIO(); cg.save(out,format="PNG")
            st.download_button("📥 内観CG PNG", out.getvalue(), "interior_cg_preview.png", "image/png", use_container_width=True)

            data=structure_json(geometry,rect)
            st.download_button("📥 構造JSON", json.dumps(data,ensure_ascii=False,indent=2).encode(), "floorplan_structure.json", "application/json", use_container_width=True)
            st.download_button("📥 3D OBJ", make_obj(geometry).encode(), "floorplan_structure.obj", "text/plain", use_container_width=True)

            st.warning("このCGはAI解析結果から自動生成する3Dプレビューです。壁・窓・ドア位置、寸法、家具配置を施工図レベルで保証するものではありません。")
        except requests.exceptions.RequestException as e:
            st.error("AIモデル取得中の通信エラーです。時間を置いて再試行してください。")
            with st.expander("詳細"): st.code(repr(e))
        except RuntimeError as e:
            st.error(str(e))
            with st.expander("詳細"): st.exception(e)
        except Exception as e:
            st.error("予期しないエラーが発生しました。")
            with st.expander("詳細"): st.exception(e)
        finally:
            gc.collect()

st.divider()
st.caption("構造モデル: Yytsi/floorplan-to-3d-walls / MIT。公開モデルはCubiCasa5K中心の学習のため、日本の実施設計図では誤認識する場合があります。")
