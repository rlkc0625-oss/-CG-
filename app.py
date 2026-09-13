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
from PIL import Image, ImageOps, ImageEnhance
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


def auto_crop(img):
    """Remove large photographic/blank margins while preserving the drawing."""
    arr = np.asarray(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # Keep pixels sufficiently darker than paper/background.
    mask = (gray < 245).astype(np.uint8) * 255
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    ys, xs = np.where(mask > 0)
    if len(xs) < 500:
        return img
    x0, x1 = max(0, xs.min() - 12), min(img.width, xs.max() + 13)
    y0, y1 = max(0, ys.min() - 12), min(img.height, ys.max() + 13)
    crop = img.crop((x0, y0, x1, y1))
    # Avoid pathological tiny crops.
    if crop.width < img.width * 0.15 or crop.height < img.height * 0.15:
        return img
    return crop


def preprocess(image):
    w, h = image.size
    scale = min(SIZE / w, SIZE / h)
    iw, ih = max(1, round(w * scale)), max(1, round(h * scale))
    resized = image.resize((iw, ih), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (SIZE, SIZE), (124, 124, 124))
    left, top = (SIZE - iw) // 2, (SIZE - ih) // 2
    canvas.paste(resized, (left, top))
    arr = np.asarray(canvas).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return canvas, tensor, (left, top, iw, ih)


def candidate_images(image):
    cropped = auto_crop(image)
    candidates = []
    seen = set()
    for source_name, src in (("original", image), ("cropped", cropped)):
        gray = ImageOps.grayscale(src).convert("RGB")
        enhanced = ImageEnhance.Contrast(gray).enhance(1.35)
        for angle in (0, 90, 180, 270):
            base = src.rotate(angle, expand=True) if angle else src
            key = (base.size, source_name, angle)
            if key not in seen:
                candidates.append((f"{source_name}-{angle}", base))
                seen.add(key)
            if angle == 0:
                candidates.append((f"{source_name}-contrast", enhanced))
    return candidates


def score_mask(mask):
    total = float(mask.size)
    ratios = [float(np.mean(mask == i)) for i in range(4)]
    wall, door, window = ratios[1], ratios[2], ratios[3]
    floor = ratios[0]
    # Prefer a meaningful floor plus architectural classes, reject near-uniform predictions.
    score = 0.0
    score += 4.0 * min(floor / 0.20, 1.0)
    score += 4.0 * min(wall / 0.025, 1.0)
    score += 2.0 * min(door / 0.004, 1.0)
    score += 2.0 * min(window / 0.004, 1.0)
    if floor > 0.92 or wall > 0.25:
        score -= 6.0
    # Spatial spread matters: useful structures should not live in a tiny corner.
    ys, xs = np.where(np.isin(mask, [1, 2, 3]))
    if len(xs) > 100:
        spread = ((xs.max()-xs.min()) / SIZE) * ((ys.max()-ys.min()) / SIZE)
        score += 3.0 * min(spread / 0.25, 1.0)
    return score


def predict(image):
    model = load_model()
    best = None
    errors = []
    candidates = candidate_images(image)
    progress = st.progress(0, text="図面の向き・余白を確認しています…")
    try:
        for idx, (name, candidate) in enumerate(candidates):
            try:
                canvas, tensor, rect = preprocess(candidate)
                with torch.inference_mode():
                    logits = model(tensor.unsqueeze(0))
                    mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                score = score_mask(mask)
                if best is None or score > best[0]:
                    best = (score, name, canvas, mask, rect)
            except RuntimeError as e:
                if "memory" in str(e).lower():
                    raise RuntimeError("AI推論でメモリ不足になりました。図面画像を少し小さくして再試行してください。") from e
                errors.append(f"{name}: {e}")
            progress.progress((idx + 1) / len(candidates))
    finally:
        progress.empty()
    gc.collect()
    if best is None:
        raise RuntimeError("図面のAI解析に失敗しました。別の画像形式で再試行してください。")
    _, chosen, canvas, mask, rect = best
    # Keep only the content rectangle; this prevents letterbox artifacts from becoming geometry.
    left, top, iw, ih = rect
    cleaned = np.zeros_like(mask)
    cleaned[top:top + ih, left:left + iw] = mask[top:top + ih, left:left + iw]
    return canvas, cleaned, rect, chosen


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


def _touches_canvas(poly, margin=3):
    p = np.asarray(poly, dtype=float)
    if len(p) == 0:
        return True
    return bool((p[:,0].min() <= margin) or (p[:,1].min() <= margin) or (p[:,0].max() >= SIZE-1-margin) or (p[:,1].max() >= SIZE-1-margin))


def largest_floor(geometry):
    floors = geometry.get("floor", [])
    candidates = [p for p in floors if not _touches_canvas(p.get("outer", []))]
    if not candidates:
        candidates = floors
    return max(candidates, key=lambda p: polygon_area(p["outer"]), default=None)


def _all_points(geometry):
    pts = []
    for name in ("wall", "door", "window"):
        for poly in geometry.get(name, []):
            pts.extend(poly.get("outer", []))
    return np.asarray(pts, dtype=float) if pts else None


def bounds_from_geometry(geometry, scale):
    pts = _all_points(geometry)
    if pts is None or len(pts) < 4:
        return None
    x0, x1 = float(pts[:,0].min()), float(pts[:,0].max())
    y0, y1 = float(pts[:,1].min()), float(pts[:,1].max())
    if x1-x0 < 30 or y1-y0 < 30:
        return None
    return (x0*scale, x1*scale, (SIZE-y1)*scale, (SIZE-y0)*scale)


def bounds_from_mask(mask, scale):
    m = np.isin(mask, [0,1,2,3]).astype(np.uint8)
    # Prefer non-background predicted pixels, but exclude only tiny isolated components.
    ys, xs = np.where(m > 0)
    if len(xs) < 100:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    if x1-x0 < 30 or y1-y0 < 30:
        return None
    return (x0*scale, x1*scale, (SIZE-y1)*scale, (SIZE-y0)*scale)


def choose_bounds(geometry, mask, scale):
    fp = largest_floor(geometry)
    if fp is not None:
        p = np.asarray(fp["outer"], dtype=float)
        if len(p) >= 3 and polygon_area(p) >= 800:
            q = np.column_stack([p[:,0]*scale, (SIZE-p[:,1])*scale])
            return (q[:,0].min(), q[:,0].max(), q[:,1].min(), q[:,1].max())
    return bounds_from_geometry(geometry, scale) or bounds_from_mask(mask, scale) or (0.5, 12.0, 0.5, 8.0)


def _wall_segments(geometry, scale):
    segments = []
    for poly in geometry.get("wall", []):
        p = np.asarray(poly.get("outer", []), dtype=float)
        if len(p) < 2:
            continue
        for a,b in zip(p, np.vstack([p[1:],p[:1]])):
            x0,y0=float(a[0]*scale),(SIZE-float(a[1]))*scale
            x1,y1=float(b[0]*scale),(SIZE-float(b[1]))*scale
            length=math.hypot(x1-x0,y1-y0)
            if length >= 0.10:
                segments.append((x0,y0,x1,y1))
    return segments


def cuboid(ax,x0,x1,y0,y1,z0,z1,color,alpha=1.0):
    verts=[[(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],[(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],[(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],[(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],[(x1,y1,z0),(x0,y1,z0),(x0,y1,z1),(x1,y1,z1)],[(x0,y1,z0),(x0,y0,z0),(x0,y0,z1),(x0,y1,z1)]]
    ax.add_collection3d(Poly3DCollection(verts,facecolors=color,edgecolors="none",alpha=alpha))


def add_wall(ax,x0,y0,x1,y1,height,thickness,color):
    dx,dy=x1-x0,y1-y0; L=math.hypot(dx,dy)
    if L<1e-6:return
    nx,ny=-dy/L,dx/L; t=thickness/2
    p0=(x0+nx*t,y0+ny*t);p1=(x1+nx*t,y1+ny*t);p2=(x1-nx*t,y1-ny*t);p3=(x0-nx*t,y0-ny*t)
    verts=[[(p0[0],p0[1],0),(p1[0],p1[1],0),(p1[0],p1[1],height),(p0[0],p0[1],height)],[(p1[0],p1[1],0),(p2[0],p2[1],0),(p2[0],p2[1],height),(p1[0],p1[1],height)],[(p2[0],p2[1],0),(p3[0],p3[1],0),(p3[0],p3[1],height),(p2[0],p2[1],height)],[(p3[0],p3[1],0),(p0[0],p0[1],0),(p0[0],p0[1],height),(p3[0],p3[1],height)]]
    ax.add_collection3d(Poly3DCollection(verts,facecolors=color,edgecolors="none",alpha=.98))


def add_panel(ax,poly,scale,height,color,alpha=.8):
    p=np.asarray(poly.get("outer",[]),dtype=float)
    if len(p)<3:return
    q=np.column_stack([p[:,0]*scale,(SIZE-p[:,1])*scale]); cx,cy=q[:,0].mean(),q[:,1].mean()
    rx=max((q[:,0].max()-q[:,0].min())/2,.025); ry=max((q[:,1].max()-q[:,1].min())/2,.025)
    if rx>=ry:
        x0,x1=cx-rx,cx+rx; y0,y1=cy-.04,cy+.04
    else:
        x0,x1=cx-.04,cx+.04; y0,y1=cy-ry,cy+ry
    cuboid(ax,x0,x1,y0,y1,0,height,color,alpha)


def make_cg(geometry,style_name,lighting,view_name,mask):
    scale=.025; wall_h=2.4; palette=STYLE[style_name]
    xmin,xmax,ymin,ymax=choose_bounds(geometry,mask,scale)
    room_w=max(xmax-xmin,.5); room_d=max(ymax-ymin,.5)
    fig=plt.figure(figsize=(10,7),dpi=160); ax=fig.add_subplot(111,projection="3d")
    fig.patch.set_facecolor("#ECEAE5"); ax.set_facecolor("#ECEAE5")
    cuboid(ax,xmin,xmax,ymin,ymax,0,.06,palette["floor"],1)
    walls=_wall_segments(geometry,scale)
    thickness=max(.07,min(.16,min(room_w,room_d)*.025))
    for s in walls:add_wall(ax,*s,wall_h,thickness,palette["wall"])
    if not walls:
        t=max(.08,min(.14,min(room_w,room_d)*.025))
        cuboid(ax,xmin,xmax,ymin,ymin+t,0,wall_h,palette["wall"]);cuboid(ax,xmin,xmax,ymax-t,ymax,0,wall_h,palette["wall"]);cuboid(ax,xmin,xmin+t,ymin,ymax,0,wall_h,palette["wall"]);cuboid(ax,xmax-t,xmax,ymin,ymax,0,wall_h,palette["wall"])
    for p in geometry.get("door",[]):add_panel(ax,p,scale,2.0,palette["wood"],.95)
    for p in geometry.get("window",[]):add_panel(ax,p,scale,1.25,"#8FAFC4",.7)
    cx,cy=(xmin+xmax)/2,(ymin+ymax)/2
    sofa_w=min(room_w*.42,2.4); sofa_d=min(room_d*.16,.9)
    cuboid(ax,cx-sofa_w/2,cx+sofa_w/2,ymin+room_d*.18,ymin+room_d*.18+sofa_d,.12,.48,palette["accent"])
    table_w=min(room_w*.25,1.5); table_d=min(room_d*.16,.85)
    cuboid(ax,cx-table_w/2,cx+table_w/2,cy-table_d/2,cy+table_d/2,.55,.68,palette["wood"])
    if view_name in ("LDK","キッチン","ダイニング"):
        cuboid(ax,xmin+room_w*.08,xmin+room_w*.46,ymax-room_d*.13,ymax-room_d*.03,.78,.92,palette["wood"])
    if lighting=="夜・間接照明":
        for frac in (.28,.5,.72):
            x=xmin+room_w*frac; cuboid(ax,x-.08,x+.08,ymax-.18,ymax-.08,wall_h-.08,wall_h-.02,"#FFF2C7",.9)
    elif lighting=="夕方・暖色":
        cuboid(ax,xmin+room_w*.35,xmin+room_w*.65,ymax-.10,ymax-.04,wall_h-.05,wall_h,"#FFE2B0",.85)
    elev,azim={"玄関":(18,-65),"キッチン":(20,25),"ダイニング":(22,-25)}.get(view_name,(24,-55))
    ax.view_init(elev=elev,azim=azim); ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax);ax.set_zlim(0,wall_h);ax.set_box_aspect((room_w,room_d,wall_h));ax.set_axis_off()
    fig.tight_layout(pad=.5); buf=io.BytesIO();fig.savefig(buf,format="png",bbox_inches="tight",facecolor=fig.get_facecolor());plt.close(fig)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def color_mask(mask):
    colors=[(238,238,234),(55,55,60),(225,125,55),(55,150,225)]
    out=np.zeros((SIZE,SIZE,3),dtype=np.uint8)
    for i,c in enumerate(colors):out[mask==i]=c
    return Image.fromarray(out)


def overlay(base,mask):
    out=np.asarray(base).copy().astype(np.float32)
    for i,c in ((1,(55,55,60)),(2,(225,125,55)),(3,(55,150,225))):
        m=mask==i;out[m]=out[m]*.4+np.asarray(c)*.6
    return Image.fromarray(np.clip(out,0,255).astype(np.uint8))


def structure_json(geometry,rect,chosen):
    return {"version":3,"canvas_size":[SIZE,SIZE],"content_rect":list(rect),"classes":CLASSES,"polygons":geometry,"preprocess_variant":chosen,"note":"AI解析から生成した近似3Dプレビューです。施工図精度は保証しません。"}


def make_obj(geometry):
    scale=.02; heights={"wall":2.4,"door":2.0,"window":1.2}; lines=["# floorplan-to-3d preview","# Approximate scale only"];vc=0
    for name in ("wall","door","window"):
        for poly in geometry.get(name,[]):
            pts=poly.get("outer",[])
            if len(pts)<3:continue
            base=vc+1
            for x,y in pts:lines.append(f"v {x*scale:.4f} {(SIZE-y)*scale:.4f} 0")
            for x,y in pts:lines.append(f"v {x*scale:.4f} {(SIZE-y)*scale:.4f} {heights[name]:.4f}")
            n=len(pts);lines.append(f"g {name}")
            for i in range(n):
                j=(i+1)%n;lines.append(f"f {base+i} {base+j} {base+n+j} {base+n+i}")
            vc+=n*2
    return "\n".join(lines)+"\n"


st.title("🏠 図面 → 内観CGメーカー")
st.caption("図面の余白・向きを自動補正して構造解析し、3D内観プレビューまで生成します。")
uploaded=st.file_uploader("① 間取り図をアップロード",type=["png","jpg","jpeg","webp"])
style=st.selectbox("② 内装テイスト",list(STYLE.keys()))
lighting=st.selectbox("③ 照明",["昼・自然光","夕方・暖色","夜・間接照明"])
view=st.selectbox("④ 視点",["LDK","リビング","キッチン","ダイニング","玄関"])

if uploaded:
    try:
        original=read_image(uploaded);st.image(original,caption="元の間取り図",use_container_width=True)
    except Exception as e:
        st.error(str(e));st.stop()
    if st.button("🚀 図面からCGを作成",type="primary",use_container_width=True):
        try:
            with st.spinner("AIモデルを準備しています…"):load_model()
            with st.spinner("図面を自動補正して解析しています…"):
                processed,mask,rect,chosen=predict(original);geometry=make_geometry(mask)
            counts={k:len(geometry[k]) for k in ("wall","door","window")}
            c1,c2,c3=st.columns(3)
            with c1:st.image(processed,caption=f"AI入力（自動選択: {chosen}）",use_container_width=True)
            with c2:st.image(color_mask(mask),caption="AI判定",use_container_width=True)
            with c3:st.image(overlay(processed,mask),caption="構造確認",use_container_width=True)
            st.write(f"壁 {counts['wall']} / ドア {counts['door']} / 窓 {counts['window']}")
            if sum(counts.values())==0:
                st.error("構造を認識できませんでした。図面をできるだけ正面から撮影し、余白を減らして再試行してください。")
                st.stop()
            with st.spinner("3D内観CGを生成しています…"):cg=make_cg(geometry,style,lighting,view,mask)
            st.success("3D内観CGの生成が完了しました。")
            st.image(cg,caption=f"3D内観プレビュー｜{view}｜{style}｜{lighting}",use_container_width=True)
            out=io.BytesIO();cg.save(out,format="PNG");st.download_button("📥 内観CG PNG",out.getvalue(),"interior_cg_preview.png","image/png",use_container_width=True)
            data=structure_json(geometry,rect,chosen);st.download_button("📥 構造JSON",json.dumps(data,ensure_ascii=False,indent=2).encode(),"floorplan_structure.json","application/json",use_container_width=True)
            st.download_button("📥 3D OBJ",make_obj(geometry).encode(),"floorplan_structure.obj","text/plain",use_container_width=True)
            st.warning("このCGはAI解析結果から生成する近似3Dプレビューです。壁・窓・ドア位置、寸法、家具配置を施工図レベルでは保証しません。")
        except requests.exceptions.RequestException as e:
            st.error("AIモデル取得中の通信エラーです。時間を置いて再試行してください。")
            with st.expander("詳細"):st.code(repr(e))
        except RuntimeError as e:
            st.error(str(e))
            with st.expander("詳細"):st.exception(e)
        except Exception as e:
            st.error("予期しないエラーが発生しました。")
            with st.expander("詳細"):st.exception(e)
        finally:gc.collect()

st.divider();st.caption("構造モデル: Yytsi/floorplan-to-3d-walls / MIT。公開モデルはCubiCasa5K中心の学習のため、日本の実施設計図では誤認識する場合があります。")
