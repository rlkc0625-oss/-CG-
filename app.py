import io, os, math, json, hashlib, tempfile
from pathlib import Path
import numpy as np
import cv2
import requests
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance
import plotly.graph_objects as go
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union, triangulate
import trimesh

st.set_page_config(page_title='図面→3D内観CG', page_icon='🏠', layout='wide')

MODEL_REPO = "Yytsi/floorplan-to-3d-walls"
MODEL_FILENAME = "best.safetensors"
MODEL_SHA256 = "d7f6a0fd06e2931aecfc8c4849192c5e153701578026efc78d9a6246731a8d6c"
CACHE = Path.home() / ".cache" / "floorplan_cg"
MODEL_PATH = CACHE / MODEL_FILENAME


STYLES = {
    'ナチュラル': {'floor':'#c9a77a','wall':'#f2eee6','ceiling':'#faf9f5','wood':'#b58b5b','accent':'#d8c8b2','glass':'#b8d9e8'},
    'グレージュモダン': {'floor':'#9c9187','wall':'#d8d3cb','ceiling':'#f3f0eb','wood':'#806b5a','accent':'#77716c','glass':'#b7d4df'},
    'ホテルライク': {'floor':'#77716c','wall':'#d6d0c8','ceiling':'#efede8','wood':'#554b45','accent':'#8c7d6d','glass':'#b5d0da'},
    '北欧': {'floor':'#d8be91','wall':'#f4f1e9','ceiling':'#fffdf9','wood':'#c29b69','accent':'#c8d0c6','glass':'#b9d8e5'},
    '和モダン': {'floor':'#6d5b4d','wall':'#e3ddd3','ceiling':'#f3eee5','wood':'#765d4b','accent':'#988a76','glass':'#b5ced6'},
}


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def download_model():
    # Hugging Face Xet/LFS の実体ファイルを取得。
    # ファイルサイズはリポジトリ側の表示と転送実体で差が出ることがあるため、
    # サイズではなく公式SHA256を最終判定に使います。
    CACHE.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and sha256(MODEL_PATH) == MODEL_SHA256:
        return MODEL_PATH
    MODEL_PATH.unlink(missing_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILENAME,
            revision="main",
            cache_dir=str(CACHE / "hf"),
        )
        got = sha256(downloaded)
        if got != MODEL_SHA256:
            raise RuntimeError(f"AIモデルのSHA256検証に失敗しました。取得値: {got}")
        import shutil
        shutil.copy2(downloaded, MODEL_PATH)
        return MODEL_PATH
    except Exception as e:
        raise RuntimeError(f"AIモデルを取得できませんでした: {e}") from e


@st.cache_resource(show_spinner=False)
def load_model():
    import torch
    import segmentation_models_pytorch as smp
    from safetensors.torch import load_file
    p=download_model()
    model=smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=4)
    state=load_file(str(p), device='cpu')
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def read_image(upload):
    data=upload.getvalue()
    if not data: raise ValueError('画像が空です。')
    if len(data)>25*1024*1024: raise ValueError('画像は25MB以下にしてください。')
    im=Image.open(io.BytesIO(data))
    im=ImageOps.exif_transpose(im).convert('RGB')
    if min(im.size)<300: raise ValueError('解像度が低すぎます。')
    if max(im.size)>2400:
        s=2400/max(im.size); im=im.resize((round(im.width*s),round(im.height*s)),Image.Resampling.LANCZOS)
    return im


def deskew_crop(im):
    a=np.asarray(im); g=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
    # background suppression and page quadrilateral
    blur=cv2.GaussianBlur(g,(5,5),0)
    edges=cv2.Canny(blur,40,120)
    cnts,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    H,W=g.shape; best=None
    for c in sorted(cnts,key=cv2.contourArea,reverse=True)[:30]:
        peri=cv2.arcLength(c,True); poly=cv2.approxPolyDP(c,0.02*peri,True)
        area=cv2.contourArea(c)
        if len(poly)==4 and area>0.35*W*H:
            pts=poly.reshape(4,2).astype(np.float32)
            best=pts; break
    if best is None: return im
    def order(p):
        s=p.sum(1); d=np.diff(p,axis=1).ravel(); return np.array([p[np.argmin(s)],p[np.argmin(d)],p[np.argmax(s)],p[np.argmax(d)]],np.float32)
    p=order(best); tl,tr,br,bl=p
    w=max(np.linalg.norm(tr-tl),np.linalg.norm(br-bl)); h=max(np.linalg.norm(bl-tl),np.linalg.norm(br-tr))
    w=int(max(800,min(2400,w))); h=int(max(600,min(2400,h)))
    dst=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],np.float32)
    M=cv2.getPerspectiveTransform(p,dst)
    out=cv2.warpPerspective(a,M,(w,h),borderValue=(255,255,255))
    return Image.fromarray(out)


def letterbox(im,size=512):
    a=np.asarray(im); h,w=a.shape[:2]; s=min(size/w,size/h); nw,nh=max(1,round(w*s)),max(1,round(h*s))
    r=cv2.resize(a,(nw,nh),interpolation=cv2.INTER_AREA)
    canvas=np.full((size,size,3),255,np.uint8); x=(size-nw)//2; y=(size-nh)//2; canvas[y:y+nh,x:x+nw]=r
    return canvas,(s,x,y,nw,nh)


def predict_masks(im,model):
    import torch
    x,meta=letterbox(im,512)
    t=torch.from_numpy(x.astype(np.float32)/255.).permute(2,0,1).unsqueeze(0)
    mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    t=(t-mean)/std
    with torch.inference_mode(): logits=model(t); cls=logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    s,px,py,nw,nh=meta; orig=np.asarray(im).shape[:2]
    crop=cls[py:py+nh,px:px+nw]
    crop=cv2.resize(crop,(orig[1],orig[0]),interpolation=cv2.INTER_NEAREST)
    return crop


def polygons_from_mask(mask, cls, min_area=150):
    b=(mask==cls).astype(np.uint8)
    b=cv2.morphologyEx(b,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,hier=cv2.findContours(b,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if hier is None: return []
    polys=[]
    for i,c in enumerate(contours):
        if hier[0][i][3]!=-1: continue
        if cv2.contourArea(c)<min_area: continue
        ext=cv2.approxPolyDP(c,1.5,True).reshape(-1,2)
        if len(ext)<3: continue
        holes=[]
        child=hier[0][i][2]
        while child!=-1:
            hc=contours[child]
            if cv2.contourArea(hc)>=20:
                holes.append(cv2.approxPolyDP(hc,1.5,True).reshape(-1,2))
            child=hier[0][child][0]
        try:
            p=Polygon(ext,holes=[h for h in holes if len(h)>=3]).buffer(0)
            if not p.is_empty and p.area>=min_area: polys.append(p)
        except Exception: pass
    return polys


def to_world_geom(geom, px_per_m):
    def cvpt(x,y): return (x/px_per_m,-y/px_per_m)
    if isinstance(geom,Polygon):
        ext=[cvpt(x,y) for x,y in geom.exterior.coords]
        holes=[[cvpt(x,y) for x,y in r.coords] for r in geom.interiors]
        return Polygon(ext,holes)
    if isinstance(geom,MultiPolygon): return MultiPolygon([to_world_geom(g,px_per_m) for g in geom.geoms])
    return geom


def explode_polys(g):
    if g.is_empty: return []
    if isinstance(g,Polygon): return [g]
    if isinstance(g,MultiPolygon): return [p for p in g.geoms if p.area>0]
    if isinstance(g,GeometryCollection): return [p for p in g.geoms if isinstance(p,Polygon) and p.area>0]
    return []


def tri_faces(poly,z):
    verts=[]; faces=[]
    tris=triangulate(poly)
    for t in tris:
        if not poly.covers(t.representative_point()): continue
        coords=list(t.exterior.coords)[:3]; base=len(verts); verts += [(x,y,z) for x,y in coords]; faces.append((base,base+1,base+2))
    return verts,faces


def extrude(poly,z0,z1):
    verts=[]; faces=[]
    coords=list(poly.exterior.coords)[:-1]
    if len(coords)<3: return verts,faces
    n=len(coords); verts += [(x,y,z0) for x,y in coords]+[(x,y,z1) for x,y in coords]
    for i in range(n):
        j=(i+1)%n; faces.append((i,j,n+j)); faces.append((i,n+j,n+i))
    top=triangulate(poly)
    for t in top:
        if not poly.covers(t.representative_point()): continue
        tc=list(t.exterior.coords)[:3]; b=len(verts); verts += [(x,y,z1) for x,y in tc]; faces.append((b,b+1,b+2))
    return verts,faces


def build_meshes(mask, px_per_m):
    # labels: 0 floor, 1 wall, 2 door, 3 window
    floor=unary_union(polygons_from_mask(mask,0,max(120,mask.shape[0]*mask.shape[1]*0.00015)))
    wall=unary_union(polygons_from_mask(mask,1,max(40,mask.shape[0]*mask.shape[1]*0.00003)))
    door=unary_union(polygons_from_mask(mask,2,max(30,mask.shape[0]*mask.shape[1]*0.00002)))
    window=unary_union(polygons_from_mask(mask,3,max(30,mask.shape[0]*mask.shape[1]*0.00002)))
    # keep wall only where near floor; then carve openings
    if not wall.is_empty and not floor.is_empty:
        wall=wall.buffer(3).intersection(floor.buffer(80))
    opening=(door.union(window)).buffer(7) if not door.is_empty or not window.is_empty else GeometryCollection()
    if not wall.is_empty and not opening.is_empty: wall=wall.difference(opening)
    floor_world=to_world_geom(floor,px_per_m); wall_world=to_world_geom(wall,px_per_m); door_world=to_world_geom(door,px_per_m); window_world=to_world_geom(window,px_per_m)
    return floor_world,wall_world,door_world,window_world


def scene_mesh(floor,wall,door,window,style):
    s=STYLES[style]; meshes=[]; names=[]
    def add_geom(g,z0,z1,name,kind):
        vv=[]; ff=[]
        for p in explode_polys(g):
            if z1>z0:
                a,b=extrude(p,z0,z1)
            else:
                a,b=tri_faces(p,z0)
            off=len(vv); vv.extend(a); ff.extend([(i+off,j+off,k+off) for i,j,k in b])
        if vv and ff:
            m=trimesh.Trimesh(vertices=np.array(vv),faces=np.array(ff),process=False)
            m.remove_unreferenced_vertices(); meshes.append((name,m,kind))
    add_geom(floor,0,0.06,'floor','floor')
    add_geom(wall,0.06,2.45,'walls','wall')
    # Doors/windows as panels; wall openings are carved above.
    add_geom(door,0.06,2.15,'doors','door')
    add_geom(window,0.95,2.20,'windows','window')
    return meshes


def plot_scene(meshes,style,lighting,view):
    s=STYLES[style]; fig=go.Figure()
    colors={'floor':s['floor'],'wall':s['wall'],'door':s['wood'],'window':s['glass']}
    for name,m,kind in meshes:
        v=m.vertices; f=m.faces
        fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],color=colors.get(kind,s['wall']),opacity=0.92 if kind=='window' else 1.0,flatshading=True,name=name,hoverinfo='skip'))
    eye={'LDK全体':dict(x=1.5,y=-1.5,z=1.2),'リビング→キッチン':dict(x=1.1,y=-1.9,z=1.0),'キッチン→リビング':dict(x=-1.3,y=1.7,z=1.0),'ダイニング→LDK':dict(x=1.7,y=0.8,z=1.0),'玄関':dict(x=-1.7,y=-0.6,z=1.0)}[view]
    if lighting=='夕方・暖色': bg='#e9e0d5'
    elif lighting=='夜・間接照明': bg='#25252a'
    else: bg='#f5f3ef'
    fig.update_layout(height=720,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor=bg,scene=dict(xaxis_visible=False,yaxis_visible=False,zaxis_visible=False,aspectmode='data',camera=dict(eye=eye)))
    return fig


def structural_score(mask):
    areas=[int(np.sum(mask==i)) for i in range(4)]
    total=sum(areas) or 1
    # Healthy plan has meaningful floor/wall pixels, not almost all one class.
    score=0
    if areas[0]/total>0.08: score+=1
    if areas[1]/total>0.005: score+=1
    if areas[2]+areas[3]>20: score+=1
    if max(areas)/total<0.92: score+=1
    return score,areas


st.title('🏠 図面 → 3D内観CG')
st.caption('図面の構造を3Dメッシュへ再構築します。')
upload=st.file_uploader('図面をアップロード',type=['jpg','jpeg','png','webp'])
view=st.selectbox('視点',['LDK全体','リビング→キッチン','キッチン→リビング','ダイニング→LDK','玄関'])
style=st.selectbox('インテリア',['ナチュラル','グレージュモダン','ホテルライク','北欧','和モダン'])
lighting=st.selectbox('照明',['昼・自然光','夕方・暖色','夜・間接照明'])
scale=st.number_input('図面縮尺（1mあたりの画素数）',min_value=20.0,max_value=500.0,value=80.0,step=5.0,help='寸法線が読めない写真でも3D寸法を安定させるための値です。')

if upload:
    try:
        im=deskew_crop(read_image(upload))
        st.image(im,caption='解析対象図面',use_container_width=True)
        if st.button('3D内観を生成',type='primary',use_container_width=True):
            with st.spinner('図面解析 → 形状再構築 → 開口処理 → 3Dメッシュ検証…'):
                model=load_model()
                mask=predict_masks(im,model)
                score,areas=structural_score(mask)
                if score<2: raise RuntimeError('図面の構造認識信頼度が低いため、安全のため3D生成を停止しました。')
                floor,wall,door,window=build_meshes(mask,float(scale))
                if floor.is_empty: raise RuntimeError('室内床領域を抽出できませんでした。')
                if wall.is_empty: raise RuntimeError('壁領域を抽出できませんでした。')
                meshes=scene_mesh(floor,wall,door,window,style)
                if not meshes: raise RuntimeError('3Dメッシュを生成できませんでした。')
                totalv=sum(len(m.vertices) for _,m,_ in meshes); totalf=sum(len(m.faces) for _,m,_ in meshes)
                if totalv<30 or totalf<20: raise RuntimeError('3Dメッシュの情報量が不足しています。')
                fig=plot_scene(meshes,style,lighting,view)
                st.success(f'3D生成完了　床 {areas[0]:,}px / 壁 {areas[1]:,}px / ドア {areas[2]:,}px / 窓 {areas[3]:,}px')
                st.plotly_chart(fig,use_container_width=True)
                # Combined scene exports.
                combined=trimesh.util.concatenate([m for _,m,_ in meshes])
                glb=combined.export(file_type='glb'); obj=combined.export(file_type='obj')
                st.download_button('3D GLB',glb,'floorplan_3d.glb','model/gltf-binary')
                st.download_button('3D OBJ',obj,'floorplan_3d.obj','text/plain')
                structure={'version':'real-3d-rebuild-1','scale_px_per_m':float(scale),'style':style,'lighting':lighting,'view':view,'pixel_area':areas,'mesh_vertices':totalv,'mesh_faces':totalf}
                st.download_button('構造JSON',json.dumps(structure,ensure_ascii=False,indent=2).encode(),'structure.json','application/json')
    except Exception as e:
        st.error(f'生成できませんでした: {type(e).__name__}: {e}')
        if 'AIモデル' not in str(e) and 'huggingface' not in str(e).lower():
            st.info('図面を正面から撮影し、建物部分ができるだけ大きく写った画像で再実行してください。')
