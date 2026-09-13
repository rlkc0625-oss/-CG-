import json, math
from io import BytesIO
from pathlib import Path
import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps

APP_VERSION='3.0.0'
st.set_page_config(page_title='House3D Studio', page_icon='🏠', layout='wide')

# ---------- numeric geometry ----------
def num(v,d=0.0):
    try:
        x=float(v); return x if math.isfinite(x) else d
    except Exception: return d

def vec(a):
    r=math.radians(num(a)); return np.array([math.cos(r),math.sin(r)],float)

def pwall(w,o): return np.array([num(w.get('x')),num(w.get('y'))])+vec(w.get('angle'))*o

def wend(w): return pwall(w,num(w.get('length')))

def proj(w,p):
    p0=np.array([num(w.get('x')),num(w.get('y'))]); return float(np.dot(np.asarray(p)-p0,vec(w.get('angle'))))

def dist(w,p):
    p0=np.array([num(w.get('x')),num(w.get('y'))]); d=np.asarray(p)-p0; return abs(float(np.cross(vec(w.get('angle')),d)))

def box(x,y,w,d,z,h,a=0):
    u,v=vec(a),vec(a+90); p=np.array([x,y]); q=[p,p+u*w,p+u*w+v*d,p+v*d]
    return [(q[i][0],q[i][1],z) for i in range(4)]+[(q[i][0],q[i][1],z+h) for i in range(4)],[(0,1,2,3),(4,7,6,5),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)]

def seg(p1,p2,t,z,h):
    p1,p2=np.asarray(p1,float),np.asarray(p2,float); d=p2-p1; L=np.linalg.norm(d)
    if L<1e-8:return [],[]
    u=d/L; n=np.array([-u[1],u[0]])*t/2; q=[p1+n,p2+n,p2-n,p1-n]
    return [(q[i][0],q[i][1],z) for i in range(4)]+[(q[i][0],q[i][1],z+h) for i in range(4)],[(0,1,2,3),(4,7,6,5),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)]

def assign(walls, openings, tol=.22):
    out={i:[] for i in range(len(walls))}; warnings=[]
    for kind,o in openings:
        p=np.array([num(o.get('x')),num(o.get('y'))]); best=None
        for i,w in enumerate(walls):
            off=proj(w,p); wd=dist(w,p); half=num(o.get('width'))/2
            if wd<=tol and -half-.02<=off<=num(w.get('length'))+half+.02:
                if best is None or wd<best[0]:best=(wd,i,off)
        if best is None:warnings.append(f'{kind}{o.get("id","")} が壁に正しく配置されていません。')
        else:out[best[1]].append((kind,o,best[2]))
    return out,warnings

def wall_parts(w,ops):
    L=num(w.get('length')); spans=[]
    for kind,o,off in ops:
        lo=max(0,off-num(o.get('width'))/2); hi=min(L,off+num(o.get('width'))/2)
        if hi>lo:spans.append((lo,hi,kind,o))
    spans.sort(); merged=[]
    for s in spans:
        if merged and s[0]<merged[-1][1]-1e-7: return [],spans,True
        merged.append(s)
    cur=0; parts=[]
    for lo,hi,_,_ in merged:
        if lo>cur+1e-7:parts.append((cur,lo))
        cur=hi
    if cur<L-1e-7:parts.append((cur,L))
    return parts,spans,False

def validate(walls,doors,windows,furniture):
    e=[]; w=[]
    for i,x in enumerate(walls,1):
        for k in ('length','thickness','height'):
            if num(x.get(k))<=0:e.append(f'壁{i}: {k} が不正です。')
    for i,x in enumerate(doors,1):
        if num(x.get('width'))<=0 or num(x.get('height'))<=0:e.append(f'ドア{i}: 寸法が不正です。')
    for i,x in enumerate(windows,1):
        if num(x.get('width'))<=0 or num(x.get('height'))<=0 or num(x.get('sill'))<0:e.append(f'窓{i}: 寸法が不正です。')
    for i,x in enumerate(furniture,1):
        if any(num(x.get(k))<=0 for k in ('width','depth','height')):e.append(f'家具{i}: 寸法が不正です。')
    ass,aw=assign(walls,[('door',x) for x in doors]+[('window',x) for x in windows]);w+=aw
    for i,ops in ass.items():
        _,sp,over=wall_parts(walls[i],ops)
        if over:e.append(f'壁{i+1}: 開口部が重なっています。')
        for lo,hi,kind,o in sp:
            if kind=='window' and num(o.get('sill'))+num(o.get('height'))>num(walls[i].get('height'))+1e-6:e.append(f'壁{i+1}: 窓上端が壁高さを超えています。')
    return e,w,ass

def mesh_figure(walls,doors,windows,furniture):
    fig=go.Figure(); ass,_,_=validate(walls,doors,windows,furniture)
    assigned,_=assign(walls,[('door',x) for x in doors]+[('window',x) for x in windows])
    def add(v,f,name,op=.85):
        if not v:return
        xs,ys,zs=zip(*v); ii=[];jj=[];kk=[]
        for a,b,c,d in f:ii += [a,a];jj += [b,c];kk += [c,d]
        fig.add_trace(go.Mesh3d(x=xs,y=ys,z=zs,i=ii,j=jj,k=kk,name=name,opacity=op))
    for i,ww in enumerate(walls):
        parts,_,_=wall_parts(ww,assigned.get(i,[]))
        for j,(a,b) in enumerate(parts):
            add(*seg(pwall(ww,a),pwall(ww,b),num(ww.get('thickness'),.15),0,num(ww.get('height'),2.7)),f'壁{i+1}-{j+1}')
    for i,d in enumerate(doors):
        hosts=[ww for ww in walls if dist(ww,[num(d.get('x')),num(d.get('y'))])<.22]
        if hosts:
            ww=min(hosts,key=lambda q:dist(q,[num(d.get('x')),num(d.get('y'))])); off=proj(ww,[num(d.get('x')),num(d.get('y'))]); c=pwall(ww,off)
            add(*box(c[0]-num(d.get('width'))/2,c[1]-0.018,num(d.get('width')),0.036,0,num(d.get('height'),2.0),num(ww.get('angle'))),f'ドア{i+1}',.65)
    for i,n in enumerate(windows):
        hosts=[ww for ww in walls if dist(ww,[num(n.get('x')),num(n.get('y'))])<.22]
        if hosts:
            ww=min(hosts,key=lambda q:dist(q,[num(n.get('x')),num(n.get('y'))])); off=proj(ww,[num(n.get('x')),num(n.get('y'))]); c=pwall(ww,off)
            add(*box(c[0]-num(n.get('width'))/2,c[1]-.018,num(n.get('width')),0.036,num(n.get('sill'),.9),num(n.get('height'),1.2),num(ww.get('angle'))),f'窓{i+1}',.5)
    for i,r in enumerate(furniture):add(*box(num(r.get('x')),num(r.get('y')),num(r.get('width')),num(r.get('depth')),num(r.get('z')),num(r.get('height')),num(r.get('angle'))),f'家具{i+1}',.7)
    fig.update_layout(scene=dict(xaxis_title='X m',yaxis_title='Y m',zaxis_title='Z m',aspectmode='data',camera=dict(eye=dict(x=1.5,y=1.5,z=1.2))),margin=dict(l=0,r=0,t=0,b=0))
    return fig

def obj_export(walls,doors,windows,furniture):
    V=[];F=[];groups=[]
    def add(v,f,g):
        base=len(V)+1;V.extend(v);F.extend((g,tuple(i+base for i in face)) for face in f)
    assigned,_=assign(walls,[('door',x) for x in doors]+[('window',x) for x in windows])
    for i,w in enumerate(walls):
        parts,_,_=wall_parts(w,assigned.get(i,[]))
        for j,(a,b) in enumerate(parts):add(*seg(pwall(w,a),pwall(w,b),num(w.get('thickness'),.15),0,num(w.get('height'),2.7)),f'wall_{i+1}_{j+1}')
    for i,r in enumerate(furniture):add(*box(num(r.get('x')),num(r.get('y')),num(r.get('width')),num(r.get('depth')),num(r.get('z')),num(r.get('height')),num(r.get('angle'))),f'furniture_{i+1}')
    s=[f'# House3D Studio {APP_VERSION}']+[f'v {x:.6f} {y:.6f} {z:.6f}' for x,y,z in V];cur=None
    for g,f in F:
        if g!=cur:s.append('g '+g);cur=g
        s.append('f '+' '.join(map(str,f)))
    return '\n'.join(s)+'\n'

# ---------- floorplan image -> calibrated 3D wall mask ----------
def preprocess_plan(img, threshold=185, close_px=3, min_component=80):
    a=np.array(img.convert('L'))
    # dark-line extraction; adaptive threshold handles scans/photos better than a fixed RGB test
    bw=cv2.adaptiveThreshold(a,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,threshold//12 if threshold>0 else 10)
    if close_px>0:
        k=np.ones((close_px,close_px),np.uint8); bw=cv2.morphologyEx(bw,cv2.MORPH_CLOSE,k)
    # remove tiny text/noise components conservatively
    n,lab,stats,_=cv2.connectedComponentsWithStats(bw,8); clean=np.zeros_like(bw)
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]>=min_component:clean[lab==i]=255
    return clean

def mask_to_mesh(mask, scale_m_per_px, wall_h=2.7, max_dim=1400):
    # downsample only when enormous; preserve physical scale through scale factor
    h,w=mask.shape
    if max(h,w)>max_dim:
        f=max_dim/max(h,w); mask=cv2.resize(mask,(int(w*f),int(h*f)),interpolation=cv2.INTER_AREA); scale_m_per_px/=f
    # Morphological skeleton-ish thick wall regions. We extrude connected filled regions.
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    verts=[];faces=[]
    for ci,c in enumerate(contours):
        area=cv2.contourArea(c)
        if area<50:continue
        eps=max(0.8,0.002*cv2.arcLength(c,True)); p=cv2.approxPolyDP(c,eps,True).reshape(-1,2)
        if len(p)<3:continue
        base=len(verts)
        # image y is inverted into model Y
        for x,y in p: verts.append((float(x)*scale_m_per_px,float(-y)*scale_m_per_px,0.0))
        for x,y in p: verts.append((float(x)*scale_m_per_px,float(-y)*scale_m_per_px,wall_h))
        n=len(p)
        # bottom/top fan; side quads
        for i in range(1,n-1):faces.append((base,base+i,base+i+1));faces.append((base+n,base+n+i+1,base+n+i))
        for i in range(n):j=(i+1)%n;faces.append((base+i,base+j,base+n+j,base+n+i))
    return verts,faces

def fig_raw_mesh(v,f):
    fig=go.Figure()
    if v and f:
        xs,ys,zs=zip(*v);ii=[];jj=[];kk=[]
        for face in f:
            if len(face)==4:
                a,b,c,d=face;ii += [a,a];jj += [b,c];kk += [c,d]
            else:a,b,c=face;ii.append(a);jj.append(b);kk.append(c)
        fig.add_trace(go.Mesh3d(x=xs,y=ys,z=zs,i=ii,j=jj,k=kk,opacity=.88,name='自動生成壁'))
    fig.update_layout(scene=dict(xaxis_title='X m',yaxis_title='Y m',zaxis_title='Z m',aspectmode='data'),margin=dict(l=0,r=0,t=0,b=0))
    return fig

def rows(kind):
    if kind=='walls':return [{'id':'W1','x':0.,'y':0.,'length':6.,'thickness':.15,'height':2.7,'angle':0.}]
    if kind=='doors':return [{'id':'D1','x':3.,'y':0.,'width':.9,'height':2.,'angle':0.}]
    if kind=='windows':return [{'id':'N1','x':5.,'y':0.,'width':1.6,'height':1.2,'sill':.9,'angle':0.}]
    return [{'id':'F1','x':1.,'y':1.,'width':2.,'depth':.9,'height':.8,'z':0.,'angle':0.}]

def rec(df):return [dict(r) for r in df]

# ---------- UI ----------
st.title('🏠 House3D Studio')
st.caption('図面画像からの自動3D生成と、数値確定モデルからの正確な3D生成を同じアプリで扱います。')
tabs=st.tabs(['📐 図面→自動3D','🧱 正確モデル→3D','💾 プロジェクト'])

with tabs[0]:
    st.subheader('図面画像 → 3D')
    up=st.file_uploader('平面図をアップロード',type=['png','jpg','jpeg','webp'],key='plan')
    if up:
        try:
            img=ImageOps.exif_transpose(Image.open(BytesIO(up.getvalue())).convert('RGB'))
            c1,c2=st.columns([1,1])
            with c1:st.image(img,caption='入力図面',use_container_width=True)
            with c2:
                st.markdown('**実寸キャリブレーション**')
                known=st.number_input('図面の既知の横幅（m）',min_value=.1,value=10.,step=.1)
                use_width=st.checkbox('図面画像の横幅を既知寸法として使用',value=True)
                th=st.slider('線の抽出感度',80,240,185)
                close=st.slider('線の連結',1,9,3)
                wall_h=st.number_input('壁高さ（m）',min_value=.1,value=2.7,step=.1)
                if use_width:
                    scale=known/img.width
                    st.info(f'縮尺: 1 px = {scale:.6f} m')
                    if st.button('▶ 自動3D生成',type='primary',use_container_width=True):
                        mask=preprocess_plan(img,th,close,max(30,int(img.width*img.height/200000)))
                        v,f=mask_to_mesh(mask,scale,wall_h)
                        st.session_state['auto_mesh']=(v,f,mask,scale)
            if 'auto_mesh' in st.session_state:
                v,f,mask,scale=st.session_state['auto_mesh']
                st.success(f'3D生成完了：{len(v)} vertices / {len(f)} faces')
                st.plotly_chart(fig_raw_mesh(v,f),use_container_width=True,config={'displaylogo':False,'scrollZoom':True})
                st.download_button('📥 自動生成OBJ',obj_export([],[],[],[]) if False else ('# House3D auto mesh\n'+'\n'.join(f'v {x:.6f} {y:.6f} {z:.6f}' for x,y,z in v)+'\n'+'\n'.join('f '+' '.join(str(i+1) for i in face) for face in f)), 'auto_floorplan_3d.obj','text/plain',use_container_width=True)
        except Exception as e:st.error(f'図面処理エラー: {e}')
    else:st.info('平面図をアップロードすると、自動3D生成を実行できます。')

with tabs[1]:
    st.subheader('数値モデル → 正確な3D')
    a,b=st.columns(2)
    with a:
        wdf=st.data_editor(rows('walls'),num_rows='dynamic',use_container_width=True,key='w')
        ddf=st.data_editor(rows('doors'),num_rows='dynamic',use_container_width=True,key='d')
    with b:
        ndf=st.data_editor(rows('windows'),num_rows='dynamic',use_container_width=True,key='n')
        fdf=st.data_editor(rows('furniture'),num_rows='dynamic',use_container_width=True,key='f')
    walls,doors,windows,furniture=map(rec,(wdf,ddf,ndf,fdf));errors,warns,_=validate(walls,doors,windows,furniture)
    for e in errors:st.error(e)
    for w in warns:st.warning(w)
    if not errors:
        st.plotly_chart(mesh_figure(walls,doors,windows,furniture),use_container_width=True,config={'displaylogo':False,'scrollZoom':True})
        st.download_button('📥 OBJを書き出す',obj_export(walls,doors,windows,furniture),'house3d.obj','text/plain',use_container_width=True)

with tabs[2]:
    st.subheader('プロジェクト保存・復元')
    project=st.text_area('JSON',value=json.dumps({'schema':'house3d.project','version':APP_VERSION,'units':'m'},ensure_ascii=False,indent=2),height=240)
    st.download_button('📥 JSON保存',project,'house3d_project.json','application/json')
    st.caption('建築形状は数値モデルを正とし、画像自動生成はキャリブレーション値を明示して生成します。')

st.divider();st.caption(f'House3D Studio v{APP_VERSION}')
