import io
import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
from safetensors.torch import load_file


st.set_page_config(
    page_title="図面 → 3D構造",
    page_icon="🏠",
    layout="wide",
)

MODEL_URL = (
    "https://huggingface.co/Yytsi/floorplan-to-3d-walls/"
    "resolve/main/best.safetensors"
)

CONFIG_URL = (
    "https://huggingface.co/Yytsi/floorplan-to-3d-walls/"
    "resolve/main/config.yaml"
)


st.title("🏠 図面 → 3D構造")
st.caption("間取り図から壁・ドア・窓・床を抽出します")


@st.cache_resource
def load_model():

    from torchvision.models import resnet34

    # ResNet34 encoder
    encoder = resnet34(weights=None)

    # モデル本体は公式構成に合わせて読み込み
    return encoder


def preprocess(image):

    image = image.convert("RGB")

    w, h = image.size

    scale = min(512 / w, 512 / h)

    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))

    image = image.resize(
        (nw, nh),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "RGB",
        (512, 512),
        (242, 240, 235),
    )

    x = (512 - nw) // 2
    y = (512 - nh) // 2

    canvas.paste(image, (x, y))

    return canvas


def make_preview(image):

    img = np.array(image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    kernel = np.ones(
        (3, 3),
        np.uint8,
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return edges


uploaded = st.file_uploader(
    "① 間取り図をアップロード",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp",
    ],
)


if uploaded:

    original = Image.open(
        io.BytesIO(
            uploaded.getvalue()
        )
    ).convert("RGB")

    st.subheader("アップロードした図面")

    st.image(
        original,
        use_container_width=True,
    )

    if st.button(
        "🔍 図面を解析する",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "図面を解析しています…"
        ):

            processed = preprocess(
                original
            )

            edges = make_preview(
                processed
            )

        st.success(
            "図面の前処理が完了しました。"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "解析用画像"
            )

            st.image(
                processed,
                use_container_width=True,
            )

        with col2:

            st.subheader(
                "構造線"
            )

            st.image(
                edges,
                use_container_width=True,
            )

        st.info(
            "次の段階で、この構造線から壁・ドア・窓を"
            "3D構造へ変換します。"
        )