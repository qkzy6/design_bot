import streamlit as st
import os
import requests
import cv2
import numpy as np
from PIL import Image, ImageChops
import io
import dashscope
from dashscope import ImageSynthesis

# ==========================================
# 1. 基础配置 & 依赖检查
# ==========================================
st.set_page_config(page_title="AI 家具设计 (阿里云官方版)", page_icon="🛋️", layout="wide")

# 检查关键模块是否可用（Streamlit Cloud 不支持运行时 pip install）
try:
    from dashscope.file import File
except ImportError:
    st.error("❌ 缺少 dashscope>=1.19.0，请确保 requirements.txt 中已声明该依赖。")
    st.stop()

# 配置 API Key（从 Streamlit Secrets 读取）
try:
    dashscope.api_key = st.secrets["DASHSCOPE_API_KEY"]
except KeyError:
    st.error("❌ 未设置 DASHSCOPE_API_KEY！请在 Streamlit Cloud 后台 → Secrets 中添加：\n\nDASHSCOPE_API_KEY = '你的密钥'")
    st.stop()

# ==========================================
# 2. 图像处理函数（纯 CPU，无 GUI）
# ==========================================
def process_clean_sketch(uploaded_file):
    """清洗草图：去底色，提取黑白线条"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    return Image.fromarray(binary)

def process_multiply(render_img, sketch_img):
    """正片叠底：把线稿叠回渲染图上"""
    if render_img.size != sketch_img.size:
        sketch_img = sketch_img.resize(render_img.size, Image.LANCZOS)
    render_img = render_img.convert("RGB")
    sketch_gray = sketch_img.convert("L")
    sketch_rgb = Image.merge("RGB", (sketch_gray, sketch_gray, sketch_gray))
    return ImageChops.multiply(render_img, sketch_rgb)

# ==========================================
# 3. 调用阿里云万相 API
# ==========================================
def call_aliyun_wanx(prompt, control_image):
    temp_filename = "temp_sketch.png"
    try:
        control_image.save(temp_filename)

        with st.spinner("☁️ 正在上传草图到阿里云..."):
            file_url_obj = File.upload(temp_filename)
            sketch_url = file_url_obj.url

        rsp = ImageSynthesis.call(
            model="wanx-sketch-to-image-v1",
            input={
                'image': sketch_url,
                'prompt': prompt + ", 室内设计, 家具, 8k分辨率, 杰作, 高清材质, 柔和光线"
            },
            n=1,
            size='1024*1024'
        )

        if rsp.status_code == 200:
            return rsp.output.results[0].url, None
        else:
            return None, f"阿里云报错: {rsp.code} - {rsp.message}"

    except Exception as e:
        return None, f"SDK 异常: {str(e)}"
    finally:
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except OSError:
                pass

# ==========================================
# 4. 主界面
# ==========================================
st.title("🛋️ AI 家具设计 (阿里云官方版)")

col_input, col_process = st.columns([1, 1.5])

with col_input:
    uploaded_file = st.file_uploader("上传草图", type=["jpg", "png", "jpeg"])
    prompt_text = st.text_area(
        "设计描述",
        "现代极简风格衣柜，胡桃木纹理，高级灰色调，柔和室内光线，照片级真实感",
        height=120
    )
    run_btn = st.button("🚀 开始生成", type="primary", use_container_width=True)

if run_btn and uploaded_file:
    with col_process:
        with st.status("AI 正在工作中...", expanded=True) as status:

            st.write("🧹 正在清洗草图...")
            uploaded_file.seek(0)
            cleaned_img = process_clean_sketch(uploaded_file)
            st.image(cleaned_img, width=200, caption="清洗后线稿")

            st.write("☁️ 调用阿里云生成...")
            img_url, error = call_aliyun_wanx(prompt_text, cleaned_img)

            if error:
                status.update(label="❌ 生成失败", state="error")
                st.error(error)
                st.stop()

            st.write("📥 下载渲染图...")
            try:
                response = requests.get(img_url, timeout=20)
                response.raise_for_status()
                generated_img = Image.open(io.BytesIO(response.content)).convert("RGB")
            except Exception as e:
                status.update(label="❌ 图像下载失败", state="error")
                st.error(f"无法获取生成结果: {e}")
                st.stop()

            st.write("🎨 合成标注...")
            final_img = process_multiply(generated_img, cleaned_img)
            status.update(label="✅ 全部完成！", state="complete")

        st.image(final_img, caption="最终效果图", use_column_width=True)

        buf = io.BytesIO()
        final_img.save(buf, format="JPEG", quality=95)
        st.download_button(
            "⬇️ 下载高清原图",
            data=buf.getvalue(),
            file_name="design_final.jpg",
            mime="image/jpeg",
            type="primary"
        )
