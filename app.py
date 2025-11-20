import streamlit as st
import subprocess
import sys
import time
import os
import requests  # ← 新增：用于下载生成的图片

# ==========================================
# 0. 自动环境修复 (核武器级补丁)
# ==========================================
try:
    from dashscope.file import File
except ImportError:
    st.warning("⚠️ 检测到阿里云 SDK 版本过低，正在自动升级... (请等待约 30 秒)")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "dashscope>=1.19.0"])
        st.success("✅ 升级成功！正在重启应用...")
        time.sleep(2)
        st.experimental_rerun()  # ← 替换为实验性 rerun（兼容新旧版本）
    except Exception as e:
        st.error(f"自动升级失败，请手动执行: pip install --upgrade dashscope>=1.19.0\n错误: {e}")
        st.stop()

# 正常导入其他库
import cv2
import numpy as np
from PIL import Image, ImageChops
import io
import dashscope
from dashscope import ImageSynthesis

# ==========================================
# 1. 基础配置
# ==========================================
st.set_page_config(page_title="AI 家具设计 (阿里云官方版)", page_icon="🛋️", layout="wide")

# 读取密钥
try:
    api_key = st.secrets["DASHSCOPE_API_KEY"]
    dashscope.api_key = api_key
except Exception as e:
    st.error("❌ 未找到密钥！请在 .streamlit/secrets.toml 中配置 DASHSCOPE_API_KEY")
    st.stop()

# ==========================================
# 2. 图像处理函数 (本地 CPU)
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
    """正片叠底：把线稿叠回去"""
    if render_img.size != sketch_img.size:
        sketch_img = sketch_img.resize(render_img.size, Image.LANCZOS)
    render_img = render_img.convert("RGB")
    sketch_img = sketch_img.convert("L")  # 转为灰度更合理
    sketch_rgb = Image.merge("RGB", (sketch_img, sketch_img, sketch_img))
    return ImageChops.multiply(render_img, sketch_rgb)

# ==========================================
# 3. 阿里云 API 调用 (含官方上传)
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
            img_url = rsp.output.results[0].url
            return img_url, None
        else:
            return None, f"阿里云报错: {rsp.code} - {rsp.message}"

    except Exception as e:
        return None, f"SDK 异常: {str(e)}"
    finally:
        # 确保临时文件被删除
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# ==========================================
# 4. 界面逻辑
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
                generated_response = requests.get(img_url, timeout=15)
                generated_response.raise_for_status()
                generated_img = Image.open(io.BytesIO(generated_response.content)).convert("RGB")
            except Exception as e:
                status.update(label="❌ 下载失败", state="error")
                st.error(f"无法下载生成图像: {e}")
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
