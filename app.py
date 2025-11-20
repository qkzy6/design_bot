import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops
import io
import os
import requests
import dashscope
from dashscope import ImageSynthesis

# ==========================================
# 1. 基础配置
# ==========================================
st.set_page_config(page_title="AI 家具设计 (阿里云无依赖版)", page_icon="🛋️", layout="wide")

# 读取并设置 API Key
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
    """清洗草图"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    return Image.fromarray(binary)

def process_multiply(render_img, sketch_img):
    """正片叠底"""
    if render_img.size != sketch_img.size:
        sketch_img = sketch_img.resize(render_img.size)
    render_img = render_img.convert("RGB")
    sketch_img = sketch_img.convert("RGB")
    return ImageChops.multiply(render_img, sketch_img)

# ==========================================
# 3. 核心：临时文件上传助手 (替代 dashscope.file)
# ==========================================
def get_public_url(local_file_path):
    """
    将本地文件上传到 file.io 临时网盘，获取公网 URL
    (解决阿里云无法读取本地文件的问题，且不需要安装额外SDK)
    """
    url = "https://file.io"
    try:
        with open(local_file_path, 'rb') as f:
            # file.io 免费，文件被下载一次后自动删除，非常适合这种临时中转
            response = requests.post(url, files={"file": f})
        
        if response.status_code == 200:
            return response.json()["link"]
        else:
            print(f"上传失败: {response.text}")
            return None
    except Exception as e:
        print(f"上传异常: {e}")
        return None

# ==========================================
# 4. 阿里云 API 调用逻辑
# ==========================================
def call_aliyun_wanx(prompt, control_image):
    """
    调用通义万相-线稿生图模型
    """
    # 1. 保存临时文件
    temp_filename = "temp_sketch_input.png"
    control_image.save(temp_filename)
    
    try:
        # --- 🚨 核心修改：使用通用 HTTP 上传，不依赖 SDK ---
        with st.spinner("☁️ 正在上传草图到中转服务器..."):
            sketch_cloud_url = get_public_url(temp_filename)
            
        if not sketch_cloud_url:
            return None, "图片上传失败，无法获取公网链接"
            
        # 2. 发起生成请求
        # 文档：https://help.aliyun.com/zh/dashscope/developer-reference/api-details-9
        rsp = ImageSynthesis.call(
            model="wanx-sketch-to-image-v1", 
            prompt=prompt + ", 室内设计, 家具, 8k分辨率, 杰作, 高清材质, 柔和光线",
            sketch_image_url=sketch_cloud_url, # 传入 file.io 的链接
            n=1,
            size='1024*1024'
        )
        
        # 3. 处理结果
        if rsp.status_code == 200:
            img_url = rsp.output.results[0].url
            return img_url, None
        else:
            return None, f"阿里云报错: {rsp.code} - {rsp.message}"
            
    except Exception as e:
        return None, f"调用异常: {str(e)}"

# ==========================================
# 5. 界面逻辑
# ==========================================
st.title("🛋️ AI 家具设计 (阿里云稳健版)")

col_input, col_process = st.columns([1, 1.5])

with col_input:
    st.markdown("### 1. 上传草图")
    uploaded_file = st.file_uploader("请上传家具手绘图", type=["jpg", "png", "jpeg"])
    
    st.markdown("### 2. 设计要求")
    prompt_text = st.text_area(
        "描述", 
        "现代极简风格衣柜，胡桃木纹理，高级灰色调，柔和室内光线，照片级真实感", 
        height=120
    )
    
    run_btn = st.button("🚀 开始生成", type="primary", use_container_width=True)

if run_btn and uploaded_file:
    with col_process:
        st.markdown("### 3. 生成结果")
        
        with st.status("AI 正在工作中...", expanded=True) as status:
            
            st.write("🧹 正在清洗草图...")
            uploaded_file.seek(0)
            cleaned_img = process_clean_sketch(uploaded_file)
            st.image(cleaned_img, width=200, caption="清洗后线稿")
            
            st.write("☁️ 正在调用阿里云 (通义万相)...")
            img_url, error = call_aliyun_wanx(prompt_text, cleaned_img)
            
            if error:
                status.update(label="生成失败", state="error")
                st.error(error)
                st.stop()
            
            st.write("📥 下载渲染图...")
            generated_response = requests.get(img_url)
            generated_img = Image.open(io.BytesIO(generated_response.content))
            
            st.write("🎨 正在合成尺寸标注...")
            final_img = process_multiply(generated_img, cleaned_img)
            
            status.update(label="✅ 全部完成！", state="complete")

        st.image(final_img, caption="最终效果图", use_column_width=True)
        
        # 下载按钮
        buf = io.BytesIO()
        final_img.save(buf, format="JPEG", quality=95)
        st.download_button(
            "⬇️ 下载高清原图", 
            data=buf.getvalue(), 
            file_name="design_final.jpg", 
            mime="image/jpeg", 
            type="primary"
        )
