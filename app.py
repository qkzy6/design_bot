import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops
import io
import os
import dashscope
from dashscope import ImageSynthesis
import requests

# ==========================================
# 1. 基础配置
# ==========================================
st.set_page_config(page_title="AI 家具设计 (阿里云版)", page_icon="🛋️", layout="wide")

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
    """清洗草图：去底色，提取黑白线条"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # 自适应二值化 (C=5 保留细节)
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    return Image.fromarray(binary)

def process_multiply(render_img, sketch_img):
    """正片叠底：把线稿叠回去"""
    # 1. 统一尺寸 (以渲染图为准)
    if render_img.size != sketch_img.size:
        sketch_img = sketch_img.resize(render_img.size)
    
    # 2. 转换模式
    render_img = render_img.convert("RGB")
    sketch_img = sketch_img.convert("RGB")
    
    # 3. 叠底合成
    return ImageChops.multiply(render_img, sketch_img)

# ==========================================
# 3. 阿里云 API 调用逻辑
# ==========================================
def call_aliyun_wanx(prompt, control_image):
    """
    调用通义万相-线稿生图模型
    """
    # 1. 阿里云 SDK 需要本地文件路径
    # 我们把清洗好的图片临时存一下
    temp_filename = "temp_sketch_input.png"
    control_image.save(temp_filename)
    
    # 获取绝对路径，并在前面加上 file:// 协议头
    local_file_uri = f"file://{os.path.abspath(temp_filename)}"

    try:
        # 2. 发起生成请求 (同步调用，简单直接)
        # 文档：https://help.aliyun.com/zh/dashscope/developer-reference/api-details-9
        rsp = ImageSynthesis.call(
            model="wanx-sketch-to-image-v1", # 专门的线稿生图模型
            input={
                'image': local_file_uri,
                'prompt': prompt + ", 室内设计, 家具, 8k分辨率, 杰作, 高清材质, 柔和光线"
            },
            n=1,
            size='1024*1024'
        )
        
        # 3. 处理结果
        if rsp.status_code == 200:
            # 获取图片 URL
            img_url = rsp.output.results[0].url
            return img_url, None
        else:
            # 报错
            return None, f"阿里云报错: {rsp.code} - {rsp.message}"
            
    except Exception as e:
        return None, f"SDK 调用异常: {str(e)}"

# ==========================================
# 4. 界面逻辑
# ==========================================
st.title("🛋️ AI 家具设计 (阿里云引擎)")

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
            # 展示一下清洗结果，让用户放心
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
