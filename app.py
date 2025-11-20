import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops
import io
import requests
import base64
import json

# ==========================================
# 1. 基础配置
# ==========================================
st.set_page_config(page_title="AI 家具设计 (FLUX版)", page_icon="🛋️", layout="wide")

try:
    API_KEY = st.secrets["SILICONFLOW_API_KEY"]
except Exception as e:
    st.error("❌ 未找到密钥！请在 secrets.toml 中配置 SILICONFLOW_API_KEY")
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

def image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG") # 转为 JPEG 压缩体积
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ==========================================
# 3. 硅基流动 API 调用 (FLUX.1-schnell)
# ==========================================
def call_siliconflow_sd(prompt, control_image):
    
    # 接口地址
    url = "https://api.siliconflow.cn/v1/images/generations"
    
    # 转 Base64
    base64_str = image_to_base64(control_image)
    image_data = f"data:image/jpeg;base64,{base64_str}"
    
    # 构造请求
    payload = {
        # 🚨 核心修改：换成了目前免费且强大的 FLUX 模型
        "model": "black-forest-labs/FLUX.1-schnell",
        "prompt": prompt + ", interior design, furniture, masterpiece, 8k, photorealistic, cinematic lighting",
        "image": image_data, 
        "image_size": "1024x1024",
        "num_inference_steps": 20, # FLUX 只需要很少的步数
        "guidance_scale": 3.5,      # FLUX 推荐较低的引导值
        "prompt_enhancement": False
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        print(f"正在请求模型: {payload['model']}...")
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            return data['data'][0]['url'], None
        else:
            return None, f"API 报错 ({response.status_code}): {response.text}"
            
    except Exception as e:
        return None, f"网络请求异常: {str(e)}"

# ==========================================
# 4. 界面逻辑
# ==========================================
st.title("🛋️ AI 家具设计 (FLUX版)")
st.caption("Powered by SiliconFlow & FLUX.1-schnell")

col_input, col_process = st.columns([1, 1.5])

with col_input:
    uploaded_file = st.file_uploader("上传草图", type=["jpg", "png", "jpeg"])
    prompt_text = st.text_area(
        "设计描述", 
        "modern minimalist wardrobe, walnut wood texture, soft lighting, 8k resolution", 
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
            
            st.write("☁️ 调用云端 GPU (FLUX)...")
            img_url, error = call_siliconflow_sd(prompt_text, cleaned_img)
            
            if error:
                status.update(label="生成失败", state="error")
                st.error(error)
                st.stop()
            
            st.write("📥 下载渲染图...")
            generated_response = requests.get(img_url)
            generated_img = Image.open(io.BytesIO(generated_response.content))
            
            st.write("🎨 合成尺寸标注...")
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
