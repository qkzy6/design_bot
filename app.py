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
st.set_page_config(page_title="AI 家具设计 (百度千帆 V2 版)", page_icon="🛋️", layout="wide")

try:
    # 🚨 核心修改：只读取一个 API Key
    API_KEY = st.secrets["BAIDU_API_KEY"]
except Exception as e:
    st.error("❌ 未找到密钥！请在 secrets.toml 中配置 BAIDU_API_KEY")
    st.stop()

# ==========================================
# 2. 图像处理函数 (不变)
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
    """图片转 Base64 字符串"""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG") 
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ==========================================
# 3. 百度千帆 API 调用逻辑 (核心)
# ==========================================

def call_baidu_sdxl(prompt, control_image):
    """
    调用百度千帆 Stable-Diffusion-XL (图生图模式)
    使用单 API Key 作为 Access Token
    """
    # 🚨 核心修改：URL 中直接使用 API_KEY 作为 access_token
    # 假设 API Key 已经具备访问 SDXL 的权限
    url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/text2image/sd_xl?access_token={API_KEY}"
    
    base64_img = image_to_base64(control_image)
    
    payload = {
        "prompt": prompt + ", interior design, furniture, 8k, photorealistic",
        "negative_prompt": "blurry, low quality, watermark, text, messy lines",
        "size": "1024x1024",
        "steps": 30,
        "n": 1,
        "image": base64_img, # Base64 图生图输入
        "strength": 0.75,    
        "sampler_index": "DPM++ SDE Karras"
    }
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
        # 移除 token 获取步骤，直接发请求
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        data = response.json()
        
        if "data" in data and len(data["data"]) > 0:
            return data["data"][0]["b64_image"], None
        else:
            # 捕获权限和业务错误
            return None, f"百度 API 业务报错: {data.get('error_msg', data.get('error_code', str(data)))}"
            
    except Exception as e:
        return None, f"请求异常: {str(e)}"

# ==========================================
# 4. 界面逻辑
# ==========================================
st.title("🛋️ AI 家具设计 (百度千帆 V2 版)")

col_input, col_process = st.columns([1, 1.5])

with col_input:
    uploaded_file = st.file_uploader("上传草图", type=["jpg", "png", "jpeg"])
    prompt_text = st.text_area("设计描述", "modern wardrobe, walnut wood texture, soft lighting", height=120)
    run_btn = st.button("🚀 开始生成", type="primary", use_container_width=True)

if run_btn and uploaded_file:
    with col_process:
        with st.status("AI 正在工作中...", expanded=True) as status:
            
            st.write("🧹 正在清洗草图...")
            uploaded_file.seek(0)
            cleaned_img = process_clean_sketch(uploaded_file)
            st.image(cleaned_img, width=200, caption="清洗后线稿")
            
            st.write("☁️ 调用百度 SDXL (Base64传输)...")
            img_b64, error = call_baidu_sdxl(prompt_text, cleaned_img)
            
            if error:
                status.update(label="失败", state="error")
                st.error(error)
                st.stop()
            
            st.write("🎨 合成标注...")
            generated_img = Image.open(io.BytesIO(base64.b64decode(img_b64)))
            final_img = process_multiply(generated_img, cleaned_img)
            status.update(label="✅ 完成！", state="complete")

        st.image(final_img, caption="最终效果", use_column_width=True)
        
        buf = io.BytesIO()
        final_img.save(buf, format="JPEG")
        st.download_button("⬇️ 下载", data=buf.getvalue(), file_name="design.jpg", mime="image/jpeg", type="primary")
