import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops
import io
import requests
import base64
import json
import time

# ==========================================
# 1. 基础配置
# ==========================================
st.set_page_config(page_title="AI 家具设计 (百度千帆 V1 版)", page_icon="🛋️", layout="wide")

try:
    # 🚨 核心：必须同时读取 API Key (client_id) 和 Secret Key (client_secret)
    API_KEY = st.secrets["BAIDU_API_KEY"]
    SECRET_KEY = st.secrets["BAIDU_SECRET_KEY"]
except Exception as e:
    st.error("❌ 配置缺失！请在 secrets.toml 中配置 BAIDU_API_KEY 和 BAIDU_SECRET_KEY")
    st.stop()

# ==========================================
# 2. 鉴权逻辑 (获取 Access Token)
# ==========================================

@st.cache_data(ttl=60*60*24*30) 
def get_access_token(api_key, secret_key):
    """
    第一步：使用 AK/SK 获取临时的 Access Token (缓存 30 天)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": api_key,
        "client_secret": secret_key
    }
    try:
        # 使用 requests 发起 POST 请求
        response = requests.post(url, params=params, timeout=10)
        
        if response.status_code == 200:
            return response.json().get("access_token")
        else:
            print(f"Token Request Failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Token 获取异常: {e}")
        return None

# ==========================================
# 3. 图像处理函数 (本地 CPU)
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
# 4. API 调用逻辑 (核心业务)
# ==========================================

def call_baidu_sdxl(prompt, control_image):
    """
    调用百度千帆 Stable-Diffusion-XL (图生图模式)
    """
    # 1. 获取 Access Token
    token = get_access_token(API_KEY, SECRET_KEY)
    if not token:
        return None, "无法获取 Access Token，请检查 AK/SK 或权限。"

    # 2. 构造请求 URL (使用 Access Token)
    url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/text2image/sd_xl?access_token={token}"
    
    # 3. Base64 传输图片
    base64_img = image_to_base64(control_image)
    
    payload = {
        "prompt": prompt + ", interior design, furniture, 8k, photorealistic",
        "negative_prompt": "blurry, low quality, watermark, text, messy lines",
        "size": "1024x1024",
        "steps": 30,
        "n": 1,
        "image": base64_img, 
        "strength": 0.75,    
        "sampler_index": "DPM++ SDE Karras"
    }
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
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
# 5. 界面逻辑
# ==========================================
st.title("🛋️ AI 家具设计 (百度千帆 V1/OAuth 版)")

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
            
            st.write("☁️ 调用百度 SDXL (OAuth 鉴权)...")
            img_b64, error = call_baidu_sdxl(prompt_text, cleaned_img)
            
            if error:
                status.update(label="失败", state="error")
                st.error(error)
                st.stop()
            
            st.write("🎨 合成标注...")
            generated_img = Image.open(io.BytesIO(base64.b64decode(img_b64)))
            
            final_img = process_multiply(generated_img, cleaned_img)
            status.update(label="✅ 完成！", state="complete")

        st.image(final_img, caption="最终效果", use_container_width=True)
        
        buf = io.BytesIO()
        final_img.save(buf, format="JPEG")
        st.download_button("⬇️ 下载", data=buf.getvalue(), file_name="design.jpg", mime="image/jpeg", type="primary")
