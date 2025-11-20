import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops
import io
import requests
import time
import base64
import hmac
import hashlib
import uuid
import json

# ==========================================
# 1. 基础配置
# ==========================================
st.set_page_config(page_title="AI 家具设计 (V1标准版)", page_icon="🛋️", layout="wide")

try:
    ACCESS_KEY = st.secrets["LIBLIB_ACCESS_KEY"]
    SECRET_KEY = st.secrets["LIBLIB_SECRET_KEY"]
    MODEL_UUID = st.secrets["LIBLIB_TEMPLATE_UUID"]
except Exception as e:
    st.error("❌ 配置缺失！请在 .streamlit/secrets.toml 中配置 Key 和 UUID")
    st.stop()

# ==========================================
# 2. 签名生成
# ==========================================
def get_liblib_headers(uri):
    timestamp = str(int(time.time() * 1000))
    signature_nonce = str(uuid.uuid4())
    content = '&'.join((uri, timestamp, signature_nonce))
    
    digest = hmac.new(
        SECRET_KEY.encode('utf-8'), 
        content.encode('utf-8'), 
        hashlib.sha1
    ).digest()
    
    sign = base64.urlsafe_b64encode(digest).rstrip(b'=').decode('utf-8')
    
    return {
        "Content-Type": "application/json",
        "AccessKey": ACCESS_KEY,
        "Timestamp": timestamp,
        "SignatureNonce": signature_nonce,
        "Signature": sign
    }

# ==========================================
# 3. 图像处理
# ==========================================
def process_clean_sketch(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    return Image.fromarray(binary)

def process_multiply(render_img, sketch_img):
    if render_img.size != sketch_img.size:
        sketch_img = sketch_img.resize(render_img.size)
    render_img = render_img.convert("RGB")
    sketch_img = sketch_img.convert("RGB")
    return ImageChops.multiply(render_img, sketch_img)

def image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ==========================================
# 4. API 调用 (V1 标准接口)
# ==========================================
def call_liblib_api(prompt, control_image):
    domain = "https://api.liblib.art"
    
    # --- ✅ 修正 1: 使用 V1 标准路径 ---
    submit_uri = "/api/www/v1/generation/image"
    
    base64_img = image_to_base64(control_image)
    
    # --- ✅ 修正 2: 使用 V1 标准参数 (全下划线 snake_case) ---
    # V1 文档规定：generate_params -> controlnet -> units
    payload = {
        "template_uuid": MODEL_UUID, 
        "generate_params": {
            "prompt": prompt + ", interior design, furniture, best quality, 8k",
            "steps": 25,
            "width": 1024,
            "height": 1024,
            "img_count": 1,
            "controlnet": {
                "units": [
                    {
                        "type": "canny", 
                        "weight": 0.8,
                        "image_base64": base64_img
                    }
                ]
            }
        }
    }
    
    headers = get_liblib_headers(submit_uri)
    full_url = domain + submit_uri
    
    try:
        response = requests.post(full_url, headers=headers, json=payload)
        
        # --- 🐞 调试信息 ---
        if response.status_code != 200:
            return None, {
                "URL": full_url,
                "Status": response.status_code,
                "Response Text": response.text,
                "Payload": str(payload)[:200] + "..." 
            }
            
        data = response.json()
        if data.get('code') != 0:
            return None, f"API 业务报错: {data.get('msg')}"
            
        generate_uuid = data['data']['generate_uuid']
        
    except Exception as e:
        return None, f"请求异常: {e}"
    
    # --- 2. 轮询结果 ---
    status_uri = "/api/www/v1/generation/status"
    progress_bar = st.progress(0, text="任务已提交...")
    
    for i in range(60):
        time.sleep(2)
        progress_bar.progress((i + 1) / 60, text="AI 渲染中...")
        
        check_headers = get_liblib_headers(status_uri) 
        try:
            check_res = requests.get(
                domain + status_uri, 
                headers=check_headers, 
                params={"generate_uuid": generate_uuid}
            )
            res_data = check_res.json()
            status = res_data.get('data', {}).get('status')
            
            if status == 1:
                progress_bar.progress(1.0, text="渲染完成！")
                return res_data['data']['images'][0]['image_url'], None
            elif status == -1: 
                return None, f"服务端生成失败"
        except:
            pass
            
    return None, "超时"

# ==========================================
# 5. 界面
# ==========================================
st.title("🛋️ AI 家具设计 (V1标准版)")

uploaded_file = st.file_uploader("上传草图", type=["jpg", "png", "jpeg"])
prompt_text = st.text_area("设计描述", "modern wardrobe, walnut wood, 8k", height=100)
run_btn = st.button("🚀 开始生成", type="primary")

if run_btn and uploaded_file:
    st.write("🧹 清洗草图...")
    uploaded_file.seek(0)
    cleaned_img = process_clean_sketch(uploaded_file)
    st.image(cleaned_img, width=200)
    
    st.write("☁️ 调用 API...")
    img_url, error = call_liblib_api(prompt_text, cleaned_img)
    
    if error:
        st.error("❌ 生成失败！")
        if isinstance(error, dict):
            with st.expander("🐞 点击查看报错详情", expanded=True):
                st.write(f"**Status:** {error['Status']}")
                st.code(error['Response Text'])
        else:
            st.write(error)
        st.stop()
    
    st.success("✅ 成功！")
    generated_response = requests.get(img_url)
    generated_img = Image.open(io.BytesIO(generated_response.content))
    final_img = process_multiply(generated_img, cleaned_img)
    st.image(final_img, caption="最终效果", use_column_width=True)
    
    buf = io.BytesIO()
    final_img.save(buf, format="JPEG", quality=95)
    st.download_button("⬇️ 下载", buf.getvalue(), "design.jpg", "image/jpeg")
