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
st.set_page_config(page_title="AI 家具设计 (调试版)", page_icon="🐞", layout="wide")

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
# 4. API 调用 (含调试信息)
# ==========================================
def call_liblib_api(prompt, control_image):
    # --- 尝试 1: 使用 WebUI 接口 (基于你的文档截图) ---
    domain = "https://api.liblib.art"
    submit_uri = "/api/generate/webui/text2img"
    
    # 准备数据
    base64_img = image_to_base64(control_image)
    payload = {
        "templateUuid": MODEL_UUID, 
        "generateParams": {
            "prompt": prompt + ", interior design, furniture, best quality, 8k",
            "steps": 25,
            "width": 1024,
            "height": 1024,
            "imgCount": 1,
            "controlNet": [
                {
                    "enabled": True,
                    "module": "canny", 
                    # 尝试使用通用模型名，防止模型不匹配
                    "model": "control_v11p_sd15_canny", 
                    "image": base64_img,
                    "weight": 0.8
                }
            ]
        }
    }
    
    headers = get_liblib_headers(submit_uri)
    full_url = domain + submit_uri
    
    try:
        response = requests.post(full_url, headers=headers, json=payload)
        
        # --- 🐞 遇到错误时，返回详细调试信息 ---
        if response.status_code != 200:
            debug_info = {
                "URL": full_url,
                "Status": response.status_code,
                "Headers Sent": headers,
                "Response Text": response.text,
                "Payload": str(payload)[:200] + "..." # 只截取一部分防止太长
            }
            return None, debug_info # 返回 debug 字典
            
        data = response.json()
        if data.get('code') != 0:
            return None, f"API 业务报错: {data.get('msg')}"
            
        generate_uuid = data['data']['generateUuid']
        
    except Exception as e:
        return None, f"请求异常: {e}"
    
    # --- 轮询 ---
    status_uri = "/api/generate/webui/status"
    progress_bar = st.progress(0, text="任务已提交...")
    
    for i in range(60):
        time.sleep(2)
        progress_bar.progress((i + 1) / 60)
        
        check_headers = get_liblib_headers(status_uri) 
        try:
            check_res = requests.get(
                domain + status_uri, 
                headers=check_headers, 
                params={"generateUuid": generate_uuid}
            )
            res_data = check_res.json()
            status = res_data.get('data', {}).get('generateStatus')
            
            if status == 1:
                progress_bar.progress(1.0, text="渲染完成！")
                return res_data['data']['images'][0]['imageUrl'], None
            elif status == 2: 
                return None, f"生成失败"
        except:
            pass
            
    return None, "超时"

# ==========================================
# 5. 界面
# ==========================================
st.title("🛋️ AI 家具设计 (调试模式)")

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
        st.error("❌ 生成失败！请查看下方调试信息：")
        
        # --- 🐞 核心：展示调试信息 ---
        if isinstance(error, dict): # 如果返回的是 debug 字典
            with st.expander("🐞 点击查看 API 报错详情 (截图发给我)", expanded=True):
                st.write(f"**Status Code:** {error['Status']}")
                st.write(f"**Request URL:** `{error['URL']}`")
                st.write("**Response Body (服务器返回的内容):**")
                st.code(error['Response Text'], language="json")
                st.write("**Payload Preview:**")
                st.code(error['Payload'])
        else:
            st.write(error)
        
        st.stop()
    
    st.success("✅ 成功！")
    generated_response = requests.get(img_url)
    generated_img = Image.open(io.BytesIO(generated_response.content))
    final_img = process_multiply(generated_img, cleaned_img)
    st.image(final_img, caption="最终效果", use_column_width=True)
