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

# ==========================================
# 1. 基础配置
# ==========================================
st.set_page_config(page_title="AI 家具设计全自动生成器", page_icon="🛋️", layout="wide")

try:
    ACCESS_KEY = st.secrets["LIBLIB_ACCESS_KEY"]
    SECRET_KEY = st.secrets["LIBLIB_SECRET_KEY"]
    MODEL_UUID = st.secrets["LIBLIB_TEMPLATE_UUID"]
except Exception as e:
    st.error("❌ 配置缺失！请在 .streamlit/secrets.toml 中配置 Key 和 UUID")
    st.stop()

# ==========================================
# 2. 签名生成函数
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
# 3. 图像处理函数
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
# 4. API 调用逻辑 (UUID 专用接口)
# ==========================================
def call_liblib_api(prompt, control_image):
    domain = "https://api.liblib.art"
    
    # --- 🚨 核心修正：使用 UUID 专用接口路径 ---
    submit_uri = "/api/www/v1/generation/uuid/generate"
    
    base64_img = image_to_base64(control_image)
    
    # --- 构造参数 (V1 UUID 接口格式) ---
    payload = {
        "uuid": MODEL_UUID,  # 注意：这个接口参数名叫 uuid，不是 template_uuid
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
    
    try:
        full_url = domain + submit_uri
        print(f"Requesting: {full_url}")
        
        response = requests.post(full_url, headers=headers, json=payload)
        
        # 打印调试信息
        print(f"Code: {response.status_code}")
        print(f"Body: {response.text}")
        
        if response.status_code != 200:
            return None, f"提交失败 ({response.status_code}): {response.text}"
            
        data = response.json()
        if data.get('code') != 0:
            return None, f"API 业务报错: {data.get('msg')}"
            
        generate_uuid = data['data']['generate_uuid']
        
    except Exception as e:
        return None, f"请求异常: {e}"
    
    # --- 轮询结果 ---
    status_uri = "/api/www/v1/generation/status"
    
    progress_bar = st.progress(0, text="☁️ 任务已提交，等待 GPU 响应...")
    
    for i in range(60):
        time.sleep(2)
        progress_bar.progress((i + 1) / 60, text=f"☁️ AI 渲染中... ({i*2}s)")
        
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
        except Exception as check_e:
            print(f"轮询出错: {check_e}")
            pass
            
    return None, "等待超时 (60秒未完成)"

# ==========================================
# 5. 界面逻辑
# ==========================================
st.title("🛋️ AI 家具设计 (UUID专用版)")

col_input, col_process = st.columns([1, 2])

with col_input:
    uploaded_file = st.file_uploader("上传草图", type=["jpg", "png", "jpeg"])
    prompt_text = st.text_area("设计描述", "modern minimalist wardrobe, walnut wood texture, soft lighting, 8k", height=120)
    run_btn = st.button("🚀 开始生成", type="primary")

if run_btn and uploaded_file:
    with col_process:
        with st.status("全自动处理中...", expanded=True) as status:
            st.write("🧹 清洗草图...")
            uploaded_file.seek(0)
            cleaned_img = process_clean_sketch(uploaded_file)
            st.image(cleaned_img, width=200, caption="清洗后")
            
            st.write("☁️ 调用 LiblibAI...")
            img_url, error = call_liblib_api(prompt_text, cleaned_img)
            
            if error:
                status.update(label="生成失败", state="error")
                st.error(error)
                st.stop()
            
            st.write("📥 下载渲染图...")
            generated_response = requests.get(img_url)
            generated_img = Image.open(io.BytesIO(generated_response.content))
            
            st.write("🎨 正片叠底合成...")
            final_img = process_multiply(generated_img, cleaned_img)
            
            status.update(label="✅ 完成！", state="complete")

        st.image(final_img, caption="最终效果", use_column_width=True)
        
        buf = io.BytesIO()
        final_img.save(buf, format="JPEG", quality=95)
        st.download_button("⬇️ 下载原图", buf.getvalue(), "design.jpg", "image/jpeg", type="primary")
