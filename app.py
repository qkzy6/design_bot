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
import urllib.parse

# ==========================================
# 1. 基础配置
# ==========================================
st.set_page_config(page_title="AI 家具设计 (多线路版)", page_icon="🛋️", layout="wide")

try:
    ACCESS_KEY = st.secrets["LIBLIB_ACCESS_KEY"]
    SECRET_KEY = st.secrets["LIBLIB_SECRET_KEY"]
    MODEL_UUID = st.secrets["LIBLIB_TEMPLATE_UUID"]
except Exception as e:
    st.error("❌ 配置缺失！请在 secrets.toml 中配置 Key 和 UUID")
    st.stop()

# ==========================================
# 2. 签名生成 (通用版)
# ==========================================
def get_liblib_headers(full_url):
    # 自动提取 path 进行签名 (例如 /api/generate/...)
    parsed = urllib.parse.urlparse(full_url)
    uri = parsed.path
    
    timestamp = str(int(time.time() * 1000))
    signature_nonce = str(uuid.uuid4())
    
    # 拼接签名原串
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
# 4. API 调用 (支持动态 URL)
# ==========================================
def call_liblib_api(prompt, control_image, submit_url):
    
    base64_img = image_to_base64(control_image)
    
    # 构造 WebUI 格式参数 (兼容性最好)
    payload = {
        "templateUuid": MODEL_UUID, 
        "generateParams": {
            "prompt": prompt + ", interior design, furniture, best quality, 8k, photorealistic",
            "steps": 25,
            "width": 1024,
            "height": 1024,
            "imgCount": 1,
            "controlNet": [
                {
                    "enabled": True,
                    "module": "canny", 
                    "model": "diffusers_xl_canny_full", 
                    "image": base64_img,
                    "weight": 0.8
                }
            ]
        }
    }
    
    # 签名
    headers = get_liblib_headers(submit_url)
    
    try:
        print(f"请求地址: {submit_url}")
        response = requests.post(submit_url, headers=headers, json=payload)
        
        # --- 🚨 增强报错显示 ---
        if response.status_code != 200:
            return None, f"提交失败 ({response.status_code}): {response.text}"
            
        data = response.json()
        if data.get('code') != 0:
            return None, f"API 业务报错: {data.get('msg')}"
            
        generate_uuid = data['data']['generateUuid']
        
    except Exception as e:
        return None, f"请求异常: {e}"
    
    # --- 轮询 ---
    # 自动推导查询地址 (替换 text2img 为 status)
    # 逻辑：把 .../text2img 替换为 .../status
    status_url = submit_url.replace("text2img", "status").replace("generation/image", "generation/status")
    
    # 如果自动推导不对，强制修正常见的 WebUI 查询地址
    if "webui" in submit_url and "status" not in status_url:
         status_url = submit_url.rsplit('/', 1)[0] + "/status"

    progress_bar = st.progress(0, text="☁️ 任务提交成功，等待生成...")
    
    for i in range(60):
        time.sleep(2)
        progress_bar.progress((i + 1) / 60, text=f"☁️ AI 渲染中... ({i*2}s)")
        
        check_headers = get_liblib_headers(status_url) 
        try:
            check_res = requests.get(
                status_url, 
                headers=check_headers, 
                params={"generateUuid": generate_uuid}
            )
            res_data = check_res.json()
            
            status = res_data.get('data', {}).get('generateStatus')
            
            if status == 1:
                progress_bar.progress(1.0, text="渲染完成！")
                return res_data['data']['images'][0]['imageUrl'], None
            elif status == 2:
                return None, f"服务端生成失败"
        except:
            pass
            
    return None, "等待超时"

# ==========================================
# 5. 界面逻辑
# ==========================================
st.title("🛋️ AI 家具设计 (多线路版)")

# --- 侧边栏：线路切换 ---
with st.sidebar:
    st.header("🔌 接口线路切换")
    st.info("如果你遇到 404 错误，请尝试切换下面的线路，直到成功为止。")
    
    api_option = st.radio(
        "选择 API 地址:",
        (
            "线路 1: WebUI 标准 (api.liblib.art)",
            "线路 2: WebUI 备用 (无 /api 前缀)",
            "线路 3: V1 兼容模式"
        )
    )
    
    if api_option == "线路 1: WebUI 标准 (api.liblib.art)":
        submit_url = "https://api.liblib.art/api/generate/webui/text2img"
    elif api_option == "线路 2: WebUI 备用 (无 /api 前缀)":
        submit_url = "https://api.liblib.art/generate/webui/text2img"
    else:
        submit_url = "https://api.liblib.art/api/www/v1/generation/image"
        
    st.code(submit_url, language="text")
    st.caption("当前使用的请求地址👆")

col_input, col_process = st.columns([1, 2])

with col_input:
    uploaded_file = st.file_uploader("上传草图", type=["jpg", "png", "jpeg"])
    prompt_text = st.text_area("设计描述", "modern wardrobe, walnut wood texture, 8k", height=100)
    run_btn = st.button("🚀 开始生成", type="primary")

if run_btn and uploaded_file:
    with col_process:
        with st.status("运行中...", expanded=True) as status:
            st.write("🧹 清洗草图...")
            uploaded_file.seek(0)
            cleaned_img = process_clean_sketch(uploaded_file)
            st.image(cleaned_img, width=200)
            
            st.write(f"☁️ 调用 API ({api_option})...")
            img_url, error = call_liblib_api(prompt_text, cleaned_img, submit_url)
            
            if error:
                status.update(label="失败", state="error")
                st.error(error)
                # 如果失败，打印出返回的 HTML/JSON 详情
                if "404" in str(error):
                    st.warning("👉 404 意味着地址错了。请在左侧尝试切换到其他线路！")
                st.stop()
            
            st.write("📥 下载与合成...")
            generated_response = requests.get(img_url)
            generated_img = Image.open(io.BytesIO(generated_response.content))
            final_img = process_multiply(generated_img, cleaned_img)
            
            status.update(label="✅ 完成！", state="complete")

        st.image(final_img, caption="最终效果", use_column_width=True)
