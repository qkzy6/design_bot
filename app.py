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
# 2. 核心：签名生成函数 (HMAC-SHA1)
# ==========================================
def get_liblib_headers(uri):
    timestamp = str(int(time.time() * 1000))
    signature_nonce = str(uuid.uuid4())
    
    # 签名原串拼接
    content = '&'.join((uri, timestamp, signature_nonce))
    
    digest = hmac.new(
        SECRET_KEY.encode('utf-8'), 
        content.encode('utf-8'), 
        hashlib.sha1
    ).digest()
    
    sign = base64.urlsafe_b64encode(digest).rstrip(b'=').decode('utf-8')
    
    headers = {
        "Content-Type": "application/json",
        "AccessKey": ACCESS_KEY,
        "Timestamp": timestamp,
        "SignatureNonce": signature_nonce,
        "Signature": sign
    }
    return headers

# ==========================================
# 3. 图像处理函数
# ==========================================
def process_clean_sketch(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    # 参数优化：C=5 保留更多线条细节
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
# 4. API 调用逻辑 (WebUI 接口 + API 域名)
# ==========================================
def call_liblib_api(prompt, control_image):
    # --- 🚨 核心修正 1: 域名用 api ---
    domain = "https://api.liblib.art"
    
    # --- 🚨 核心修正 2: 路径用 webui ---
    # 这是你截图里显示的路径，必须配上 api 域名
    submit_uri = "/api/generate/webui/text2img"
    
    base64_img = image_to_base64(control_image)
    
    # --- 🚨 核心修正 3: 参数结构改回 WebUI 格式 (驼峰命名) ---
    # WebUI 接口通常要求 templateUuid，而不是 template_uuid
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
                    # 注意：如果是SDXL模型，这里可能需要改成 "diffusers_xl_canny_full"
                    # 如果报错说模型不匹配，请尝试改这个字段
                    "model": "control_v11p_sd15_canny", 
                    "image": base64_img,
                    "weight": 0.8
                }
            ]
        }
    }
    
    # 生成签名
    headers = get_liblib_headers(submit_uri)
    
    try:
        full_url = domain + submit_uri
        print(f"正在请求: {full_url}") 
        
        response = requests.post(full_url, headers=headers, json=payload)
        
        print(f"状态码: {response.status_code}")
        print(f"返回: {response.text}")
        
        if response.status_code != 200:
            return None, f"提交失败 ({response.status_code}): {response.text}"
            
        data = response.json()
        if data.get('code') != 0:
            return None, f"API 业务报错: {data.get('msg')}"
            
        # WebUI 接口返回的字段通常是 generateUuid
        generate_uuid = data['data']['generateUuid']
        
    except Exception as e:
        return None, f"请求异常: {e}"
    
    # --- 2. 轮询结果 ---
    # WebUI 查询接口
    status_uri = "/api/generate/webui/status"
    
    progress_bar = st.progress(0, text="☁️ 任务已提交，等待 GPU 响应...")
    
    for i in range(60):
        time.sleep(2)
        progress_bar.progress((i + 1) / 60, text=f"☁️ AI 渲染中... ({i*2}s)")
        
        check_headers = get_liblib_headers(status_uri) 
        
        try:
            # WebUI 接口通常把 uuid 放在 params 里
            check_res = requests.get(
                domain + status_uri, 
                headers=check_headers, 
                params={"generateUuid": generate_uuid}
            )
            res_data = check_res.json()
            
            # 1=成功 (WebUI 状态码)
            status = res_data.get('data', {}).get('generateStatus')
            
            if status == 1:
                progress_bar.progress(1.0, text="渲染完成！")
                return res_data['data']['images'][0]['imageUrl'], None
            elif status == 2: # 2=失败
                return None, f"服务端生成失败"
        except Exception as check_e:
            print(f"轮询出错: {check_e}")
            pass
            
    return None, "等待超时 (60秒未完成)"
# ==========================================
# 5. 界面逻辑
# ==========================================
st.title("🛋️ AI 家具设计工作流")

col_input, col_process = st.columns([1, 2])

with col_input:
    uploaded_file = st.file_uploader("上传草图", type=["jpg", "png", "jpeg"])
    prompt_text = st.text_area("设计描述", "modern minimalist wardrobe, walnut texture, soft lighting, 8k", height=100)
    run_btn = st.button("🚀 开始生成", type="primary", use_container_width=True)

if run_btn and uploaded_file:
    with col_process:
        with st.status("全自动处理中...", expanded=True) as status:
            
            st.write("🧹 清洗草图...")
            uploaded_file.seek(0)
            cleaned_img = process_clean_sketch(uploaded_file)
            st.image(cleaned_img, width=200, caption="清洗后")
            
            st.write("☁️ 调用 LiblibAI (标准接口)...")
            img_url, error = call_liblib_api(prompt_text, cleaned_img)
            
            if error:
                status.update(label="生成失败", state="error")
                st.error(error)
                # 打印出完整的错误信息以便调试
                print(error)
                st.stop()
            
            st.write("📥 下载渲染图...")
            generated_response = requests.get(img_url)
            generated_img = Image.open(io.BytesIO(generated_response.content))
            
            st.write("🎨 正片叠底合成...")
            final_img = process_multiply(generated_img, cleaned_img)
            
            status.update(label="✅ 完成！", state="complete")

        st.image(final_img, caption="最终成品图", use_column_width=True)
        
        buf = io.BytesIO()
        final_img.save(buf, format="JPEG", quality=95)
        st.download_button("⬇️ 下载原图", buf.getvalue(), "design.jpg", "image/jpeg", type="primary")



