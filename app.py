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
st.set_page_config(page_title="AI 家具设计生成器", page_icon="🛋️", layout="wide")

try:
    ACCESS_KEY = st.secrets["LIBLIB_ACCESS_KEY"]
    SECRET_KEY = st.secrets["LIBLIB_SECRET_KEY"]
    MODEL_UUID = st.secrets["LIBLIB_TEMPLATE_UUID"]
except Exception as e:
    st.error("❌ 系统配置缺失，请联系管理员配置 API 密钥。")
    st.stop()

# ==========================================
# 2. 签名与鉴权
# ==========================================
def get_liblib_headers(full_url):
    # 自动解析 path 用于签名
    parsed = urllib.parse.urlparse(full_url)
    uri = parsed.path
    
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
# 4. API 调用 (智能适配版)
# ==========================================
def call_liblib_api(prompt, control_image, submit_url):
    base64_img = image_to_base64(control_image)
    
    # --- 智能参数适配 ---
    # 如果 URL 里包含 "webui"，说明是 WebUI 接口，使用驼峰命名
    is_webui = "webui" in submit_url.lower()
    
    if is_webui:
        # WebUI 格式 (templateUuid)
        payload = {
            "templateUuid": MODEL_UUID,
            "generateParams": {
                "prompt": prompt + ", interior design, furniture, best quality, 8k",
                "steps": 25,
                "width": 1024,
                "height": 1024,
                "imgCount": 1,
                "controlNet": [{
                    "enabled": True,
                    "module": "canny",
                    "model": "diffusers_xl_canny_full", # SDXL专用
                    "image": base64_img,
                    "weight": 0.8
                }]
            }
        }
    else:
        # 标准 V1 格式 (template_uuid)
        payload = {
            "template_uuid": MODEL_UUID,
            "generate_params": {
                "prompt": prompt + ", interior design, furniture, best quality, 8k",
                "steps": 25,
                "width": 1024,
                "height": 1024,
                "img_count": 1,
                "controlnet": {
                    "units": [{
                        "type": "canny",
                        "weight": 0.8,
                        "image_base64": base64_img
                    }]
                }
            }
        }
    
    headers = get_liblib_headers(submit_url)
    
    try:
        print(f"请求 URL: {submit_url}")
        print(f"模式: {'WebUI' if is_webui else 'Standard V1'}")
        
        response = requests.post(submit_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            return None, f"提交失败 ({response.status_code}): {response.text}"
            
        data = response.json()
        if data.get('code') != 0:
            return None, f"API 报错: {data.get('msg')}"
            
        # 兼容两种返回字段
        generate_uuid = data['data'].get('generateUuid') or data['data'].get('generate_uuid')
        
    except Exception as e:
        return None, f"请求异常: {e}"
    
    # --- 轮询结果 ---
    # 自动推导查询 URL
    parsed = urllib.parse.urlparse(submit_url)
    domain = f"{parsed.scheme}://{parsed.netloc}"
    
    if is_webui:
        status_url = f"{domain}/api/generate/webui/status"
    else:
        status_url = f"{domain}/api/www/v1/generation/status"
        
    progress_bar = st.progress(0, text="☁️ 任务已提交...")
    
    for i in range(60):
        time.sleep(2)
        progress_bar.progress((i + 1) / 60, text=f"☁️ AI 渲染中... ({i*2}s)")
        
        check_headers = get_liblib_headers(status_url) 
        try:
            check_res = requests.get(
                status_url, 
                headers=check_headers, 
                params={"generateUuid": generate_uuid} if is_webui else {"generate_uuid": generate_uuid}
            )
            res_data = check_res.json()
            
            # 兼容两种状态字段
            status = res_data.get('data', {}).get('generateStatus') # WebUI
            if status is None:
                status = res_data.get('data', {}).get('status') # V1
            
            if status == 1:
                progress_bar.progress(1.0, text="渲染完成！")
                return res_data['data']['images'][0]['imageUrl'] if is_webui else res_data['data']['images'][0]['image_url'], None
            elif status == 2 or status == -1:
                return None, "服务端生成失败"
        except:
            pass
            
    return None, "等待超时"

# ==========================================
# 5. 界面逻辑
# ==========================================
st.title("🛋️ AI 家具设计生成器")

# --- 🔧 隐藏的管理员设置 ---
with st.sidebar:
    with st.expander("🔧 高级接口设置 (管理员)", expanded=False):
        # 默认值设为我们猜测最可能的地址
        # 如果 404，请在这里手动修改为文档里的地址！
        custom_api_url = st.text_input(
            "API URL", 
            value="https://api.liblib.art/api/www/v1/generation/generate",
            help="如果报错 404，请尝试修改此处地址"
        )

col_input, col_process = st.columns([1, 1.5])

with col_input:
    st.markdown("### 1. 上传草图")
    uploaded_file = st.file_uploader("请上传白底黑线的家具手绘图 (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    st.markdown("### 2. 设计要求")
    prompt_text = st.text_area(
        "描述", 
        "modern minimalist wardrobe, walnut wood texture, soft lighting, 8k resolution, masterpiece", 
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
            
            st.write("☁️ 正在调用云端 GPU...")
            # 传入管理员设置的 URL
            img_url, error = call_liblib_api(prompt_text, cleaned_img, custom_api_url)
            
            if error:
                status.update(label="生成失败", state="error")
                st.error(error)
                if "404" in str(error):
                     st.warning("👉 提示：请点击左侧侧边栏的 **'🔧 高级接口设置'**，尝试修改 API URL。")
                st.stop()
            
            st.write("📥 正在合成尺寸标注...")
            generated_response = requests.get(img_url)
            generated_img = Image.open(io.BytesIO(generated_response.content))
            final_img = process_multiply(generated_img, cleaned_img)
            
            status.update(label="✅ 设计完成！", state="complete")

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
