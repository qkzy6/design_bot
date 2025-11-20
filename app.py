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
    
    # 拼接签名原串: uri & timestamp & nonce
    content = '&'.join((uri, timestamp, signature_nonce))
    
    # HMAC-SHA1 加密
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
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
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
# 4. API 调用逻辑 (已修正 URI)
# ==========================================
def call_liblib_api(prompt, control_image):
    domain = "https://api.liblib.art"
    
    # ✅ 修正点1：使用截图中的正确接口地址
    submit_uri = "/api/generate/webui/text2img"
    
    # 准备 ControlNet 图片
    base64_img = image_to_base64(control_image)
    
    # ✅ 修正点2：构造符合截图结构的 Payload
    payload = {
        "templateUuid": MODEL_UUID,
        "generateParams": {
            "prompt": prompt + ", interior design, furniture, best quality, 8k",
            "steps": 25,
            "width": 1024, # 注意：Juggernaut XL 建议用 1024x1024
            "height": 1024,
            "imgCount": 1,
            "controlNet": [  # 注意这里是列表 list
                {
                    "enabled": True,
                    "module": "canny",  # 预处理器
                    "model": "control_v11p_sd15_canny", # ⚠️核心：这里可能需要根据你的底模修改，如果是XL模型，这里要填XL的controlnet模型名
                    "image": base64_img, # 注意参数名是 image 还是 image_base64，通常 WebUI 接口用 image
                    "weight": 0.8
                }
            ]
        }
    }
    
    # 发起请求
    headers = get_liblib_headers(submit_uri)
    
    try:
        response = requests.post(domain + submit_uri, headers=headers, json=payload)
        
        if response.status_code != 200:
            return None, f"提交失败 ({response.status_code}): {response.text}"
            
        data = response.json()
        if data.get('code') != 0:
            return None, f"API 报错: {data.get('msg')}"
            
        generate_uuid = data['data']['generateUuid'] # 注意大小写可能不同，通常是 generateUuid 或 generate_uuid
        
    except Exception as e:
        return None, f"请求异常: {e}"
    
    # --- 轮询结果 ---
    # ✅ 修正点3：对应的查询接口通常是这个
    status_uri = "/api/generate/webui/status" 
    
    progress_bar = st.progress(0, text="☁️ 任务已提交，等待 GPU 响应...")
    
    for i in range(60):
        time.sleep(2)
        progress_bar.progress((i + 1) / 60, text="☁️ AI 正在渲染...")
        
        check_headers = get_liblib_headers(status_uri) 
        
        try:
            # 注意：generateUuid 作为参数传递
            check_res = requests.get(
                domain + status_uri, 
                headers=check_headers, 
                params={"generateUuid": generate_uuid}
            )
            res_data = check_res.json()
            
            # 1=成功, -1=失败
            if res_data['data']['generateStatus'] == 1: # 注意这里字段名可能是 generateStatus
                progress_bar.progress(1.0, text="渲染完成！")
                return res_data['data']['images'][0]['imageUrl'], None
            elif res_data['data']['generateStatus'] == 2: # 2通常是失败/超时
                return None, "生成失败: " + str(res_data['data'])
        except:
            pass
            
    return None, "等待超时"

# ==========================================
# 5. 界面
# ==========================================
st.title("🛋️ AI 家具设计工作流")
st.info("当前接口模式: WebUI 自定义模版")

col_input, col_process = st.columns([1, 2])

with col_input:
    uploaded_file = st.file_uploader("上传草图", type=["jpg", "png", "jpeg"])
    prompt_text = st.text_area(
        "设计描述", 
        "现代极简风格衣柜，胡桃木纹理，高级灰色调，柔和室内光线，照片级真实感，8k分辨率，大师级室内设计", 
        height=100
    )
    run_btn = st.button("🚀 开始生成", type="primary")

if run_btn and uploaded_file:
    with col_process:
        with st.status("正在处理...", expanded=True) as status:
            st.write("🧹 清洗草图...")
            uploaded_file.seek(0)
            cleaned_img = process_clean_sketch(uploaded_file)
            st.image(cleaned_img, width=200, caption="清洗后")
            
            st.write("☁️ 调用 LiblibAI...")
            img_url, error = call_liblib_api(prompt_text, cleaned_img)
            
            if error:
                status.update(label="失败", state="error")
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
