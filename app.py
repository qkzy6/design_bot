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

# 从 secrets.toml 读取配置
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
    """
    根据 LiblibAI 文档生成签名
    uri: 接口路径，例如 '/api/generate/webui/text2img'
    """
    timestamp = str(int(time.time() * 1000))
    signature_nonce = str(uuid.uuid4())
    
    # 1. 拼接签名原串
    content = '&'.join((uri, timestamp, signature_nonce))
    
    # 2. HMAC-SHA1 加密
    digest = hmac.new(
        SECRET_KEY.encode('utf-8'), 
        content.encode('utf-8'), 
        hashlib.sha1
    ).digest()
    
    # 3. Base64 编码
    sign = base64.urlsafe_b64encode(digest).rstrip(b'=').decode('utf-8')
    
    # 4. 构造请求头
    headers = {
        "Content-Type": "application/json",
        "AccessKey": ACCESS_KEY,
        "Timestamp": timestamp,
        "SignatureNonce": signature_nonce,
        "Signature": sign
    }
    return headers

# ==========================================
# 3. 图像处理函数 (本地 CPU)
# ==========================================
def process_clean_sketch(uploaded_file):
    """清洗草图：去底色，变黑白线稿"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # --- ✨ 关键修改：参数优化 ✨ ---
    # blockSize=31 (保持不变)
    # C=5 (之前是15。改小这个数值，可以保留更多浅色线条，防止变白纸)
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    return Image.fromarray(binary)

def process_multiply(render_img, sketch_img):
    """正片叠底：把线稿叠在渲染图上"""
    # 统一尺寸
    if render_img.size != sketch_img.size:
        sketch_img = sketch_img.resize(render_img.size)
    
    # 转换模式
    render_img = render_img.convert("RGB")
    sketch_img = sketch_img.convert("RGB")
    
    # 执行合成
    return ImageChops.multiply(render_img, sketch_img)

def image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ==========================================
# 4. API 调用逻辑 (域名修正版)
# ==========================================
def call_liblib_api(prompt, control_image):
    # --- 🚨 核心修正：域名必须是 api 开头 ---
    domain = "https://api.liblib.art"
    
    # 接口路径 (基于你的文档截图)
    submit_uri = "/api/generate/webui/text2img"
    
    # 准备图片
    base64_img = image_to_base64(control_image)
    
    # 构造 Payload
    payload = {
        "templateUuid": MODEL_UUID,
        "generateParams": {
            "prompt": prompt + ", interior design, furniture, best quality, 8k, masterpiece",
            "steps": 25,
            "width": 1024, 
            "height": 1024,
            "imgCount": 1,
            "controlNet": [
                {
                    "enabled": True,
                    "module": "canny", 
                    "model": "control_v11p_sd15_canny", 
                    "image": base64_img,
                    "weight": 0.8
                }
            ]
        }
    }
    
    # --- 1. 提交任务 ---
    # 获取签名 (注意：签名只针对 uri，不包含域名)
    headers = get_liblib_headers(submit_uri)
    
    try:
        # 拼接完整 URL
        full_url = domain + submit_uri
        print(f"正在请求: {full_url}") # 调试打印
        
        response = requests.post(full_url, headers=headers, json=payload)
        
        # 打印返回内容，如果报错方便排查
        print(f"提交状态: {response.status_code}")
        print(f"提交返回: {response.text}")
        
        if response.status_code != 200:
            return None, f"提交失败 ({response.status_code}): {response.text[:200]}..." # 只显示前200字符防止刷屏
            
        data = response.json()
        if data.get('code') != 0:
            return None, f"API 业务报错: {data.get('msg')}"
            
        generate_uuid = data['data']['generateUuid']
        
    except Exception as e:
        return None, f"请求异常: {e}"
    
    # --- 2. 轮询结果 ---
    status_uri = "/api/generate/webui/status"
    
    progress_bar = st.progress(0, text="☁️ 任务已提交，等待 GPU 响应...")
    
    for i in range(60): # 轮询 60次
        time.sleep(2)
        progress_bar.progress((i + 1) / 60, text=f"☁️ AI 渲染中... ({i*2}s)")
        
        # 查询也要签名
        check_headers = get_liblib_headers(status_uri) 
        
        try:
            check_res = requests.get(
                domain + status_uri, 
                headers=check_headers, 
                params={"generateUuid": generate_uuid}
            )
            res_data = check_res.json()
            
            # 1=成功, 2=失败/超时 (依据文档)
            # 注意：有些接口返回字段可能是 generateStatus
            status = res_data.get('data', {}).get('generateStatus')
            
            if status == 1:
                progress_bar.progress(1.0, text="渲染完成！")
                return res_data['data']['images'][0]['imageUrl'], None
            elif status == 2 or status == -1: # 失败状态
                return None, f"服务端生成失败: {res_data}"
        except Exception as check_e:
            print(f"轮询出错: {check_e}")
            pass
            
    return None, "等待超时 (60秒未完成)"

# ==========================================
# 5. 网页界面
# ==========================================
st.title("🛋️ AI 家具设计工作流")

col_input, col_process = st.columns([1, 2])

with col_input:
    st.info("💡 请上传手绘草图，尽量清晰")
    uploaded_file = st.file_uploader("上传图片", type=["jpg", "png", "jpeg"])
    
    prompt_text = st.text_area(
        "设计描述", 
        "现代极简风格衣柜，胡桃木纹理，高级灰色调，柔和室内光线，照片级真实感，8k分辨率，大师级室内设计", 
        height=120
    )
    
    run_btn = st.button("🚀 开始生成", type="primary", use_container_width=True)

if run_btn and uploaded_file:
    with col_process:
        with st.status("全自动处理中...", expanded=True) as status:
            
            # 1. 清洗
            st.write("1️⃣ 正在清洗草图 (去底色)...")
            uploaded_file.seek(0)
            cleaned_img = process_clean_sketch(uploaded_file)
            st.image(cleaned_img, width=200, caption="清洗后的线稿")
            
            # 2. 调用 AI
            st.write("2️⃣ 正在调用 LiblibAI 进行渲染...")
            img_url, error = call_liblib_api(prompt_text, cleaned_img)
            
            if error:
                status.update(label="生成失败", state="error")
                st.error(error)
                st.stop()
            
            # 3. 下载
            st.write("3️⃣ 下载渲染结果...")
            generated_response = requests.get(img_url)
            generated_img = Image.open(io.BytesIO(generated_response.content))
            
            # 4. 合成
            st.write("4️⃣ 正片叠底回填尺寸...")
            final_img = process_multiply(generated_img, cleaned_img)
            
            status.update(label="✅ 全部完成！", state="complete")

        # 展示结果
        st.image(final_img, caption="最终成品图", use_column_width=True)
        st.caption("💡 手机用户：长按图片可保存到相册")
        
        # 下载按钮
        buf = io.BytesIO()
        final_img.save(buf, format="JPEG", quality=95)
        st.download_button(
            "⬇️ 下载高清原图", 
            buf.getvalue(), 
            "design_final.jpg", 
            "image/jpeg",
            type="primary"
        )

