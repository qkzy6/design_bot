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
st.set_page_config(page_title="AI 家具设计生成器", page_icon="🛋️", layout="wide")

try:
    ACCESS_KEY = st.secrets["LIBLIB_ACCESS_KEY"]
    SECRET_KEY = st.secrets["LIBLIB_SECRET_KEY"]
    MODEL_UUID = st.secrets["LIBLIB_TEMPLATE_UUID"]
except Exception as e:
    st.error("❌ 系统配置缺失，请联系管理员配置 API 密钥。")
    st.stop()

# ==========================================
# 2. 核心：签名生成函数 (HMAC-SHA1)
# ==========================================
def get_liblib_headers(uri):
    """生成 LiblibAI 鉴权签名"""
    timestamp = str(int(time.time() * 1000))
    signature_nonce = str(uuid.uuid4())
    
    # 签名原串拼接
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
# 3. 图像处理函数 (本地 CPU)
# ==========================================
def process_clean_sketch(uploaded_file):
    """清洗草图：去除背景阴影，提取黑白线条"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # 自适应二值化 (参数 C=5 针对手绘优化)
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    return Image.fromarray(binary)

def process_multiply(render_img, sketch_img):
    """正片叠底：将线稿叠加回渲染图"""
    # 统一尺寸
    if render_img.size != sketch_img.size:
        sketch_img = sketch_img.resize(render_img.size)
    
    render_img = render_img.convert("RGB")
    sketch_img = sketch_img.convert("RGB")
    
    # 像素混合
    return ImageChops.multiply(render_img, sketch_img)

def image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ==========================================
# 4. API 调用逻辑 (严格匹配文档截图)
# ==========================================
def call_liblib_api(prompt, control_image):
    # --- 硬编码配置 (用户不可见) ---
    domain = "https://api.liblib.art"
    submit_uri = "/api/generate/webui/text2img" # 基于截图确认的路径
    status_uri = "/api/generate/webui/status"   # 配套的查询路径
    
    base64_img = image_to_base64(control_image)
    
    # --- 构造参数 (WebUI 驼峰命名格式) ---
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
                    # ⚠️ 如果使用 SDXL 模型，请保留这个名字
                    # ⚠️ 如果使用 SD1.5 模型，请改为 "control_v11p_sd15_canny"
                    "model": "diffusers_xl_canny_full", 
                    "image": base64_img,
                    "weight": 0.8
                }
            ]
        }
    }
    
    # --- 1. 提交生成任务 ---
    headers = get_liblib_headers(submit_uri)
    
    try:
        response = requests.post(domain + submit_uri, headers=headers, json=payload)
        
        if response.status_code != 200:
            return None, f"提交失败 (Code {response.status_code}): {response.text}"
            
        data = response.json()
        if data.get('code') != 0:
            return None, f"API 拒绝请求: {data.get('msg')}"
            
        generate_uuid = data['data']['generateUuid']
        
    except Exception as e:
        return None, f"网络请求异常: {e}"
    
    # --- 2. 轮询任务状态 ---
    progress_bar = st.progress(0, text="☁️ 正在云端渲染...")
    
    for i in range(60): # 等待约 2 分钟
        time.sleep(2)
        progress_bar.progress((i + 1) / 60, text="☁️ AI 正在绘制材质与光影...")
        
        # 查询接口也需要签名
        check_headers = get_liblib_headers(status_uri) 
        
        try:
            # WebUI 接口通常把 uuid 放在 URL 参数里
            check_res = requests.get(
                domain + status_uri, 
                headers=check_headers, 
                params={"generateUuid": generate_uuid}
            )
            res_data = check_res.json()
            
            # 状态码说明: 1=成功, 2=失败, 0=进行中
            status = res_data.get('data', {}).get('generateStatus')
            
            if status == 1:
                progress_bar.progress(1.0, text="渲染完成！")
                return res_data['data']['images'][0]['imageUrl'], None
            elif status == 2:
                return None, "服务端渲染失败，请检查模型是否兼容 ControlNet"
            # status == 0 继续等待
            
        except Exception:
            pass # 网络抖动则重试
            
    return None, "渲染超时，请稍后重试"

# ==========================================
# 5. 界面布局
# ==========================================
st.title("🛋️ AI 家具设计生成器")

col_input, col_process = st.columns([1, 1.5])

with col_input:
    st.markdown("### 1. 上传草图")
    uploaded_file = st.file_uploader("请上传白底黑线的家具手绘图 (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    st.markdown("### 2. 设计要求")
    prompt_text = st.text_area(
        "描述你想要的材质、颜色和光影", 
        "modern minimalist wardrobe, walnut wood texture, soft lighting, 8k resolution, masterpiece", 
        height=120
    )
    
    st.write("") # 占位
    run_btn = st.button("🚀 开始生成设计图", type="primary", use_container_width=True)

if run_btn and uploaded_file:
    with col_process:
        st.markdown("### 3. 生成结果")
        
        with st.status("AI 正在工作中...", expanded=True) as status:
            
            st.write("🧹 正在清洗草图噪点...")
            uploaded_file.seek(0)
            cleaned_img = process_clean_sketch(uploaded_file)
            
            st.write("🎨 正在进行 AI 材质渲染...")
            img_url, error = call_liblib_api(prompt_text, cleaned_img)
            
            if error:
                status.update(label="生成失败", state="error")
                st.error(error)
                st.stop()
            
            st.write("📥 正在下载并合成尺寸标注...")
            generated_response = requests.get(img_url)
            generated_img = Image.open(io.BytesIO(generated_response.content))
            final_img = process_multiply(generated_img, cleaned_img)
            
            status.update(label="✅ 设计完成！", state="complete")

        # 展示最终结果
        st.image(final_img, caption="最终效果图", use_column_width=True)
        st.caption("💡 提示：长按图片可保存到相册")
        
        # 下载按钮
        buf = io.BytesIO()
        final_img.save(buf, format="JPEG", quality=95)
        st.download_button(
            "⬇️ 下载高清原图", 
            data=buf.getvalue(), 
            file_name="design_final.jpg", 
            mime="image/jpeg",
            type="primary"
        )
