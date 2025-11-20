import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops
import io
import os
import requests
import dashscope
from dashscope import ImageSynthesis
# ⚠️ 注意：我们不再导入 dashscope.file，避免 ModuleNotFoundError

# ==========================================
# 1. 基础配置
# ==========================================
st.set_page_config(page_title="AI 家具设计 (终极 HTTP 版)", page_icon="🛋️", layout="wide")

try:
    api_key = st.secrets["DASHSCOPE_API_KEY"]
    dashscope.api_key = api_key
except Exception as e:
    st.error("❌ 未找到密钥！请在 .streamlit/secrets.toml 中配置 DASHSCOPE_API_KEY")
    st.stop()

# ==========================================
# 2. 核心：手动 HTTP 文件上传函数 (绕过 SDK 错误)
# ==========================================
def upload_file_to_aliyun(api_key, file_path):
    """
    手动构造 HTTP 请求，将文件上传到阿里云的 /files 接口，获取 OSS URL。
    """
    upload_url = "https://dashscope.aliyuncs.com/api/v1/files"
    
    headers = {
        'Authorization': f'Bearer {api_key}'
    }
    
    # 构造 multipart/form-data 请求体
    files = {
        'file': (os.path.basename(file_path), open(file_path, 'rb'), 'image/png'),
        'purpose': (None, 'image_file_extract') # 声明文件用途
    }
    
    try:
        response = requests.post(upload_url, headers=headers, files=files, timeout=60)
        
        if response.status_code == 200 and response.json().get('status') == 'success':
            # 返回的文件对象中包含一个 URL 字段 (即 OSS 地址)
            return response.json()['url'] 
        else:
            return None
            
    except Exception as e:
        print(f"HTTP UPLOAD FAILED: {e}")
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

# ==========================================
# 4. 阿里云 API 调用 (无 SDK File 依赖)
# ==========================================
def call_aliyun_wanx(prompt, control_image):
    # 1. 保存临时文件
    temp_filename = "temp_sketch.png"
    control_image.save(temp_filename)
    
    try:
        # --- 🚨 核心修复：通过 HTTP 上传文件，绕过 SDK 依赖 ---
        with st.spinner("☁️ 正在上传草图到阿里云 OSS..."):
            sketch_url = upload_file_to_aliyun(api_key, temp_filename)
            
        if not sketch_url:
            return None, "文件上传至阿里云失败，请检查 Key 或网络。"
            
        # 2. 发起生成请求 (使用 OSS URL)
        rsp = ImageSynthesis.call(
            model="wanx-sketch-to-image-v1", 
            input={
                'image': sketch_url,
                'prompt': prompt + ", 室内设计, 家具, 8k分辨率, 杰作, 高清材质, 柔和光线"
            },
            n=1,
            size='1024*1024'
        )
        
        # 3. 处理结果
        if rsp.status_code == 200:
            return rsp.output.results[0].url, None
        else:
            return None, f"阿里云报错: {rsp.code} - {rsp.message}"
            
    except Exception as e:
        # 如果是文件上传后立刻删除失败，这里也会出错。
        return None, f"SDK 异常: {str(e)}"

# ==========================================
# 5. 界面逻辑
# ==========================================
st.title("🛋️ AI 家具设计 (阿里云最终修复版)")

col_input, col_process = st.columns([1, 1.5])

with col_input:
    uploaded_file = st.file_uploader("上传草图", type=["jpg", "png", "jpeg"])
    prompt_text = st.text_area(
        "设计描述", 
        "现代极简风格衣柜，胡桃木纹理，高级灰色调，柔和室内光线，照片级真实感", 
        height=120
    )
    run_btn = st.button("🚀 开始生成", type="primary", use_container_width=True)

if run_btn and uploaded_file:
    with col_process:
        with st.status("AI 正在工作中...", expanded=True) as status:
            
            st.write("🧹 正在清洗草图...")
            uploaded_file.seek(0)
            cleaned_img = process_clean_sketch(uploaded_file)
            st.image(cleaned_img, width=200, caption="清洗后线稿")
            
            st.write("☁️ 调用阿里云生成...")
            img_url, error = call_aliyun_wanx(prompt_text, cleaned_img)
            
            if error:
                status.update(label="生成失败", state="error")
                st.error(error)
                st.stop()
            
            st.write("📥 下载渲染图...")
            generated_response = requests.get(img_url)
            generated_img = Image.open(io.BytesIO(generated_response.content))
            
            st.write("🎨 合成标注...")
            final_img = process_multiply(generated_img, cleaned_img)
            
            status.update(label="✅ 全部完成！", state="complete")

        st.image(final_img, caption="最终效果图", use_column_width=True)
        
        buf = io.BytesIO()
        final_img.save(buf, format="JPEG", quality=95)
        st.download_button("⬇️ 下载高清原图", data=buf.getvalue(), file_name="design_final.jpg", mime="image/jpeg", type="primary")
