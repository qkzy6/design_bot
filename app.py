import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops
import io
import os
import requests
import dashscope
from dashscope import ImageSynthesis
import sys
import json
import time 

# ==========================================
# 1. 基础配置
# ==========================================
st.set_page_config(page_title="AI 家具设计 (终极稳定版)", page_icon="🛋️", layout="wide")

try:
    api_key = st.secrets["DASHSCOPE_API_KEY"]
    dashscope.api_key = api_key
except Exception as e:
    st.error("❌ 未找到密钥！请在 .streamlit/secrets.toml 中配置 DASHSCOPE_API_KEY")
    st.stop()

# ==========================================
# 2. 图像处理函数 (本地 CPU)
# ==========================================
def process_clean_sketch(uploaded_file):
    """清洗草图：去底色，提取黑白线条"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    return Image.fromarray(binary)

def process_multiply(render_img, sketch_img):
    """正片叠底：把线稿叠回去"""
    if render_img.size != sketch_img.size:
        sketch_img = sketch_img.resize(render_img.size)
    render_img = render_img.convert("RGB")
    sketch_img = sketch_img.convert("RGB")
    return ImageChops.multiply(render_img, sketch_img)

# ==========================================
# 3. 核心：文件操作 (两步法)
# ==========================================

def get_file_url_from_id(api_key, file_id):
    """
    第二步：根据 file_id 查询文件的最终 OSS URL，直到文件状态变为 'success'。
    """
    status_url = f"https://dashscope.aliyuncs.com/api/v1/files/{file_id}"
    headers = {'Authorization': f'Bearer {api_key}'}
    
    # 🚨 修正点：延长循环时间到 20 次 (共 40 秒)
    for i in range(20): 
        time.sleep(2) # 每次查询间隔 2 秒
        
        response = requests.get(status_url, headers=headers, timeout=20)
        
        if response.status_code == 200:
            data = response.json()
            
            # 1. 检查最终 URL (如果存在，直接成功)
            if data.get('url'): 
                return data['url'], None 
            
            current_status = data.get('status')
            
            # 2. 检查失败状态
            if current_status == 'FAILED': 
                return None, f"文件处理失败。服务器信息: {response.text}"
            
            # 3. 如果状态是 RUNNING/PENDING/空，继续等待
            if current_status in ['RUNNING', 'PENDING', 'PROCESSING', None]:
                continue
            
            # 4. 如果状态是非预期状态，且已经等待了一段时间，可能是服务器返回格式错误
            if i > 5 and current_status not in ['SUCCESS', 'RUNNING', 'PENDING', 'PROCESSING']:
                return None, f"文件处理异常。服务器信息: {response.text}"
        
        else:
            return None, f"文件状态查询 HTTP 错误 ({response.status_code}): {response.text}"
    
    return None, "文件处理超时，请重试。"


def upload_file_to_aliyun(api_key, file_path):
    """
    第一步：上传文件并获取 file_id。
    """
    upload_url = "https://dashscope.aliyuncs.com/api/v1/files"
    headers = {'Authorization': f'Bearer {api_key}'}
    
    try:
        with open(file_path, 'rb') as file_data:
            files = {
                'file': (os.path.basename(file_path), file_data, 'image/png')
            }
            data = {'purpose': 'image-generation'} 
            
            response = requests.post(
                upload_url, headers=headers, data=data, files=files, timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                uploaded_files = data.get('data', {}).get('uploaded_files')
                
                # 提取 file_id 
                if uploaded_files and uploaded_files[0].get('file_id'):
                    file_id = uploaded_files[0]['file_id']
                    
                    # 立即调用第二步：查询 URL
                    return get_file_url_from_id(api_key, file_id)
                else:
                    return None, f"上传成功但未找到 file_id。"
            else:
                return None, f"HTTP 错误 ({response.status_code}): {response.text}"

    except Exception as e:
        return None, f"网络请求异常: {str(e)}"

# ==========================================
# 4. 阿里云 API 调用逻辑
# ==========================================
def call_aliyun_wanx(prompt, control_image):
    # 1. 保存临时文件
    temp_filename = "temp_sketch.png"
    control_image.save(temp_filename)
    
    try:
        # --- 核心步骤：上传文件获取 URL ---
        with st.spinner("☁️ 正在上传草图到阿里云 OSS..."):
            sketch_cloud_url, upload_error = upload_file_to_aliyun(api_key, temp_filename)
            
        if upload_error:
            return None, upload_error
            
        # 2. 发起生成请求
        rsp = ImageSynthesis.call(
            model="wanx-sketch-to-image-v1", 
            input={
                'image': sketch_cloud_url,
                'prompt': prompt + ", 室内设计, 家具, 8k分辨率, 杰作, 高清材质, 柔和光线"
            },
            n=1,
            size='1024*1024'
        )
        
        if rsp.status_code == 200:
            return rsp.output.results[0].url, None
        else:
            return None, f"阿里云生成报错: {rsp.code} - {rsp.message}"
            
    except Exception as e:
        return None, f"SDK 异常 (生成阶段): {str(e)}"

# ==========================================
# 5. 界面逻辑
# ==========================================
st.title("🛋️ AI 家具设计 (阿里云终极稳定版)")

col_input, col_process = st.columns([1, 1.5])

with col_input:
    uploaded_file = st.file_uploader("上传草图", type=["jpg", "png", "jpeg"])
    prompt_text = st.text_area("设计描述", "现代极简风格衣柜，胡桃木纹理，高级灰色调，柔和室内光线，照片级真实感", height=120)
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
