import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops
import io
import os
import requests
import dashscope
from dashscope import ImageSynthesis

# ==========================================
# 1. 基础配置
# ==========================================
st.set_page_config(page_title="AI 家具设计 (诊断版)", page_icon="🛋️", layout="wide")

try:
    api_key = st.secrets["DASHSCOPE_API_KEY"]
    dashscope.api_key = api_key
except Exception as e:
    st.error("❌ 未找到密钥！请在 .streamlit/secrets.toml 中配置 DASHSCOPE_API_KEY")
    st.stop()

# ==========================================
# 2. 核心：手动 HTTP 文件上传函数 (返回详细错误)
# ==========================================
def upload_file_to_aliyun(api_key, file_path):
    """
    手动构造 HTTP 请求，将文件上传到阿里云的 /files 接口，获取 OSS URL。
    """
    upload_url = "https://dashscope.aliyuncs.com/api/v1/files"
    
    headers = {
        'Authorization': f'Bearer {api_key}'
    }
    
    try:
        with open(file_path, 'rb') as file_data:
            files = {
                'file': (os.path.basename(file_path), file_data, 'image/png')
            }
            data = {'purpose': 'file-extract'} 
            
            response = requests.post(
                upload_url, 
                headers=headers, 
                data=data,          
                files=files,        
                timeout=60
            )
            
            # --- 🚨 诊断点：检查状态码并返回详细信息 ---
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    return data.get('url'), None # 成功返回 URL 和 None 错误
                else:
                    # 服务器返回 200，但业务失败
                    return None, f"上传业务失败: {data.get('message', '未知错误')}"
            else:
                # 返回非 200 的 HTTP 错误
                return None, f"HTTP 错误 ({response.status_code}): {response.text}"

    except Exception as e:
        return None, f"网络请求异常: {str(e)}"

# ==========================================
# 3. 阿里云 API 调用逻辑 (使用新的上传函数)
# ==========================================
def call_aliyun_wanx(prompt, control_image):
    temp_filename = "temp_sketch.png"
    control_image.save(temp_filename)
    
    try:
        # --- 🚨 捕获详细错误信息 ---
        with st.spinner("☁️ 正在上传草图到阿里云 OSS..."):
            sketch_cloud_url, upload_error = upload_file_to_aliyun(api_key, temp_filename)
            
        if upload_error:
            # 如果上传失败，直接返回错误
            return None, upload_error
            
        # 2. 发起生成请求 (略)
        rsp = ImageSynthesis.call(
            model="wanx-sketch-to-image-v1", 
            input={
                'image': sketch_cloud_url, # 使用 OSS URL
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
# 4. 界面逻辑
# ==========================================
st.title("🛋️ AI 家具设计 (最终诊断版)")

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
            # 这里的 img_url 会是 OSS 地址
            img_url, error = call_aliyun_wanx(prompt_text, cleaned_img)
            
            if error:
                status.update(label="生成失败", state="error")
                # 🚨 打印出详细的错误信息
                st.error(f"失败原因：{error}") 
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
