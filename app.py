import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops
import io
import os
import requests
import dashscope
from dashscope import ImageSynthesis

# --- 🛡️ 稳健的导入检查 ---
try:
    from dashscope.file import File
    SDK_READY = True
except ImportError:
    SDK_READY = False

# ==========================================
# 1. 基础配置
# ==========================================
st.set_page_config(page_title="AI 家具设计 (阿里云版)", page_icon="🛋️", layout="wide")

# 检查 SDK 版本
if not SDK_READY:
    st.error(f"🚨 环境错误：当前 dashscope 版本过低 ({dashscope.__version__})。")
    st.warning("请执行以下操作修复：\n1. 确保 requirements.txt 里写了 dashscope>=1.19.0\n2. 在 Streamlit 后台删除此 App 并重新部署 (Re-deploy)。")
    st.stop()

# 读取并设置 API Key
try:
    api_key = st.secrets["DASHSCOPE_API_KEY"]
    dashscope.api_key = api_key
except Exception as e:
    st.error("❌ 未找到密钥！请在 Secrets 中配置 DASHSCOPE_API_KEY")
    st.stop()

# ==========================================
# 2. 图像处理函数
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

# ==========================================
# 3. 阿里云 API 调用
# ==========================================
def call_aliyun_wanx(prompt, control_image):
    # 保存临时文件
    temp_filename = "temp_sketch.png"
    control_image.save(temp_filename)
    
    try:
        with st.spinner("☁️ 正在上传草图到阿里云内网..."):
            # 这里的 File 引用的是开头导入成功的模块
            file_url_obj = File.upload(temp_filename)
            sketch_url = file_url_obj.url
            
        rsp = ImageSynthesis.call(
            model="wanx-sketch-to-image-v1", 
            prompt=prompt + ", 室内设计, 家具, 8k分辨率, 杰作, 高清材质, 柔和光线",
            sketch_image_url=sketch_url,
            n=1,
            size='1024*1024'
        )
        
        if rsp.status_code == 200:
            img_url = rsp.output.results[0].url
            return img_url, None
        else:
            return None, f"阿里云报错: {rsp.code} - {rsp.message}"
            
    except Exception as e:
        return None, f"SDK 异常: {str(e)}"

# ==========================================
# 4. 界面逻辑
# ==========================================
st.title("🛋️ AI 家具设计 (阿里云版)")

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
