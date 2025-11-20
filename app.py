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
# 1. 基础配置与密钥读取
# ==========================================
st.set_page_config(page_title="AI 家具设计全自动生成器", page_icon="🛋️", layout="wide")

try:
    ACCESS_KEY = st.secrets["LIBLIB_ACCESS_KEY"]
    SECRET_KEY = st.secrets["LIBLIB_SECRET_KEY"]
    MODEL_UUID = st.secrets["LIBLIB_TEMPLATE_UUID"]
except Exception as e:
    st.error("❌ 配置缺失！请在 .streamlit/secrets.toml 中填入 Key 和 UUID")
    st.stop()


# ==========================================
# 2. 核心：LiblibAI 签名生成函数 (HMAC-SHA1)
# ==========================================
def get_liblib_headers(uri):
    """
    根据 LiblibAI 文档逻辑生成签名
    uri: 接口路径，如 '/api/www/v1/generation/image'
    """
    timestamp = str(int(time.time() * 1000))
    signature_nonce = str(uuid.uuid4())

    # 1. 拼接签名原串: uri & timestamp & nonce
    content = '&'.join((uri, timestamp, signature_nonce))

    # 2. HMAC-SHA1 加密
    digest = hmac.new(
        SECRET_KEY.encode('utf-8'),
        content.encode('utf-8'),
        hashlib.sha1
    ).digest()

    # 3. Base64 编码并去除尾部等号
    sign = base64.urlsafe_b64encode(digest).rstrip(b'=').decode('utf-8')

    # 4. 构造请求头 (Key名称严格遵循文档)
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
    """清洗图片：去底色，转黑白"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    # 自适应二值化：由灰变黑白，去除阴影
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    return Image.fromarray(binary)


def process_multiply(render_img, sketch_img):
    """正片叠底：保留线稿"""
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
# 4. API 调用逻辑
# ==========================================
def call_liblib_api(prompt, control_image):
    # --- A. 定义接口 ---
    domain = "https://api.liblib.art"
    submit_uri = "/api/www/v1/generation/image"

    # --- B. 准备参数 ---
    base64_img = image_to_base64(control_image)

    # ⚠️ 这里的 controlnet 类型 'canny' 可能需要根据你的模型调整
    # 如果是涂鸦，改成 'scribble'
    payload = {
        "template_uuid": MODEL_UUID,
        "generate_params": {
            "prompt": prompt + ", interior design, furniture, 8k, best quality",
            "steps": 25,
            "width": 1024,
            "height": 1024,
            "controlnet": {
                "units": [
                    {
                        "type": "canny",  # 线稿控制
                        "weight": 0.8,
                        "image_base64": base64_img
                    }
                ]
            }
        }
    }

    # --- C. 发起请求 ---
    headers = get_liblib_headers(submit_uri)  # 获取签名头

    try:
        response = requests.post(domain + submit_uri, headers=headers, json=payload)

        if response.status_code != 200:
            return None, f"提交失败: {response.text}"

        data = response.json()
        if data.get('code') != 0:
            return None, f"API 报错: {data.get('msg')}"

        generate_uuid = data['data']['generate_uuid']

    except Exception as e:
        return None, f"请求异常: {e}"

    # --- D. 轮询查询结果 ---
    status_uri = "/api/www/v1/generation/status"

    progress_bar = st.progress(0, text="☁️ 请求已提交，等待 GPU 响应...")

    for i in range(60):  # 等待 60次 * 2秒 = 2分钟
        time.sleep(2)
        progress_bar.progress((i + 1) / 60, text="☁️ AI 正在渲染材质...")

        # 查询也要签名！
        check_headers = get_liblib_headers(status_uri)

        try:
            check_res = requests.get(
                domain + status_uri,
                headers=check_headers,
                params={"generate_uuid": generate_uuid}
            )
            res_data = check_res.json()

            # 1=成功, -1=失败 (依据文档)
            if res_data['data']['status'] == 1:
                progress_bar.progress(1.0, text="渲染完成！")
                return res_data['data']['images'][0]['image_url'], None
            elif res_data['data']['status'] == -1:
                return None, "服务端生成失败，请检查参数或额度"
        except:
            pass

    return None, "等待超时"


# ==========================================
# 5. 网页界面
# ==========================================
st.title("🛋️ AI 家具设计工作流 (Liblib签名版)")

col_input, col_process = st.columns([1, 2])

with col_input:
    st.info("💡 请上传白底黑线的草图，或拍照上传（会自动清洗）")
    uploaded_file = st.file_uploader("上传草图", type=["jpg", "png", "jpeg"])
    prompt_text = st.text_area("设计描述", "modern wardrobe, walnut wood texture, soft lighting", height=100)
    run_btn = st.button("🚀 开始生成", type="primary", use_container_width=True)

if run_btn and uploaded_file:
    with col_process:
        # 状态容器
        with st.status("全自动处理中...", expanded=True) as status:
            # 1. 清洗
            st.write("1️⃣ 正在清洗草图噪点...")
            uploaded_file.seek(0)
            cleaned_img = process_clean_sketch(uploaded_file)

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
        st.image(final_img, caption="最终设计图", use_column_width=True)

        # 下载按钮
        buf = io.BytesIO()
        final_img.save(buf, format="JPEG", quality=95)
        st.download_button("⬇️ 下载图片", buf.getvalue(), "design_final.jpg", "image/jpeg", type="primary")