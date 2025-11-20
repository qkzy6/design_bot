import streamlit as st
import dashscope
from dashscope import ImageSynthesis
import os
import tempfile
from PIL import Image

# 1. 页面基础设置
st.set_page_config(page_title="阿里云家具渲染器", layout="wide")
st.title("🛋️ 家具草图渲染 (阿里云通义万相)")

# 2. 安全加载密钥
if "DASHSCOPE_API_KEY" in st.secrets:
    dashscope.api_key = st.secrets["DASHSCOPE_API_KEY"]
else:
    st.error("❌ 未找到密钥，请在 .streamlit/secrets.toml 配置 DASHSCOPE_API_KEY")
    st.stop()

# 3. 侧边栏设置
with st.sidebar:
    st.header("参数设置")
    # 通义万相对中文理解很好，所以默认用中文
    prompt = st.text_area(
        "描述家具细节 (支持中文):",
        value="新中式实木沙发，米白色坐垫，柔和的室内光线，高品质，4k分辨率，室内设计杂志风格",
        height=100
    )
    # 风格选择 (这是通义万相的一个特色参数)
    style = st.selectbox(
        "生成风格:",
        options=["<auto>", "realistic", "oil_painting", "watercolor", "sketch"],
        index=1,
        format_func=lambda x: "自动" if x == "<auto>" else "写实照片" if x == "realistic" else x
    )

# 4. 图片上传处理
uploaded_file = st.file_uploader("上传草图 (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    # 保存临时文件供 SDK 读取
    # Streamlit 的文件在内存里，阿里云SDK需要一个 file:// 路径
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png") 
    tfile.write(uploaded_file.getvalue())
    temp_file_path = tfile.name # 获取临时文件的绝对路径

    with col1:
        st.subheader("原始草图")
        st.image(uploaded_file, use_container_width=True)

    with col2:
        st.subheader("渲染结果")
        run_btn = st.button("🚀 开始渲染 (阿里云)", type="primary")

        if run_btn:
            try:
                with st.spinner("正在请求阿里云通义万相..."):
                    
                    # 构造文件协议路径
                    local_file_url = f"file://{temp_file_path}"

                    # 调用阿里云 API
                    resp = ImageSynthesis.call(
                        model="wanx-sketch-to-image-v1",
                        prompt=prompt,
                        sketch_image_url=local_file_url,
                        style=style if style != "<auto>" else None,
                        size='1024*1024',
                        n=1
                    )

                    # 处理返回结果
                    if resp.status_code == 200:
                        # 获取结果图片 URL
                        if resp.output and resp.output.results:
                            result_url = resp.output.results[0]['url']
                            st.image(result_url, caption="通义万相渲染结果", use_container_width=True)
                            st.success("渲染完成！")
                        else:
                            st.warning("API 返回成功但没有图片数据。")
                    else:
                        # 错误处理：提取错误信息
                        st.error(f"API 调用失败: {resp.code}")
                        st.error(f"错误信息: {resp.message}")

            except Exception as e:
                st.error(f"发生系统错误: {str(e)}")
            
            finally:
                # 清理临时文件，保持环境整洁
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
