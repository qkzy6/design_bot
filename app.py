import streamlit as st
import dashscope
from dashscope import ImageSynthesis
import os
import tempfile
from PIL import Image, ImageChops, ImageEnhance
import numpy as np

# 1. 页面基础设置
st.set_page_config(page_title="阿里云家具渲染器 (增强版)", layout="wide")
st.title("🛋️ 家具草图渲染 (阿里云通义万相) - 增强版")
st.markdown("上传你的草图，AI 渲染后可选叠加原始线条效果。")

# 2. 安全加载密钥
if "DASHSCOPE_API_KEY" in st.secrets:
    dashscope.api_key = st.secrets["DASHSCOPE_API_KEY"]
else:
    st.error("❌ 未找到密钥，请在 .streamlit/secrets.toml 配置 DASHSCOPE_API_KEY")
    st.stop()

# --- 新增：图片清洗函数 ---
def clean_sketch_background(image: Image.Image, threshold: int = 200) -> Image.Image:
    """
    通过阈值处理清洗草图背景，将接近白色的像素变为纯白透明，线条变为黑色。
    适用于白色背景的草图，去除纸张纹理和轻微阴影。
    """
    if image.mode != 'L': # 如果不是灰度图，先转灰度
        image = image.convert('L')
    
    # 将图像转换为 numpy 数组进行处理
    np_image = np.array(image)
    
    # 创建一个与图像大小相同的RGBA模式的纯白图片作为输出
    output_image_np = np.full((np_image.shape[0], np_image.shape[1], 4), 255, dtype=np.uint8)

    # 找到所有非白色像素（即线条）
    # 大于阈值的认为是背景（白色），小于等于阈值的认为是前景（线条）
    # 注意：这里的阈值处理是把“暗”的像素当成线条
    line_pixels_mask = np_image < threshold
    
    # 将线条部分设置为黑色（R=0, G=0, B=0, A=255）
    output_image_np[line_pixels_mask, :3] = 0
    # 将背景部分设置为纯白透明 (R=255, G=255, B=255, A=0)，实现背景移除效果
    # 实际上，这里我们希望线条是黑色，背景是白色（而不是透明），以更好地输入给AI
    # AI ControlNet通常是基于白底黑线或黑底白线
    # 所以我们这里直接把背景变为纯白，前景变为纯黑。
    
    # 如果要实现透明背景，可以在生成图像后使用RGBA模式并设置透明度。
    # 但对于草图AI，通常直接传黑线白底更好。
    # 所以这里修改为：线条是黑色，背景是白色
    output_image_np[~line_pixels_mask, :3] = 255 # 背景变为白色
    output_image_np[line_pixels_mask, :3] = 0   # 线条变为黑色
    output_image_np[:, 3] = 255 # 所有像素不透明 (A=255)

    return Image.fromarray(output_image_np).convert('RGB') # 返回RGB模式

# --- 新增：正片叠底函数 ---
def multiply_blend(base_image: Image.Image, blend_image: Image.Image) -> Image.Image:
    """
    对两张图片进行正片叠底融合。
    base_image: 基础图片（通常是AI渲染的彩色图）
    blend_image: 叠加图片（通常是原始草图的黑色线条图）
    """
    # 确保两张图都是 RGB 模式且大小一致
    if base_image.mode != 'RGB':
        base_image = base_image.convert('RGB')
    if blend_image.mode != 'RGB':
        blend_image = blend_image.convert('RGB')
    
    # 调整叠加图的尺寸以匹配基础图
    if base_image.size != blend_image.size:
        blend_image = blend_image.resize(base_image.size, Image.LANCZOS)

    # 将图片转换为 numpy 数组
    base_np = np.array(base_image).astype(np.float32) / 255.0
    blend_np = np.array(blend_image).astype(np.float32) / 255.0

    # 执行正片叠底计算
    # 结果颜色 = 基色 * 混合色
    # 如果blend_image是黑白线条图，白色(1.0)不改变基色，黑色(0.0)使基色变黑。
    # 为了让草图的黑线显现，需要将草图反色（白底黑线变成黑底白线，或直接让黑线与基色融合）
    # 但更直接的方法是：将草图转换为灰度图，并将其视为一个亮度通道，然后与彩色图融合。
    
    # 如果blend_image是黑白草图，我们希望黑线叠加在彩色图上
    # 做法：将草图的黑色线条部分作为乘数，白色部分（1.0）不改变颜色，黑色部分（0.0）将基色变为黑色。
    # 所以这里需要先将草图转换为灰度图，并确保黑色是0，白色是1。
    sketch_gray_np = blend_image.convert('L') # 转换为灰度
    sketch_gray_np = np.array(sketch_gray_np).astype(np.float32) / 255.0 # 归一化到0-1
    
    # 正片叠底公式：result = base * blend_alpha (其中blend_alpha是0-1的亮度值)
    # 对于黑白线条图，白色是1，黑色是0。这样，白色的地方不影响底图，黑色的地方让底图变黑。
    # 如果原始草图是白底黑线，那么线条部分像素值低，背景部分像素值高。
    # 在正片叠底中，低像素值（接近0）会导致结果变暗，高像素值（接近1）不影响结果。
    # 所以，直接用原始草图的灰度值作为乘数即可。
    
    blended_np = base_np * np.stack([sketch_gray_np, sketch_gray_np, sketch_gray_np], axis=-1)
    
    blended_image = Image.fromarray((blended_np * 255).astype(np.uint8))
    return blended_image


# 3. 侧边栏设置
with st.sidebar:
    st.header("参数设置")
    prompt = st.text_area(
        "描述家具细节:",
        value="新中式实木沙发，米白色坐垫，柔和的室内光线，高品质，4k分辨率，室内设计杂志风格",
        height=100
    )
    style = st.selectbox(
        "生成风格:",
        options=["<auto>", "realistic", "oil_painting", "watercolor", "sketch"],
        index=1,
        format_func=lambda x: "自动" if x == "<auto>" else "写实照片" if x == "realistic" else x
    )

    st.header("图像处理选项")
    # 图片清洗选项
    enable_cleaning = st.checkbox("🖼️ 启用草图背景清洗", value=True, 
                                  help="将草图背景处理为纯白，线条更清晰，有助于AI理解。适用于白底草图。")
    cleaning_threshold = st.slider("清洗阈值 (数字越低越黑)", 150, 250, 200, 5, 
                                   help="调整多少亮度以上的像素被视为背景。")

    # 正片叠底选项
    enable_blend = st.checkbox("融合原始草图线条 (正片叠底)", value=False,
                               help="将AI渲染图与原始草图线条进行正片叠底融合，保留线条感。")


# 4. 图片上传处理
uploaded_file = st.file_uploader("上传草图 (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    # 读取原始图片
    original_image = Image.open(uploaded_file).convert('RGB')
    processed_image_for_ai = original_image # 默认情况下，AI使用原始图

    with col1:
        st.subheader("原始草图 / 处理后的草图")
        # 如果启用清洗，则显示清洗后的图片
        if enable_cleaning:
            processed_image_for_ai = clean_sketch_background(original_image, cleaning_threshold)
            st.image(processed_image_for_ai, caption="清洗后的草图 (用于AI)", use_container_width=True)
        else:
            st.image(original_image, caption="原始草图", use_container_width=True)

    # --- 保存临时文件供 SDK 读取 (已修复路径问题) ---
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    # 将用于AI的图片保存到临时文件
    processed_image_for_ai.save(tfile.name, format="PNG") 
    tfile.flush()
    tfile.close()  
    temp_file_path = tfile.name
    abs_path = os.path.abspath(temp_file_path).replace('\\', '/')
    local_file_url = f"file://{abs_path}"
    # --- 临时文件处理结束 ---

    with col2:
        st.subheader("渲染结果")
        run_btn = st.button("🚀 开始渲染 (阿里云)", type="primary")

        if run_btn:
            try:
                with st.spinner("正在上传图片并请求生成..."):
                    
                    print(f"正在处理路径: {local_file_url}") # 调试信息

                    resp = ImageSynthesis.call(
                        model="wanx-sketch-to-image-v1",
                        prompt=prompt,
                        sketch_image_url=local_file_url,
                        style=style if style != "<auto>" else None,
                        size='1024*1024',
                        n=1
                    )

                    if resp.status_code == 200:
                        if resp.output and resp.output.results:
                            result_url = resp.output.results[0]['url']
                            st.info("AI 渲染图已生成。")
                            
                            # 下载 AI 渲染图
                            ai_rendered_image = Image.open(requests.get(result_url, stream=True).raw).convert('RGB')

                            final_display_image = ai_rendered_image
                            # 如果启用正片叠底
                            if enable_blend:
                                # 将原始草图转换为灰度（黑线白底）用于正片叠底
                                # 并确保它和AI图尺寸一致
                                blended_sketch = original_image.resize(ai_rendered_image.size).convert('L')
                                # 将灰度图转为RGB以便与彩色图融合
                                blended_sketch = blended_sketch.convert('RGB')
                                
                                final_display_image = multiply_blend(ai_rendered_image, blended_sketch)
                                st.image(final_display_image, caption="AI渲染图 + 原始草图正片叠底", use_container_width=True)
                                st.success("渲染并融合完成！")
                            else:
                                st.image(final_display_image, caption="AI 渲染图", use_container_width=True)
                                st.success("渲染完成！")

                        else:
                            st.warning("API 返回成功但没有图片数据。")
                    else:
                        st.error(f"API 调用失败: {resp.code}")
                        st.error(f"错误信息: {resp.message}")
                        st.caption(f"Request ID: {resp.request_id}")

            except Exception as e:
                st.error(f"发生系统错误: {str(e)}")
            
            finally:
                # 清理临时文件
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except Exception as e:
                        print(f"清理临时文件失败: {e}")

# ⚠️ 注意：为了下载图片，你需要安装 requests 库
# pip install requests
