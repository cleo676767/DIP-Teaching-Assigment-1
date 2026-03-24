import gradio as gr
import cv2
import numpy as np

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])

def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):
    # Convert the PIL image to a NumPy array (RGB)
    image = np.array(image)

    # Pad image to avoid cutting edges when rotating/translating
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros(
        (pad_size * 2 + image.shape[0], pad_size * 2 + image.shape[1], 3),
        dtype=np.uint8,
    ) + np.array((255, 255, 255), dtype=np.uint8).reshape(1, 1, 3)
    image_new[
        pad_size : pad_size + image.shape[0], pad_size : pad_size + image.shape[1]
    ] = image
    image = image_new

    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    M_scale_rotate = cv2.getRotationMatrix2D(center, rotation, scale)

    M_translate = np.array([[1, 0, translation_x], [0, 1, translation_y]], dtype=np.float32)

    if flip_horizontal:
        M_flip = np.array([[-1, 0, w], [0, 1, 0]], dtype=np.float32)  
    else:
        M_flip = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    M_combined = to_3x3(M_translate) @ to_3x3(M_scale_rotate) @ to_3x3(M_flip)
    M_final = M_combined[:2, :]  

    transformed_image = cv2.warpAffine(
        image, M_final, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255)
    )

    return transformed_image

def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## 🌀 图像变换沙盒 (Image Transformation Playground)")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="上传图片 Upload Image")

                scale = gr.Slider(
                    minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale 缩放"
                )
                rotation = gr.Slider(
                    minimum=-180,
                    maximum=180,
                    step=1,
                    value=0,
                    label="Rotation (degrees) 旋转角度",
                )
                translation_x = gr.Slider(
                    minimum=-300, maximum=300, step=10, value=0, label="Translation X 平移X"
                )
                translation_y = gr.Slider(
                    minimum=-300, maximum=300, step=10, value=0, label="Translation Y 平移Y"
                )
                flip_horizontal = gr.Checkbox(label="Flip Horizontal 水平翻转")

            image_output = gr.Image(label="变换后图像 Transformed Image")

        inputs = [
            image_input,
            scale,
            rotation,
            translation_x,
            translation_y,
            flip_horizontal,
        ]

        for inp in inputs:
            inp.change(apply_transform, inputs, image_output)

    return demo


if __name__ == "__main__":
    interactive_transform().launch()
