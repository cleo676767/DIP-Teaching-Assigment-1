import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):    
    if image is None:
        return None

    img = np.array(image)
    h, w = img.shape[:2]

    if len(source_pts) == 0 or len(target_pts) == 0:
        return img

    n = min(len(source_pts), len(target_pts))
    p = source_pts[:n].astype(np.float32)  # source
    q = target_pts[:n].astype(np.float32)  # target

    p_map = q  
    q_map = p  

    yy, xx = np.mgrid[0:h, 0:w]
    X = np.stack([xx, yy], axis=-1).astype(np.float32)  # (h,w,2)

    # X: (h,w,2), p_map: (n,2)
    diff = X[..., None, :] - p_map[None, None, :, :]      # (h,w,n,2)
    dist2 = np.sum(diff * diff, axis=-1) + eps            # (h,w,n)
    W = 1.0 / (dist2 ** alpha)                            # (h,w,n)
    W_sum = np.sum(W, axis=-1, keepdims=True) + eps       # (h,w,1)

    p_star = np.sum(W[..., None] * p_map[None, None, :, :], axis=2) / W_sum  # (h,w,2)
    q_star = np.sum(W[..., None] * q_map[None, None, :, :], axis=2) / W_sum  # (h,w,2)

    # f(x) = x + (q_star - p_star)
    disp = q_star - p_star
    X_src = X + disp  

    map_x = X_src[..., 0].astype(np.float32)
    map_y = X_src[..., 1].astype(np.float32)

    warped = cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )
    return warped

    warped_image = np.array(image)
    ### FILL: Implement MLS or RBF based image warping

    return warped_image

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()
