import sys
sys.path.append('./lib')

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import torch
import kornia as K
import kornia.feature as KF
import io
from streamlit.components.v1 import html

st.set_page_config(layout="wide", page_title=" Image Stitching Hub", page_icon="🌌")

# Import custom SIFT pipeline
from sift_functions import generate_image_pyramid, calculate_dog, SIFT_feature_detection
from helper_functions import quick_resize
from image_stitcher import convert_keypoints_to_cv2, get_descriptors, match_keypoints

# Load LoFTR model
@st.cache_resource
def load_loftr():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KF.LoFTR(pretrained='outdoor').to(device).eval()
    if device.type == 'cuda':
        model = model.half()
    return model, device

loftr_model, device = load_loftr()
use_half = device.type == 'cuda'

# LoFTR stitching logic
@torch.inference_mode()
def stitch_with_loftr(img1_bgr, img2_bgr):
    img1_gray = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

    tensor1 = K.image_to_tensor(img1_gray, False).to(device)
    tensor2 = K.image_to_tensor(img2_gray, False).to(device)

    if use_half:
        tensor1 = tensor1.half() / 255.0
        tensor2 = tensor2.half() / 255.0
    else:
        tensor1 = tensor1.float() / 255.0
        tensor2 = tensor2.float() / 255.0

    batch = {"image0": tensor1, "image1": tensor2}
    output = loftr_model(batch)

    mkpts0 = output['keypoints0'].cpu().numpy()
    mkpts1 = output['keypoints1'].cpu().numpy()

    if len(mkpts0) < 4:
        return None, " Not enough matches found."

    H, _ = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 5.0)
    if H is None:
        return None, " Homography estimation failed."

    return warp_images(img1_bgr, img2_bgr, H), None

# Common warping function
def warp_images(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners2, H)
    all_corners = np.concatenate((np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2), transformed_corners), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = [-xmin, -ymin]
    H_translation = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])

    panorama = cv2.warpPerspective(img2, H_translation @ H, (xmax - xmin, ymax - ymin))
    panorama[translation[1]:h1 + translation[1], translation[0]:w1 + translation[0]] = img1

    return panorama

# SIFT-based stitching logic
def stitch_with_sift(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    octaves1 = generate_image_pyramid(gray1)
    octaves2 = generate_image_pyramid(gray2)
    dog1 = calculate_dog(octaves1)
    dog2 = calculate_dog(octaves2)

    kp1 = SIFT_feature_detection(dog1, gray1)
    kp2 = SIFT_feature_detection(dog2, gray2)

    cv2_kp1 = convert_keypoints_to_cv2(kp1)
    cv2_kp2 = convert_keypoints_to_cv2(kp2)
    kp1, desc1 = get_descriptors(gray1, cv2_kp1)
    kp2, desc2 = get_descriptors(gray2, cv2_kp2)

    matches = match_keypoints(desc1, desc2)
    if len(matches) < 4:
        return None, " Not enough matches."

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None, " Homography computation failed."

    return warp_images(img1, img2, H), None

# ------------------------ Streamlit Frontend ------------------------ #

# Inject CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Header with flip animation on section visibility
st.markdown("""
    <div class='header flip-section' id='header-section'>
        <h1>🌌 Image Stitching Hub</h1>
        <p>Stitch your images with SIFT or LoFTR, Upload or capture with webcam.</p>
    </div>
""", unsafe_allow_html=True)

# JavaScript for flip animation on section visibility
html("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    const section = document.getElementById('header-section');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                section.classList.add('flipped');
                observer.unobserve(section);
            }
        });
    }, { threshold: 0.5 });
    observer.observe(section);
});
</script>
""", height=0)

# Method selection with cards
st.markdown("<div class='method-container'>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
        <div class='method-card'>
            <h3>SIFT</h3>
            <ul>
                <li>Classic feature detection</li>
                <li>Gaussian pyramid blending</li>
                <li>Robust for textured images</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
        <div class='method-card'>
            <h3>LoFTR</h3>
            <ul>
                <li>Deep learning-based</li>
                <li>Great for low-texture areas</li>
                <li>Modern and precise</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
        <div class='method-card'>
            <h3>Both</h3>
            <ul>
                <li>Compare SIFT & LoFTR</li>
                <li>See results side-by-side</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

method = st.radio("Choose your stitching method:", ["SIFT", "LoFTR", "Both"], horizontal=True, label_visibility="collapsed")

# Image input section
st.markdown("<div class='content-block'><h2>Input Your Images</h2></div>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["Upload Your Images", "Capture from Webcam"])

with tab1:
    col_left, col_right = st.columns(2)
    with col_left:
        left_file = st.file_uploader("LEFT Image", type=["jpg", "jpeg", "png"], key="left_upload", help="Upload the left part of your panorama")
    with col_right:
        right_file = st.file_uploader("RIGHT Image", type=["jpg", "jpeg", "png"], key="right_upload", help="Upload the right part of your panorama")

with tab2:
    col_left_cam, col_right_cam = st.columns(2)
    with col_left_cam:
        left_cam = st.camera_input("Capture LEFT Image", key="left_cam")
    with col_right_cam:
        right_cam = st.camera_input("Capture RIGHT Image", key="right_cam")

# Load images
def load_image(file_or_bytes):
    if file_or_bytes is not None:
        if isinstance(file_or_bytes, bytes):
            img = Image.open(io.BytesIO(file_or_bytes)).convert("RGB")
        else:
            img = Image.open(file_or_bytes).convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return None

img1 = load_image(left_file) if left_file else load_image(left_cam)
img2 = load_image(right_file) if right_file else load_image(right_cam)

if img1 is not None and img2 is not None:
    st.success(" Images loaded successfully! Ready to stitch.")

    # Stitch button with flip animation
    st.markdown("<div class='cta-container'><button id='start-button' class='flip-button'>🚀 Stitch the Images</button></div>", unsafe_allow_html=True)
    if st.button("Stitch the Images", key="stitch"):
        if method == "SIFT":
            with st.spinner("Stitching with SIFT across the stars..."):
                result, error = stitch_with_sift(img1, img2)
        elif method == "LoFTR":
            with st.spinner("Stitching with LoFTR through the nebula..."):
                result, error = stitch_with_loftr(img1, img2)
        elif method == "Both":
            col1, col2 = st.columns(2)
            with st.spinner("Running both methods through the cosmos..."):
                res_sift, err1 = stitch_with_sift(img1, img2)
                res_loftr, err2 = stitch_with_loftr(img1, img2)

            if res_sift is not None:
                col1.image(cv2.cvtColor(res_sift, cv2.COLOR_BGR2RGB), caption="SIFT Panorama", use_container_width=True)
            else:
                col1.error(err1)
            if res_loftr is not None:
                col2.image(cv2.cvtColor(res_loftr, cv2.COLOR_BGR2RGB), caption="LoFTR Panorama", use_container_width=True)
            else:
                col2.error(err2)
            st.stop()

        if result is not None:
            st.markdown("<div class='result-boom'>🎉 Boom! It's your result!</div>", unsafe_allow_html=True)
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption=f"🚀 {method} Panorama", use_container_width=True)
            # Download button
            buf = io.BytesIO()
            img_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            img_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="💾 Download Panorama",
                data=byte_im,
                file_name="stitched_panorama.png",
                mime="image/png"
            )
        else:
            st.error(error)

# Footer
st.markdown("""
    <footer>
        <p>Built with by [Avaneesh-Pandey | Subhash-Mishra | Nikhil-kumar | Jatin] | Deployed on Hugging Face Spaces</p>
    </footer>
""", unsafe_allow_html=True)