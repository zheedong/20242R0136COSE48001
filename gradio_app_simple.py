import os
from io import BytesIO

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler, AutoencoderKL, ControlNetModel, StableDiffusionControlNetPipeline
from insightface.app import FaceAnalysis
from insightface.utils import face_align

from uniportrait import inversion
from uniportrait.uniportrait_attention_processor import attn_args
from uniportrait.uniportrait_pipeline import UniPortraitPipeline

port = 7860

device = "cuda"
torch_dtype = torch.float16

# Base model paths
base_model_path = "SG161222/Realistic_Vision_V5.1_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
controlnet_pose_ckpt = "lllyasviel/control_v11p_sd15_openpose"

# Specific model paths
image_encoder_path = "models/IP-Adapter/models/image_encoder"
ip_ckpt = "models/IP-Adapter/models/ip-adapter_sd15.bin"
face_backbone_ckpt = "models/glint360k_curricular_face_r101_backbone.bin"
uniportrait_faceid_ckpt = "models/uniportrait-faceid_sd15.bin"
uniportrait_router_ckpt = "models/uniportrait-router_sd15.bin"

# Load ControlNet
pose_controlnet = ControlNetModel.from_pretrained(controlnet_pose_ckpt, torch_dtype=torch_dtype)

# Load Stable Diffusion pipeline
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch_dtype)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=[pose_controlnet],
    torch_dtype=torch_dtype,
    scheduler=noise_scheduler,
    vae=vae,
)

# Load UniPortrait pipeline
uniportrait_pipeline = UniPortraitPipeline(
    pipe,
    image_encoder_path,
    ip_ckpt=ip_ckpt,
    face_backbone_ckpt=face_backbone_ckpt,
    uniportrait_faceid_ckpt=uniportrait_faceid_ckpt,
    uniportrait_router_ckpt=uniportrait_router_ckpt,
    device=device,
    torch_dtype=torch_dtype,
)

# Load face detection assets
face_app = FaceAnalysis(providers=['CUDAExecutionProvider'], allowed_modules=["detection"])
face_app.prepare(ctx_id=0, det_size=(640, 640))


def pad_np_bgr_image(np_image, scale=1.25):
    assert scale >= 1.0, "scale should be >= 1.0"
    pad_scale = scale - 1.0
    h, w = np_image.shape[:2]
    top = bottom = int(h * pad_scale)
    left = right = int(w * pad_scale)
    ret = cv2.copyMakeBorder(np_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return ret, (left, top)


def process_faceid_image(pil_faceid_image):
    np_faceid_image = np.array(pil_faceid_image.convert("RGB"))
    img = cv2.cvtColor(np_faceid_image, cv2.COLOR_RGB2BGR)
    faces = face_app.get(img)  # BGR image
    if len(faces) == 0:
        # Padding and try again
        _h, _w = img.shape[:2]
        _img, left_top_coord = pad_np_bgr_image(img)
        faces = face_app.get(_img)
        if len(faces) == 0:
            print("Warning: No face detected in the image. Continue processing...")
            return None

        min_coord = np.array([0, 0])
        max_coord = np.array([_w, _h])
        sub_coord = np.array([left_top_coord[0], left_top_coord[1]])
        for face in faces:
            face.bbox = np.minimum(
                np.maximum(face.bbox.reshape(-1, 2) - sub_coord, min_coord), max_coord
            ).reshape(4)
            face.kps = face.kps - sub_coord

    faces = sorted(
        faces,
        key=lambda x: abs((x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])),
        reverse=True,
    )
    if len(faces) == 0:
        print("No faces found in the image.")
        return None
    faceid_face = faces[0]
    norm_face = face_align.norm_crop(img, landmark=faceid_face.kps, image_size=224)
    pil_faceid_align_image = Image.fromarray(
        cv2.cvtColor(norm_face, cv2.COLOR_BGR2RGB)
    )

    return pil_faceid_align_image


def prepare_single_faceid_cond_kwargs(pil_faceid_image=None):
    pil_faceid_align_images = []
    if pil_faceid_image:
        processed_image = process_faceid_image(pil_faceid_image)
        if processed_image is not None:
            pil_faceid_align_images.append(processed_image)

    single_faceid_cond_kwargs = None
    if len(pil_faceid_align_images) > 0:
        single_faceid_cond_kwargs = {"refs": pil_faceid_align_images}

    return single_faceid_cond_kwargs


def text_to_multi_id_generation_process(pil_faceid_images):
    # Fixed parameters
    faceid_scale = 0.7
    face_structure_scale = 0.3
    num_samples_per_prompt = 4
    image_resolution = "512x768"
    inference_steps = 25

    # Fixed prompts and negative prompts
    prompts = [
        "A heartfelt reminder wedding portrait in front of Gyeongbokgung Palace: parents and their son and daughter dressed in traditional Korean hanbok, wearing an elegant white hanbok with intricate lace details, both holding colorful floral bouquets. The family stands closely together, smiling warmly, with the majestic architecture of Gyeongbokgung Palace and a serene blue sky in the background, creating a harmonious and timeless atmosphere.",
        "A DSLR photo of a heartfelt reminder wedding portrait on a tropical beach: a couple seated on a fallen palm tree by the shore, the man in a white suit with a black bowtie and the woman in an elegant white lace wedding dress, holding a bouquet of colorful flowers. Seated between them are their young daughter in a delicate white lace dress with floral accessories, and their young son with a white shirt and beige trousers. The background features lush green palm trees, a serene blue ocean, and beautiful sunlight, creating a warm and idyllic atmosphere. Extreme detail, high quality.",
        "A heartfelt reminder wedding portrait in a professional studio: a couple seated at the center, the man in a beige suit with a bowtie and the woman in an elegant white lace wedding dress, both holding colorful floral bouquets. Surrounding them are their two children: a son in a light pink shirt and trousers, and a daughter dressed in white dresses with delicate lace details, each wearing floral crowns and holding small bouquets. The family stands closely together, smiling warmly, with a soft pink backdrop and even studio lighting emphasizing the harmonious and joyful atmosphere.",
        "A festive holiday portrait in front of a decorated fireplace: the family wearing coordinated cozy sweaters with holiday patterns. The parents stand behind the children, who are seated on a plush rug holding gift boxes. The room is filled with holiday decorations like wreaths, stockings, and a beautifully lit Christmas tree, creating a warm and joyful atmosphere.",
    ]

    negative_prompts = [
        "two people, one person, three people, out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature,",
        "two people, one person, three people, out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature,",
        "two people, one person, three people, out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature,",
        "two people, one person, three people, out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature,",
    ]

    seeds = [3, 3, 0, 0]  # Seeds for each prompt

    # Process images
    pil_images = []
    for file in pil_faceid_images:
        if isinstance(file, bytes):
            pil_image = Image.open(BytesIO(file)).convert("RGB")
        else:
            pil_image = file
        pil_images.append(pil_image)

    cond_faceids = []
    for pil_faceid_image in pil_images:
        faceid_cond_kwargs = prepare_single_faceid_cond_kwargs(pil_faceid_image)
        if faceid_cond_kwargs is not None:
            cond_faceids.append(faceid_cond_kwargs)

    # Reset attention arguments
    attn_args.reset()
    attn_args.lora_scale = 1.0 if len(cond_faceids) == 1 else 0.0
    attn_args.multi_id_lora_scale = 1.0 if len(cond_faceids) > 1 else 0.0
    attn_args.faceid_scale = faceid_scale if len(cond_faceids) > 0 else 0.0
    attn_args.num_faceids = len(cond_faceids)
    print(attn_args)

    # Generate images
    h, w = map(int, image_resolution.split("x"))
    images = []
    for prompt, negative_prompt, seed in zip(prompts, negative_prompts, seeds):
        prompts_list = [prompt] * num_samples_per_prompt
        negative_prompts_list = [negative_prompt] * num_samples_per_prompt
        generated_images = uniportrait_pipeline.generate(
            prompt=prompts_list,
            negative_prompt=negative_prompts_list,
            cond_faceids=cond_faceids,
            face_structure_scale=face_structure_scale,
            seed=seed if seed != -1 else None,
            guidance_scale=7.5,
            num_inference_steps=inference_steps,
            image=[torch.zeros([1, 3, h, w])],
            controlnet_conditioning_scale=[0.0],
        )
        images.extend(generated_images)

    # Collect final outputs
    final_out = []
    for pil_image in images:
        final_out.append(pil_image)

    return final_out


def text_to_multi_id_generation_block():
    gr.Markdown("## 가족 사진 생성")
    gr.HTML(text_to_multi_id_description)
    gr.HTML(text_to_multi_id_tips)
    with gr.Row():
        with gr.Column(scale=2, min_width=100):
            pil_faceid_images = gr.File(
                file_count="multiple",
                file_types=["image"],
                type="binary",
                label="얼굴 이미지 업로드 (최대 5개)",
            )
        with gr.Column(scale=1, min_width=100):
            run_button = gr.Button(value="생성하기")
    with gr.Row():
        result_gallery = gr.Gallery(
            label="생성된 이미지",
            show_label=True,
            elem_id="gallery",
            columns=4,
            preview=True,
            format="png",
        )

    ips = [pil_faceid_images]
    run_button.click(fn=text_to_multi_id_generation_process, inputs=ips, outputs=[result_gallery])


if __name__ == "__main__":
    os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

    title = r"""
            <div style="text-align: center;">
                <h1> 가족 사진 생성 데모 </h1>
                <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                    <p>2024학년도 2학기 산학캡스톤 디자인 '기술 한 스푼' 팀이 이 프로젝트를 진행했습니다.</p>
                    <p>원본 논문은 UniPortrait를 사용하였습니다.</p>
                </div>
                </br>
            </div>
        """

    title_description = r"""
        이 데모는 고정된 프롬프트와 업로드한 얼굴 이미지를 사용하여 이미지를 생성합니다.<br>
        각 프롬프트마다 4장의 이미지를 생성하며, 총 16장의 이미지를 생성합니다.<br>
        """

    text_to_multi_id_description = r"""🚀🚀🚀빠른 시작:<br>
        1. 얼굴이 있는 여러 이미지를 업로드한 후 <b>생성하기</b> 버튼을 클릭하세요. 🤗<br>
        """

    text_to_multi_id_tips = r"""💡💡💡팁:<br>
        1. 업로드하는 이미지에 선명한 얼굴이 포함되어 있어야 더 좋은 결과를 얻을 수 있습니다.<br>
        2. 생성된 이미지는 아래 갤러리에 표시됩니다.<br>
        """

    block = gr.Blocks(title="가족 사진 생성").queue()
    with block:
        gr.HTML(title)
        gr.HTML(title_description)

        with gr.TabItem("텍스트-다중-아이디 생성"):
            text_to_multi_id_generation_block()

    block.launch(server_name='0.0.0.0', share=True, server_port=port, allowed_paths=["/"])