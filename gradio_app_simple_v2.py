import os
from io import BytesIO
import random

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
from controlnet_aux import OpenposeDetector

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
pose_processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

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

def get_n_random_pose_images(folder_path: str, n: int = 4):
    """폴더 내 여러 이미지(.png, .jpg, .jpeg, .webp) 중 n개를 중복 없이 랜덤 선택."""
    valid_ext = {'.png', '.jpg', '.jpeg', '.webp'}
    if not os.path.exists(folder_path):
        return []

    files = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if os.path.splitext(f.lower())[-1] in valid_ext
    ]
    if len(files) < n:
        print(f"Warning: '{folder_path}'에 이미지가 {len(files)}장밖에 없어 {n}개 모두 사용할 수 없습니다.")
        # 필요하다면 random.sample() 호출 시 에러가 납니다. 
        # 여기서는 그냥 files 전체를 반환하거나, 중복 허용(random.choices)을 사용할 수 있음.
        # 여기서는 중복 허용 버전으로 할게요.
        return random.choices(files, k=n)  # 중복 허용
        
    return random.sample(files, k=n)

def text_to_multi_id_generation_process(pil_faceid_images):
    num_persons = len(pil_faceid_images)

    if num_persons == 2:
        num_persons_word = "two"
        negative_prompt = "nsfw, extra hands, extra arms, one person, three people, out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature,",
        negative_prompts = negative_prompt * 4
        pose_folder = "assets/2_people"

    elif num_persons == 3:
        num_persons_word = "three"
        negative_prompt = "nsfw, extra hands, extra arms, one person, two people, out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature,",
        negative_prompts = negative_prompt * 4
        pose_folder = "assets/3_people"

    elif num_persons == 4:
        num_persons_word = "four"
        negative_prompt = "nsfw, extra hands, extra arms, one person, two people, three people, out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature,",
        negative_prompts = negative_prompt * 4
        pose_folder = "assets/4_people"

    else:
        num_persons_word = ""
        negative_prompt = "nsfw, extra hands, extra arms, out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature,",
        negative_prompts = negative_prompt * 4
        pose_folder = None  # ControlNet 미적용

    # prompts/negative_prompts는 4개씩 있다고 가정
    # (각각의 prompt에 대해 서로 다른 pose를 매칭)
    prompts = [
        f"A DSLR photo of moody and atmospheric film noir scene featuring {num_persons_word} people with intense and brooding expressions, captured in dramatic lighting and shadowy contrasts for a sense of mystery and intrigue. High qaulity, Greyscale, Cinematic",
        f"A vivid and graphic comic book-style illustration of {num_persons_word} people with animated and exaggerated expressions, featuring larger-than-life contours and dynamic action that convey a sense of lightheartedness and fun.",
        f"A photo of {num_persons_word} people in a family",
        f"A photo of {num_persons_word} people, best quality, realistic details, vivid colors, natural lighting",
    ]

    # 이제 각 프롬프트마다 1장의 이미지를 생성하므로, 총 4×1=4장
    # 하지만 "프롬프트 x 포즈"를 4×4=16장 만들고 싶다면 
    # -> 각 prompt마다 서로 다른 pose 이미지를 매칭해야 함.
    # pose_image_list = get_n_random_pose_images(pose_folder, n=4)
    # => 이 4개의 pose 이미지와 4개의 prompt를 1:1 매칭 (prompt i -> pose_image_list[i])
    # => num_samples_per_prompt = 1
    # => 최종 4 prompts × 4 pose = 4×1 = 4장이 아니라, 각 prompt에 대해 1장씩만 생성하므로 4장 총합
    # 
    # 만약 "각 prompt + 각 pose"로 조합(4prompt×4pose=16장)을 만들려면 
    # 이중루프나 다른 로직을 써야 함.
    
    # 여기서는 "각 prompt에 대해 하나씩, 그 pose도 각자 달라서 4장"이 아니라
    # "prompt 4개 × pose 4개 → 총 16장"을 만들려면:
    # 1) pose 이미지를 4개 뽑아 놓음
    # 2) 바깥 루프는 pose 4개, 안쪽 루프는 prompt 4개
    #    or prompt 4개를 돌면서, pose 4개를 각각 써서 4×4=16장 
    
    # 아래 예시는 "각 pose마다 모든 prompt를 반복" => 16장
    pose_image_list = []
    if pose_folder:
        pose_image_list = get_n_random_pose_images(pose_folder, n=4)
    else:
        # ControlNet 미적용: pose_list는 None으로 처리
        pose_image_list = [None]*4

    # 여기서 num_samples_per_prompt = 1로 하면
    #   (4 pose × 4 prompt × 1 sample) = 16장
    num_samples_per_prompt = 1

    inference_steps = 35
    faceid_scale = 1.0
    face_structure_scale = 0.5
    guidance_scale = 8.5

    # prompt가 4개, pose도 4개. 
    # "각 포즈에 대해 모든 프롬프트를 생성" 하는 방식.
    # 따라서 바깥 루프: pose_image_list 4개
    #        안쪽 루프: prompts 4개
    # 총 4×4=16장
    # 시드는 4개뿐이므로, 4개의 prompt에 맞춰 반복해서 사용
    seeds = [10, 10, 10, 10]

    # 업로드된 이미지 바이너리 → PIL 변환
    pil_images = []
    for file in pil_faceid_images:
        if isinstance(file, bytes):
            pil_image = Image.open(BytesIO(file)).convert("RGB")
        else:
            pil_image = file
        pil_images.append(pil_image)

    # faceid cond kwargs 구성
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

    image_resolution = "512x768"
    h, w = map(int, image_resolution.split("x"))

    final_out = []

    # 바깥 루프(포즈 4개)
    for i, pose_path in enumerate(pose_image_list):
        if pose_path and os.path.exists(pose_path):
            controlnet_conditioning_image = Image.open(pose_path).convert("RGB")
            controlnet_conditioning_image = pose_processor(controlnet_conditioning_image)
            print("ControlNet 적용:", pose_path)
            controlnet_scale = [0.8]  # ControlNet 적용 강도
        else:
            print("ControlNet 미적용")
            controlnet_conditioning_image = None
            controlnet_scale = [0.0]

        # 안쪽 루프(프롬프트 4개)
        for j, (prompt, negative_prompt) in enumerate(zip(prompts, negative_prompts)):
            seed = seeds[j % len(seeds)]  # prompt 인덱스에 맞게 seed 선택

            # prompt를 num_samples_per_prompt 번 생성
            prompts_list = [prompt]*num_samples_per_prompt
            negative_prompts_list = [negative_prompt]*num_samples_per_prompt

            print("Prompt:", prompt)

            images = uniportrait_pipeline.generate(
                prompt=prompts_list,
                negative_prompt=negative_prompts_list,
                cond_faceids=cond_faceids,
                face_structure_scale=face_structure_scale,
                seed=seed if seed != -1 else None,
                guidance_scale=guidance_scale,
                num_inference_steps=inference_steps,
                image=[controlnet_conditioning_image] if controlnet_conditioning_image else None,
                controlnet_conditioning_scale=controlnet_scale,
            )
            final_out.extend(images)

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
                label="얼굴 이미지 업로드",
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
        이 데모는 업로드한 얼굴 이미지 수에 맞춰 프롬프트와 ControlNet 포즈를 자동 적용합니다.<br>
        폴더에서 4개의 포즈 이미지를 뽑아, 4개의 프롬프트 각각과 조합하여 총 16장을 생성합니다.<br>
        """

    text_to_multi_id_description = r"""🚀🚀🚀빠른 시작:<br>
        1. 얼굴이 있는 여러 이미지를 업로드한 후 <b>생성하기</b> 버튼을 클릭하세요. 🤗<br>
        2. 업로드한 얼굴이 포함된 가족 사진을 만들어 드립니다.
        """

    text_to_multi_id_tips = r"""💡💡💡팁:<br>
        1. 배경이 깔끔한 정면 사진을 사용하는걸 추천 드립니다.
        2. 마음에 드는 사진을 저장해 주세요!
        """

    block = gr.Blocks(title="가족 사진 생성").queue()
    with block:
        gr.HTML(title)
        gr.HTML(title_description)

        with gr.TabItem("텍스트-다중-아이디 생성"):
            text_to_multi_id_generation_block()

    block.launch(server_name='0.0.0.0', share=True, server_port=port, allowed_paths=["/"])