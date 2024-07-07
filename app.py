# coding: utf-8

"""
The entrance of the gradio
"""
import os
import platform
import argparse
import tyro
import gradio as gr
import os.path as osp
from src.utils.helper import load_description
from src.gradio_pipeline import GradioPipeline
from src.config.crop_config import CropConfig
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig

def open_folder():
    open_folder_path = os.path.abspath("animations")
    if platform.system() == "Windows":
        os.startfile(open_folder_path)
    elif platform.system() == "Linux":
        os.system(f'xdg-open "{open_folder_path}"')

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

def parse_args():
    parser = argparse.ArgumentParser(description="Gradio interface for image animation")
    parser.add_argument("--share", action="store_true", help="Share the Gradio interface")
    return parser.parse_args()

cmd_args = parse_args()

# set tyro theme
tyro.extras.set_accent_color("bright_cyan")
args = tyro.cli(ArgumentConfig)

# specify configs for inference
inference_cfg = partial_fields(InferenceConfig, args.__dict__)  # use attribute of args to initial InferenceConfig
crop_cfg = partial_fields(CropConfig, args.__dict__)  # use attribute of args to initial CropConfig
gradio_pipeline = GradioPipeline(
    inference_cfg=inference_cfg,
    crop_cfg=crop_cfg,
    args=args
)


# assets
title_md = "assets/gradio_title.md"
example_portrait_dir = "assets/examples/source"
example_video_dir = "assets/examples/driving"
data_examples = [
    [osp.join(example_portrait_dir, "s9.jpg"), osp.join(example_video_dir, "d0.mp4"), True, True, True, True],
    [osp.join(example_portrait_dir, "s6.jpg"), osp.join(example_video_dir, "d0.mp4"), True, True, True, True],
    [osp.join(example_portrait_dir, "s10.jpg"), osp.join(example_video_dir, "d5.mp4"), True, True, True, True],
    [osp.join(example_portrait_dir, "s5.jpg"), osp.join(example_video_dir, "d6.mp4"), True, True, True, True],
    [osp.join(example_portrait_dir, "s7.jpg"), osp.join(example_video_dir, "d7.mp4"), True, True, True, True],
]
#################### interface logic ####################

# Define components first
eye_retargeting_slider = gr.Slider(minimum=0, maximum=0.8, step=0.01, label="target eyes-open ratio")
lip_retargeting_slider = gr.Slider(minimum=0, maximum=0.8, step=0.01, label="target lip-open ratio")
retargeting_input_image = gr.Image(type="numpy")
output_image = gr.Image(type="numpy")
output_image_paste_back = gr.Image(type="numpy")
output_video = gr.Video(include_audio=True)
output_video_concat = gr.Video(include_audio=True)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML(load_description(title_md))
    gr.Markdown(load_description("assets/gradio_description_upload.md"))
    with gr.Row():
        with gr.Accordion(open=True, label="Source Portrait"):
            image_input = gr.Image(type="filepath")
        with gr.Accordion(open=True, label="Driving Video"):
            video_input = gr.Video()
    gr.Markdown(load_description("assets/gradio_description_animation.md"))
    with gr.Row():
        with gr.Accordion(open=True, label="Animation Options"):
            with gr.Row():
                flag_relative_input = gr.Checkbox(value=True, label="Relative Motion")
                flag_do_crop_input = gr.Checkbox(value=True, label="Do Crop")
                flag_remap_input = gr.Checkbox(value=True, label="Paste-back")
                flag_eye_lip_open_enabled= gr.Checkbox(value=False, label="Target Eye Lip Open Ratio")
            with gr.Row():
                target_eye_ratio_input = gr.Slider(minimum=0, maximum=0.8, step=0.05, value=0.3, label="target eyes-open ratio")
                target_lip_ratio_input = gr.Slider(minimum=0, maximum=0.8, step=0.05, value=0.3, label="target lip-open ratio")
    with gr.Row():
        with gr.Column():
            process_button_animation = gr.Button("🚀 Animate", variant="primary")
        with gr.Column():
            process_button_reset = gr.ClearButton([image_input, video_input, output_video, output_video_concat], value="🧹 Clear")
    with gr.Row():
        with gr.Column():
            with gr.Accordion(open=True, label="The animated video in the original image space"):
                output_video.render()
        with gr.Column():
            with gr.Accordion(open=True, label="The animated video"):
                output_video_concat.render()
                btn_open_outputs = gr.Button("Open Outputs Folder")
                btn_open_outputs.click(fn=open_folder)
    with gr.Row():
        # Examples
        gr.Markdown("## You could choose the examples below ⬇️")
    with gr.Row():
        gr.Examples(
            examples=data_examples,
            inputs=[
                image_input,
                video_input,
                flag_relative_input,
                flag_do_crop_input,
                flag_remap_input,
                flag_eye_lip_open_enabled
            ],
            examples_per_page=5
        )
    gr.Markdown(load_description("assets/gradio_description_retargeting.md"))
    with gr.Row():
        eye_retargeting_slider.render()
        lip_retargeting_slider.render()
    with gr.Row():
        process_button_retargeting = gr.Button("🚗 Retargeting", variant="primary")
        process_button_reset_retargeting = gr.ClearButton(
            [
                eye_retargeting_slider,
                lip_retargeting_slider,
                retargeting_input_image,
                output_image,
                output_image_paste_back
            ],
            value="🧹 Clear"
        )
    with gr.Row():
        with gr.Column():
            with gr.Accordion(open=True, label="Retargeting Input"):
                retargeting_input_image.render()
        with gr.Column():
            with gr.Accordion(open=True, label="Retargeting Result"):
                output_image.render()
        with gr.Column():
            with gr.Accordion(open=True, label="Paste-back Result"):
                output_image_paste_back.render()
    # binding functions for buttons
    process_button_retargeting.click(
        fn=gradio_pipeline.execute_image,
        inputs=[eye_retargeting_slider, lip_retargeting_slider],
        outputs=[output_image, output_image_paste_back],
        show_progress=True
    )
    process_button_animation.click(
        fn=gradio_pipeline.execute_video,
        inputs=[
            image_input,
            video_input,
            flag_relative_input,
            flag_do_crop_input,
            flag_remap_input,
            target_eye_ratio_input,
            target_lip_ratio_input,
            flag_eye_lip_open_enabled
        ],
        outputs=[output_video, output_video_concat],
        show_progress=True
    )
    image_input.change(
        fn=gradio_pipeline.prepare_retargeting,
        inputs=image_input,
        outputs=[eye_retargeting_slider, lip_retargeting_slider, retargeting_input_image]
    )

##########################################################

demo.launch(
    share=cmd_args.share,
	inbrowser=True
)
