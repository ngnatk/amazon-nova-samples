#!/usr/bin/env python
# -*- coding: utf8 -*-

# Example usage:
# - streamlit run streamlit_nova_canvas.py
# - streamlit run streamlit_nova_canvas.py --server.runOnSave True --server.port 8501

import amazon_video_util
import base64
import boto3
import datetime
import file_utils
import json
import logging
import random
import streamlit as st

from amazon_image_gen import BedrockImageGenerator



# Setup Logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Setup AWS
bedrock_runtime = boto3.client('bedrock-runtime')
s3_client = boto3.client("s3")

# Setup Streamlit
st.set_page_config(
    page_title="Image Generation with Nova Canvas",
    layout="wide"
)
st.title("🌇 Image Generation with Nova Canvas 🎨")



def generate_image(prompt, negative_prompt, guidance_scale, num_inference_steps, seed=None, width=1280, height=720, num_images=1):
    """
    Generate images using Nova Canvas
    """

    if seed is None:
        seed = random.randint(0, 858993459)

    request_body = {
        "cfg_scale": guidance_scale,
        "steps": num_inference_steps,
    }
    inference_params = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt,
            "negativeText": negative_prompt,
        },
        "imageGenerationConfig": {
            "numberOfImages": num_images,  # Number of variations to generate. 1 to 5.
            "quality": "standard",  # Allowed values are "standard" and "premium"
            "width": width,  # See README for supported output resolutions
            "height": height,  # See README for supported output resolutions
            "cfgScale": 7.0,  # How closely the prompt will be followed
            "seed": seed
        },
    }

    # Define an output directory with a unique name.
    generation_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_directory = f"output/{generation_id}"

    # Create the generator.
    generator = BedrockImageGenerator(output_directory=output_directory)
    response = generator.generate_images(inference_params)

    if "images" in response:
        image = file_utils.save_base64_images(response["images"], output_directory, "image")
    return response, image



def generate_video(s3_destination_bucket, video_prompt, model_id="amazon.nova-reel-v1:0"):
    """
    Generate a video using the provided prompt.
    
    Args:
        s3_destination_bucket (str): The S3 bucket where the video will be stored
        video_prompt (str): Text prompt describing the desired video

    Adapted from: https://github.com/ngnatk/amazon-nova-samples/blob/main/multimodal-generation/video-generation/python/01_text_to_video_generation.py
    """

    # Create the S3 bucket
    s3_client.create_bucket(Bucket=s3_destination_bucket)

    model_input = {
        "taskType": "TEXT_VIDEO",
        "textToVideoParams": {
            "text": video_prompt,
        },
        "videoGenerationConfig": {
            "durationSeconds": 6,  # 6 is the only supported value currently
            "fps": 24,  # 24 is the only supported value currently
            "dimension": "1280x720",  # "1280x720" is the only supported value currently
            "seed": random.randint(
                0, 2147483648
            ),  # A random seed guarantees we'll get a different result each time this code runs
        },
    }

    try:
        # Start the asynchronous video generation job
        invocation = bedrock_runtime.start_async_invoke(
            modelId=model_id,
            modelInput=model_input,
            outputDataConfig={"s3OutputDataConfig": {"s3Uri": f"s3://{s3_destination_bucket}"}},
        )

        # Store the invocation ARN
        invocation_arn = invocation["invocationArn"]

        # Pretty print the response JSON
        logger.info("\nResponse:")
        logger.info(json.dumps(invocation, indent=2, default=str))

        # Save the invocation details for later reference
        amazon_video_util.save_invocation_info(invocation, model_input)

        return invocation_arn

    except Exception as e:
        logger.error(e)
        return None



def remove_background(source_image_base64, source_image_path=None):
    """
    Removes background from an image. Adapted from:
    https://github.com/aws-samples/amazon-nova-samples/blob/main/multimodal-generation/image-generation/python/07_background_removal.py
    """

    if source_image_path is not None:
        with open(source_image_path, "rb") as image_file:
            source_image_base64 = base64.b64encode(image_file.read()).decode("utf8")

    inference_params = {
        "taskType": "BACKGROUND_REMOVAL",
        "backgroundRemovalParams": {
            "image": source_image_base64,
        },
    }

    generation_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_directory = f"output/{generation_id}"
    generator = BedrockImageGenerator(output_directory=output_directory)
    response = generator.generate_images(inference_params)

    if "images" in response:
        image = file_utils.save_base64_images(response["images"], output_directory, "image")
    return response, image



def tab_canvas():
    st.sidebar.header("Model Parameters")

    prompt = st.text_area("Enter your image prompt:", height=100)

    # Model parameters
    negative_prompt = st.sidebar.text_area("Negative Prompt:", 
        value="blurry, bad anatomy, bad hands, cropped, worst quality")
    num_inference_steps = st.sidebar.slider("Number of Inference Steps", 
        min_value=1, max_value=100, value=50)
    guidance_scale = st.sidebar.slider("Guidance Scale", 
        min_value=1.0, max_value=20.0, value=7.5, step=0.5)
    seed = st.sidebar.number_input("Seed", min_value=0, value=42)
    width = st.sidebar.select_slider("Image Width", 
        options=[512, 768, 1024], value=512)
    height = st.sidebar.select_slider("Image Height", 
        options=[512, 768, 1024], value=512)

    if st.button("Generate Image"):
        if prompt:
            try:
                with st.spinner():
                    response, image = generate_image(prompt, negative_prompt, guidance_scale, num_inference_steps, seed, width, height)
                    
                    if 'images' in response:
                        # Decode and display the generated image
                        # image_data = base64.b64decode(response_body['images'][0])
                        # image = Image.open(io.BytesIO(image_data))
                        st.image(image, caption="Generated Image", use_container_width=True)
                    else:
                        st.error("No image was generated in the response")
                    
            except Exception as e:
                st.error(f"Error generating image: {str(e)}")
        else:
            st.warning("Please enter a prompt to generate an image")



def tab_background_removal():
    img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if img is not None:
        # Display the uploaded image
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Convert image to bytes, encode to base64
        img_bytes = img.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        # Remove background and display the image
        with st.spinner():
            response, response_image = remove_background(img_base64)
            st.image(response_image, caption="Processed Image", use_container_width=True)



def tab_reel():
    # img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    prompt = st.text_area("Enter your video prompt:", height=100)

    if st.button("Generate Video"):
        if prompt:
            try:
                invocation_arn = generate_video(prompt)
                st.success(f"Video generation job started: {invocation_arn}")
                st.write(json.dumps(response, indent=2, default=str))
            except Exception as e:
                st.error(f"Error generating video: {str(e)}")
        else:
            st.warning("Please enter a prompt to generate a video")


def main():
    tab1, tab2, tab3 = st.tabs(["Canvas", "Background Removal", "Reel"])

    with tab1:
        tab_canvas()

    with tab2:
        tab_background_removal()

    with tab3:
        tab_reel()



if __name__ == "__main__":
    main()
