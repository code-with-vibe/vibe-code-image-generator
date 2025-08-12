import gradio as gr
from gradio_client import Client
import os
from PIL import Image
import io
import base64
import requests

def generate_image(prompt, negative_prompt="", guidance_scale=9):
    """
    Generate an image using the Stable Diffusion API
    """
    if not prompt.strip():
        return None
    
    try:
        # Initialize the client
        client = Client("stabilityai/stable-diffusion")
        
        # Make the prediction
        result = client.predict(
            prompt=prompt,
            negative=negative_prompt,
            scale=guidance_scale,
            api_name="/infer"
        )
        
        print(f"Debug - Result type: {type(result)}")
        print(f"Debug - Result: {result}")
        
        # Handle the specific format: list of dictionaries with 'image' keys
        if isinstance(result, list):
            for i, item in enumerate(result):
                try:
                    if isinstance(item, dict) and 'image' in item:
                        # Extract the image path from the dictionary
                        image_path = item['image']
                        if os.path.exists(image_path):
                            return Image.open(image_path)
                    elif isinstance(item, str):
                        # If it's a file path, load it as PIL Image
                        if os.path.exists(item):
                            return Image.open(item)
                        # If it's a URL, download and return as PIL Image
                        elif item.startswith(('http://', 'https://')):
                            response = requests.get(item)
                            return Image.open(io.BytesIO(response.content))
                    elif hasattr(item, 'save'):  # PIL Image object
                        return item
                except Exception as e:
                    print(f"Debug - Error processing item {i}: {e}")
                    continue
            
            # If no image found, try first item as fallback
            if len(result) > 0:
                first_item = result[0]
                if isinstance(first_item, dict) and 'image' in first_item:
                    image_path = first_item['image']
                    if os.path.exists(image_path):
                        return Image.open(image_path)
                elif isinstance(first_item, str) and os.path.exists(first_item):
                    return Image.open(first_item)
        
        elif isinstance(result, dict) and 'image' in result:
            # Single dictionary result
            image_path = result['image']
            if os.path.exists(image_path):
                return Image.open(image_path)
        
        elif isinstance(result, str):
            # Single string result - could be file path or URL
            if os.path.exists(result):
                return Image.open(result)
            elif result.startswith(('http://', 'https://')):
                response = requests.get(result)
                return Image.open(io.BytesIO(response.content))
        
        elif hasattr(result, 'save'):
            # PIL Image object
            return result
        
        # If nothing worked, return None
        print("Debug - No valid image found in result")
        return None
        
    except Exception as e:
        print(f"Debug - Full error: {e}")
        return None

def create_interface():
    """
    Create and configure the Gradio interface
    """
    with gr.Blocks(
        title="🎨 VibeCode Image Generator",
        theme=gr.themes.Soft(),
        css="""
        .main-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        footer {
            visibility: hidden;
        }
        .generate-btn {
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 8px;
            color: white;
            font-weight: bold;
            padding: 12px 24px;
            transition: transform 0.2s ease;
        }
        .generate-btn:hover {
            transform: translateY(-2px);
        }
        """
    ) as demo:
        
        # Header
        gr.Markdown(
            """
            <div class="main-header">
                <h1>🎨 VibeCode Image Generator</h1>
                <p>Create stunning AI-generated images from text descriptions</p>
            </div>
            """,
            elem_classes=["main-header"]
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                gr.Markdown("### 📝 Generation Settings")
                
                prompt_input = gr.Textbox(
                    label="✨ Prompt",
                    placeholder="Describe the image you want to generate... (e.g., 'A serene landscape with mountains and a lake at sunset')",
                    lines=3,
                    value=""
                )
                
                negative_prompt_input = gr.Textbox(
                    label="🚫 Negative Prompt",
                    placeholder="What you DON'T want in the image... (e.g., 'blurry, low quality, distorted')",
                    lines=2,
                    value="blurry, low quality, distorted, ugly, duplicate"
                )
                
                guidance_scale_input = gr.Slider(
                    label="🎛️ Guidance Scale",
                    minimum=1,
                    maximum=20,
                    value=9,
                    step=0.5,
                    info="How closely the model follows your prompt (higher = more strict)"
                )
                
                generate_btn = gr.Button(
                    "🎨 Generate Image",
                    variant="primary",
                    size="lg",
                    elem_classes=["generate-btn"]
                )
                
                # Examples
                gr.Markdown("### 💡 Example Prompts")
                gr.Examples(
                    examples=[
                        ["A magical forest with glowing mushrooms and fireflies, fantasy art style", "blurry, low quality", 9],
                        ["A futuristic cityscape at night with neon lights, cyberpunk style", "daylight, vintage", 12],
                        ["A cute robot pet sitting in a garden, digital art", "scary, dark, realistic", 8],
                        ["An astronaut riding a horse on Mars, cinematic lighting", "cartoon, low resolution", 10],
                        ["A steampunk airship flying through clouds, detailed illustration", "modern, simple", 11]
                    ],
                    inputs=[prompt_input, negative_prompt_input, guidance_scale_input],
                    label="Click an example to try it out!"
                )
            
            with gr.Column(scale=1):
                # Output
                gr.Markdown("### 🖼️ Generated Image")
                
                output_image = gr.Image(
                    label="Result",
                    type="pil",
                    height=400,
                    show_label=False
                )
                
                # Status/Info
                gr.Markdown(
                    """
                    ### ℹ️ Tips for Better Results:
                    - **Be specific**: Include details about style, lighting, composition
                    - **Use negative prompts**: Exclude unwanted elements
                    - **Adjust guidance**: Higher values follow prompts more strictly
                    - **Try different scales**: 7-12 usually work well for most images
                    """
                )
        
        # Event handlers
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt_input, negative_prompt_input, guidance_scale_input],
            outputs=output_image,
            show_progress=True
        )
        
        # Allow Enter key to generate
        prompt_input.submit(
            fn=generate_image,
            inputs=[prompt_input, negative_prompt_input, guidance_scale_input],
            outputs=output_image,
            show_progress=True
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            <div style="text-align: center; color: #666; margin-top: 2rem;">
                <p>Powered by Vibe Code Org. | Built with ❤️ </p>
            </div>
            
            """
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",  # Important for Hugging Face Spaces
        server_port=7860,        # Default port for HF Spaces
        debug=False
    )