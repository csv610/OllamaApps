import streamlit as st
import ollama
from PIL import Image
import time
import tempfile
import os
import base64
from io import BytesIO
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class ModelResponse:
    """Data class to store model response and execution metrics"""
    content: str
    execution_time: float

class ImageProcessor:
    """Utility class for image processing operations"""
    
    @staticmethod
    def resize_image(image: Image.Image, max_size) -> Image.Image:
        """
        Resize image while maintaining aspect ratio if either dimension exceeds max_size
        
        Args:
            image (PIL.Image): Input image
            max_size (Optional[int]): Maximum allowed dimension
            
        Returns:
            PIL.Image: Resized image
        """
        if not max_size:
            return image
            
        # Get original dimensions
        width, height = image.size
        
        # Return original image if both dimensions are already smaller than max_size
        if width <= max_size and height <= max_size:
            return image
            
        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
            
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    @staticmethod
    def to_base64(image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            str: Base64 encoded image string
        """
        buffered = BytesIO()
        image.save(buffered, format=image.format or 'JPEG')
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    @staticmethod
    def load_image(image_path: str) -> Image.Image:
        """
        Load image from file path
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            PIL.Image: Loaded image
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            PIL.UnidentifiedImageError: If image format is invalid
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            return Image.open(image_path)
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")

class LlamaVision:
    """A class to handle vision-language model interactions using Ollama"""
    
    MODELS: List[str] = ['llama3.2-vision', 'llava:7b-v1.6', 'llava-phi3', 'llava-llama3']
    
    def __init__(self, model_name: str):
        """
        Initialize LlamaVision with a specific model
        
        Args:
            model_name (str): Name of the model to use
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Model {model_name} not in available models: {self.MODELS}")
        self.model_name = model_name

    @staticmethod
    def get_available_models() -> List[str]:
        """Return list of available vision models"""
        return LlamaVision.MODELS

    def get_response(self, image_path: str, prompt: str) -> ModelResponse:
        """
        Get response from the vision model
        
        Args:
            image_path (str): Path to the image file
            prompt (str): Text prompt for the model
            
        Returns:
            ModelResponse: Model response and execution time
        """
        try:
            start_time = time.time()
            
            # Load and convert image
            image = ImageProcessor.load_image(image_path)
            base64_image = ImageProcessor.to_base64(image)
            
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [base64_image]
                }],
            )
            
            execution_time = time.time() - start_time
            cleaned_text = response.get('message', {}).get('content', '').strip()
            
            return ModelResponse(cleaned_text, execution_time)
            
        except Exception as e:
            st.error(f"Error processing request: {str(e)}")
            return ModelResponse("", 0.0)

def streamlit_app():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="Ollama Vision",
        page_icon="🔍",
        layout="wide"
    )
    
    # Application header
    st.title("🔍 Ollama Vision")
    st.markdown("Analyze images using various vision-language models")
    
    # Sidebar configurations
    with st.sidebar:
        st.header("Configuration")
        model_options = LlamaVision.get_available_models()
        selected_model = st.selectbox(
            "Select Vision Model",
            model_options,
            help="Choose the vision-language model to use for analysis"
        )
        
        fit_image = st.checkbox(
            "Fit image to container",
            value=True,
            help="Adjust image size to fit the container width"
        )
        
        # Add max size option
        max_size = st.number_input(
            "Maximum image dimension",
            min_value=None,
            max_value=None,
            value=1024,
            help="Maximum allowed dimension for image (maintains aspect ratio). Leave empty for original size."
        )
        
    # Main content area
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG, JPEG"
    )
    
    if uploaded_file:
        # Create temporary file for the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            image_path = temp_file.name
        
        try:
            # Load and process image
            image = ImageProcessor.load_image(image_path)
            
            image = ImageProcessor.resize_image(image, max_size)
            
            # Display image
            st.image(image, caption="Uploaded Image", use_container_width=fit_image)
            
            st.write(f"- **Resized Dimensions:** {image.size[0]} × {image.size[1]}")
            
            # Initialize LlamaVision
            llama_vision = LlamaVision(model_name=selected_model)
            
            prompt = st.text_input(
                "Ask Question",
                "Describe this image in detail.",
                help="Enter a prompt for the model to analyze the image"
            )
            
            if st.button("Analyze Image", type="primary"):
                with st.spinner(f"Analyzing image using {selected_model}..."):
                    response = llama_vision.get_response(image_path, prompt)
                
                if response.content:
                    st.markdown("### Model Response")
                    st.write(response.content)
                    st.info(f"⚡ Execution Time: {response.execution_time:.2f} seconds")
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
        
        finally:
            # Clean up temporary file
            os.remove(image_path)

if __name__ == "__main__":
    streamlit_app()
