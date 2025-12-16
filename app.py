import os
import sys
import torch
import numpy as np
from PIL import Image
import streamlit as st
from io import BytesIO
import zipfile
import logging
from typing import List, Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import with error handling
try:
    from inference.demo_interface import ModelManager
except ImportError as e:
    logger.error(f"Failed to import ModelManager: {e}")
    st.error("Failed to initialize the application. Please check the installation and dependencies.")
    st.stop()

# Initialize the model manager
@st.cache_resource
def load_model() -> ModelManager:
    """
    Load and initialize the model manager with available models.
    
    Returns:
        ModelManager: Initialized model manager with loaded models
    """
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Initialize model manager
    try:
        model_manager = ModelManager(device=device)
    except Exception as e:
        logger.error(f"Failed to initialize ModelManager: {e}")
        st.error("Failed to initialize the model manager. Please check the logs for details.")
        st.stop()
    
    # Define model paths (update these to your actual model paths)
    model_paths = {
        'cvae': os.path.join(os.path.dirname(__file__), "outputs", "cvae_20251031_211019", "best_model.pth"),
        # Uncomment and update these paths when you have the other models
        # 'cgan': os.path.join(os.path.dirname(__file__), "models", "cgan", "best_model.pth"),
        # 'transformer': os.path.join(os.path.dirname(__file__), "models", "transformer", "best_model.pth"),
    }
    
    # Load models
    loaded_models = []
    for model_type, path in model_paths.items():
        if os.path.exists(path):
            success = model_manager.load_model(model_type, path)
            if success:
                loaded_models.append(model_type)
            else:
                logger.warning(f"Failed to load {model_type} model from {path}")
    
    if not loaded_models:
        st.error("No models were successfully loaded. Please check that the model files exist and are accessible.")
        st.stop()
    
    logger.info(f"Successfully loaded models: {', '.join(loaded_models)}")
    return model_manager

def main():
    """Main function to run the Streamlit app."""
    # Configure page
    st.set_page_config(
        page_title="Arabic Handwriting Generation",
        page_icon="✍️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""<style>.main .block-container {padding-top: 2rem;padding-bottom: 2rem;}.stButton>button {width: 100%;padding: 0.5rem;font-weight: bold;}.stTextArea textarea {min-height: 150px;}</style>""", unsafe_allow_html=True)
    
    # Header
    st.title("✍️ Arabic Handwriting Generation")
    st.markdown("Generate handwritten Arabic text using deep learning models.")
    st.markdown("---")
    
    try:
        # Load model
        with st.spinner("Loading models..."):
            model_manager = load_model()
            
            # Check if any models were loaded
            if not hasattr(model_manager, 'models') or not model_manager.models:
                st.error("No models were successfully loaded. Please check the logs for errors.")
                return
        
        # Sidebar
        with st.sidebar:
            st.title("⚙️ Model Settings")
            
            # Model selection
            model_type = st.selectbox(
                "Select Model",
                list(model_manager.models.keys())  # Only show loaded models
            )
            
            # Generation parameters
            st.subheader("Generation Settings")
            num_samples = st.slider(
                "Number of Samples", 
                min_value=1, 
                max_value=8, 
                value=4, 
                step=1,
                help="Number of different handwriting samples to generate"
            )
            
            st.markdown("---")
            st.markdown("### About")
            st.markdown("""This application generates handwritten Arabic text using deep learning models.
            
            **Available Models:**
            - CVAE: Conditional Variational Autoencoder
            - CGAN: Conditional GAN (if loaded)
            - Transformer: Transformer-based model (if loaded)
            """)
        
        # Main content
        st.subheader("Text Input")
        text = st.text_area(
            "Enter Arabic Text", 
            value="اكتب النص العربي هنا",
            help="Type or paste your Arabic text here"
        )
        
        # Generate button
        col1, col2 = st.columns([1, 3])
        with col1:
            generate_btn = st.button(
                "✨ Generate Handwriting",
                type="primary",
                use_container_width=True
            )
        
        if generate_btn:
            if not text.strip():
                st.warning("Please enter some text to generate.")
            else:
                with st.spinner("Generating handwriting..."):
                    try:
                        # Generate images
                        images = model_manager.generate_handwriting(
                            text=text,
                            model_type=model_type,
                            num_samples=num_samples
                        )
                        
                        # Display success message
                        st.success(f"Successfully generated {len(images)} samples!")
                        
                        # Display images in a grid
                        st.subheader("Generated Samples")
                        cols = st.columns(2)
                        for i, img in enumerate(images):
                            with cols[i % 2]:
                                st.image(
                                    img, 
                                    caption=f"Sample {i+1} - {model_type.upper()}", 
                                    use_column_width=True
                                )
                        
                        # Add download button
                        if len(images) > 0:
                            st.download_button(
                                label="💾 Download All Samples",
                                data=_images_to_zip(images),
                                file_name=f"arabic_handwriting_{model_type}.zip",
                                mime="application/zip",
                                help="Download all generated samples as a ZIP file"
                            )
                            
                    
                    except Exception as e:
                        error_msg = f"Error generating handwriting: {str(e)}"
                        logger.error(error_msg, exc_info=True)
                        st.error(error_msg)
    
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        st.error("Please check the logs for more details.")
    
    with col2:
        st.subheader("Instructions")
        st.markdown("""1. Enter your Arabic text in the input box
        2. Select a model from the sidebar
        3. Choose number of samples
        4. Click 'Generate Handwriting'
        
        **Tips:**
        - Use proper Arabic script
        - Keep text concise for best results
        - Try different models for variations
        """)
        
        # Add model information
        st.subheader("Model Info")
        if model_type in model_manager.models:
            config = model_manager.models[model_type]['config']
            st.json({
                "Model Type": model_type.upper(),
                "Input Size": f"{config.get('image_size', 'N/A')}x{config.get('image_size', 'N/A')}",
                "Parameters": f"{sum(p.numel() for p in model_manager.models[model_type]['model'].parameters()):,}",
                "Trained On": config.get('dataset', 'N/A')
            })

def _images_to_zip(images: List[np.ndarray]) -> bytes:
    """
    Convert a list of images to a zip file in memory.
    
    Args:
        images: List of numpy arrays representing images (values in [0, 1])
        
    Returns:
        bytes: ZIP file as bytes
    """
    try:
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, 'w') as zip_file:
            for i, img in enumerate(images):
                try:
                    # Ensure image is in the correct format
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    
                    # Convert to PIL Image and save to bytes
                    img_pil = Image.fromarray(img)
                    img_bytes = BytesIO()
                    img_pil.save(img_bytes, format='PNG')
                    
                    # Add to zip
                    zip_file.writestr(f'sample_{i+1:02d}.png', img_bytes.getvalue())
                    
                except Exception as img_error:
                    logger.error(f"Error processing image {i}: {img_error}")
                    continue
        
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Error creating ZIP file: {e}")
        raise

if __name__ == "__main__":
    main()
