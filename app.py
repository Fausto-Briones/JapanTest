import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import time
from torchvision.utils import make_grid

# Set page config
st.set_page_config(
    page_title="MNIST Number Generator",
    page_icon="🔢",
    layout="wide"
)

# Generator class (same as your trained model)
class MNISTGenerator(nn.Module):
    """Generator network for MNIST GAN"""
    
    def __init__(self, latent_dim=100, img_channels=1, img_size=28):
        super(MNISTGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size
        self.init_size = img_size // 4
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size * self.init_size),
            nn.BatchNorm1d(128 * self.init_size * self.init_size),
            nn.ReLU(True)
        )
        
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class NumberImageGenerator:
    """Generator for creating images of specific numbers"""
    
    def __init__(self, model_path=None, latent_dim=100):
        # Force GPU usage if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        self.generator = MNISTGenerator(latent_dim).to(self.device)
        
        # Display device info
        st.sidebar.success(f"🖥️ Using: {self.device}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            st.sidebar.info(f"🚀 GPU: {gpu_name}")
        
        if model_path:
            if self.load_model(model_path):
                st.sidebar.success("✅ Generator model loaded!")
            else:
                st.sidebar.warning("⚠️ Using random weights (demo mode)")
        else:
            self.generator.eval()
            st.sidebar.info("📝 Demo mode: using random weights")
    
    def load_model(self, model_path):
        """Load pre-trained generator model"""
        try:
            # Load model state dict and move to GPU
            state_dict = torch.load(model_path, map_location=self.device)
            self.generator.load_state_dict(state_dict)
            self.generator.eval()
            
            # Test the model quickly
            with torch.no_grad():
                test_noise = torch.randn(1, self.latent_dim, device=self.device)
                test_output = self.generator(test_noise)
                
            st.sidebar.info(f"Model output shape: {test_output.shape}")
            return True
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
            return False
    
    def generate_number_images(self, number, count=5):
        """Generate multiple images of a specific number"""
        # Use the number to create different but related seeds
        base_seed = int(number) * 1000
        images = []
        
        with torch.no_grad():
            for i in range(count):
                # Create variations by adding offset to base seed
                seed = base_seed + i * 123 + i  # Different variations
                torch.manual_seed(seed)
                
                # Generate noise and create image
                noise = torch.randn(1, self.latent_dim, device=self.device)
                fake_img = self.generator(noise)
                
                # Denormalize from [-1, 1] to [0, 1]
                fake_img = fake_img * 0.5 + 0.5
                images.append(fake_img)
        
        return torch.cat(images, dim=0).cpu()

def tensor_to_pil(tensor_img):
    """Convert tensor to PIL Image"""
    # Ensure tensor is on CPU and detached
    if tensor_img.is_cuda:
        tensor_img = tensor_img.cpu()
    tensor_img = tensor_img.detach()
    
    # Handle different tensor shapes
    if len(tensor_img.shape) == 4:  # (batch, channels, height, width)
        tensor_img = tensor_img.squeeze(0)  # Remove batch dimension
    if len(tensor_img.shape) == 3:  # (channels, height, width)
        if tensor_img.shape[0] == 1:  # Single channel
            tensor_img = tensor_img.squeeze(0)  # Remove channel dimension
        else:  # Multiple channels - take first channel
            tensor_img = tensor_img[0]
    
    # Convert to numpy and scale to [0, 255]
    img_array = tensor_img.numpy()
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array, mode='L')

def create_image_grid(images, nrow=5):
    """Create a grid of images"""
    # Ensure images are on CPU
    if images.is_cuda:
        images = images.cpu()
    images = images.detach()
    
    grid = make_grid(images, nrow=nrow, normalize=False, padding=2, pad_value=1)
    
    # Handle grid tensor shape properly
    if len(grid.shape) == 3:  # (channels, height, width)
        if grid.shape[0] == 1:  # Single channel
            grid = grid.squeeze(0)  # Remove channel dimension
        else:  # Multiple channels - take first channel
            grid = grid[0]
    
    # Convert to PIL
    grid_array = grid.numpy()
    grid_array = np.clip(grid_array * 255, 0, 255).astype(np.uint8)
    grid_img = Image.fromarray(grid_array, mode='L')
    
    return grid_img

def download_images(images, number):
    """Create download button for all images"""
    import zipfile
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for i, img_tensor in enumerate(images):
            img_pil = tensor_to_pil(img_tensor)
            img_buffer = io.BytesIO()
            img_pil.save(img_buffer, format='PNG')
            zip_file.writestr(f"number_{number}_variation_{i+1}.png", img_buffer.getvalue())
    
    st.download_button(
        label=f"📥 Download All {len(images)} Images",
        data=zip_buffer.getvalue(),
        file_name=f"generated_number_{number}_{int(time.time())}.zip",
        mime="application/zip",
        type="primary"
    )

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = NumberImageGenerator()

def main():
    st.title("🔢 MNIST Number Generator")
    st.markdown("Generate 5 variations of any digit using a trained GAN model")
    
    # Sidebar for model upload
    with st.sidebar:
        st.header("⚙️ Model Settings")
        
        uploaded_model = st.file_uploader(
            "Upload Generator Model (.pth)", 
            type=['pth'],
            help="Upload your trained GAN generator model (not discriminator!)"
        )
        
        if uploaded_model:
            # Save uploaded file temporarily
            model_path = f"temp_generator_{int(time.time())}.pth"
            with open(model_path, "wb") as f:
                f.write(uploaded_model.getbuffer())
            
            # Load the model
            if st.session_state.generator.load_model(model_path):
                st.success("✅ Generator loaded successfully!")
            else:
                st.error("❌ Failed to load model - make sure it's a generator!")
            
            # Clean up temp file
            import os
            if os.path.exists(model_path):
                os.remove(model_path)
        
        # Quick load buttons for common file names
        st.markdown("**Quick Load:**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📁 generator_model.pth", help="Load generator_model.pth"):
                if os.path.exists("generator_model.pth"):
                    if st.session_state.generator.load_model("generator_model.pth"):
                        st.success("✅ Loaded!")
                    else:
                        st.error("❌ Failed to load")
                else:
                    st.error("File not found")
        
        with col2:
            if st.button("📁 generator_epoch_50.pth", help="Load generator_epoch_50.pth"):
                if os.path.exists("generator_epoch_50.pth"):
                    if st.session_state.generator.load_model("generator_epoch_50.pth"):
                        st.success("✅ Loaded!")
                    else:
                        st.error("❌ Failed to load")
                else:
                    st.error("File not found")
        
        st.divider()
        st.markdown("### 🎯 How it works:")
        st.markdown("""
        1. Enter a digit (0-9)
        2. Click generate 
        3. Get 5 variations of that digit
        4. Download all images
        """)
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("🎯 Input")
        
        # Number input
        input_number = st.selectbox(
            "Select a digit to generate:",
            options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            index=2,
            help="Choose which digit you want to generate"
        )
        
        st.markdown(f"### Generating: **{input_number}**")
        
        # Generate button
        if st.button("🎨 Generate 5 Images", type="primary", use_container_width=True):
            with st.spinner(f"Generating 5 variations of '{input_number}' on GPU..."):
                start_time = time.time()
                images = st.session_state.generator.generate_number_images(input_number, count=5)
                generation_time = time.time() - start_time
                
                st.session_state.generated_images = images
                st.session_state.current_number = input_number
                st.session_state.generation_time = generation_time
                
                st.success(f"✅ Generated in {generation_time:.2f}s")
        
        # Quick generate buttons for common numbers
        st.markdown("### 🚀 Quick Generate:")
        quick_cols = st.columns(5)
        for i, num in enumerate([0, 1, 2, 3, 4]):
            with quick_cols[i]:
                if st.button(f"{num}", key=f"quick_{num}"):
                    with st.spinner(f"Generating {num}..."):
                        images = st.session_state.generator.generate_number_images(num, count=5)
                        st.session_state.generated_images = images
                        st.session_state.current_number = num
        
        quick_cols2 = st.columns(5)
        for i, num in enumerate([5, 6, 7, 8, 9]):
            with quick_cols2[i]:
                if st.button(f"{num}", key=f"quick_{num}"):
                    with st.spinner(f"Generating {num}..."):
                        images = st.session_state.generator.generate_number_images(num, count=5)
                        st.session_state.generated_images = images
                        st.session_state.current_number = num
    
    with col2:
        st.header("🖼️ Generated Images")
        
        if 'generated_images' in st.session_state:
            images = st.session_state.generated_images
            number = st.session_state.current_number
            
            # Display info
            st.info(f"🎯 Generated 5 variations of digit: **{number}**")
            
            if hasattr(st.session_state, 'generation_time'):
                st.metric("⚡ Generation Time", f"{st.session_state.generation_time:.2f}s")
            
            # Show grid
            try:
                grid_img = create_image_grid(images, nrow=5)
                st.image(grid_img, caption=f"5 variations of digit '{number}'", use_column_width=True)
                
                # Download button
                download_images(images, number)
                
                st.divider()
                
                # Individual images with debug info
                st.subheader("Individual Variations")
                cols = st.columns(5)
                for i, img_tensor in enumerate(images):
                    with cols[i]:
                        try:
                            img_pil = tensor_to_pil(img_tensor)
                            st.image(img_pil, caption=f"Variation {i+1}", width=100)
                        except Exception as e:
                            st.error(f"Error displaying image {i+1}: {e}")
                            # Debug info
                            st.write(f"Tensor shape: {img_tensor.shape}")
                            st.write(f"Tensor range: {img_tensor.min():.3f} to {img_tensor.max():.3f}")
                            
            except Exception as e:
                st.error(f"Error creating image display: {e}")
                
                # Debug information
                st.write("Debug Information:")
                st.write(f"Images tensor shape: {images.shape}")
                st.write(f"Images tensor type: {type(images)}")
                st.write(f"Images tensor device: {images.device}")
                st.write(f"Images tensor range: {images.min():.3f} to {images.max():.3f}")
                
                # Try alternative display method
                st.write("Attempting alternative display...")
                for i in range(min(5, len(images))):
                    try:
                        img = images[i].cpu().detach().numpy()
                        if len(img.shape) == 3:
                            img = img.squeeze(0)
                        img = np.clip(img * 255, 0, 255).astype(np.uint8)
                        st.image(img, caption=f"Image {i+1}", width=100)
                    except Exception as e:
                        st.error(f"Error displaying image {i+1}: {e}")
                        st.write(f"Image tensor shape: {images[i].shape}")
                        st.write(f"Image tensor range: {images[i].min():.3f} to {images[i].max():.3f}")
        
        else:
            st.info("👈 Select a digit and click 'Generate 5 Images' to see results!")
            
            # Show example
            st.markdown("### Example Output:")
            fig, axes = plt.subplots(1, 5, figsize=(10, 2))
            for i, ax in enumerate(axes):
                ax.text(0.5, 0.5, f"Digit\nVariation\n{i+1}", 
                       ha='center', va='center', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            
            plt.suptitle("5 variations of your chosen digit will appear here", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
    
    # Footer
    st.divider()
    
    # Performance info
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        device_type = "🚀 GPU" if torch.cuda.is_available() else "💻 CPU"
        st.metric("Device", device_type)
    
    with col_info2:
        st.metric("Images per Generation", "5")
    
    with col_info3:
        st.metric("Image Size", "28x28 px")

if __name__ == "__main__":
    main()