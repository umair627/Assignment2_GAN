import torch
from torch import nn
from torchvision.utils import save_image
from flask import Flask, render_template
import os

# Define Generator class (same as in your training code)
class Generator(nn.Module):
    def __init__(self, z_dim=128, im_chan=3, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 8, kernel_size=4, stride=1, padding=0),
            self.make_gen_block(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, stride=2, padding=1, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, padding=0, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.Tanh(),
            )

    def forward(self, noise):
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)

# Function to generate noise
def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)

# Initialize Flask app
app = Flask(__name__)

# Load the saved generator model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
z_dim = 128  # Match your training config
gen = Generator(z_dim=z_dim, im_chan=3).to(device)
gen.load_state_dict(torch.load('generator_final.pth', map_location=device))
gen.eval()  # Set to evaluation mode

# Route for the homepage
@app.route('/')
def generate_image():
    # Generate a new image
    with torch.no_grad():
        noise = get_noise(n_samples=1, z_dim=z_dim, device=device)
        fake_image = gen(noise)
        # Denormalize from [-1, 1] to [0, 1] for display
        fake_image = (fake_image + 1) / 2
        # Save the image to the static folder
        image_path = os.path.join('static', 'generated_image.png')
        save_image(fake_image, image_path)
    
    # Render the page with the new image
    return render_template('index.html', image_url='/static/generated_image.png')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)