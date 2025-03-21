from flask import Flask, render_template, jsonify
import torch
import os
from torchvision.utils import save_image
import torch.nn as nn
import time

app = Flask(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters (match your training setup)
z_dim = 128

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

# Load the pre-trained generator model
gen = Generator(z_dim=z_dim, im_chan=3).to(device)
gen.load_state_dict(torch.load('generator_final.pth', map_location=device, weights_only=True))
gen.eval()

# Directory to save generated images
STATIC_DIR = 'static'
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

def generate_image():
    print("Generating new image...")
    with torch.no_grad():
        fake_noise = torch.randn(1, z_dim, 1, 1, device=device)
        fake = gen(fake_noise)
        fake = (fake + 1) / 2
    
    timestamp = int(time.time())
    image_filename = f'generated_image_{timestamp}.png'
    image_path = os.path.join(STATIC_DIR, image_filename)
    print(f"Saving image to {image_path}")
    save_image(fake, image_path)
    return f'/static/{image_filename}'

@app.route('/')
def index():
    image_url = generate_image()
    return render_template('index.html', image_url=image_url)

@app.route('/generate', methods=['GET'])
def generate():
    print("Received request to /generate")
    image_url = generate_image()
    print(f"Returning image URL: {image_url}")
    response = jsonify({'image_url': image_url})
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use PORT env var if available
    app.run(debug=False, host='0.0.0.0', port=port)