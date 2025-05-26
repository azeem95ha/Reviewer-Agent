from PIL import Image, ImageOps

# Load the original image
original_path = 'page.jpg'
image = Image.open(original_path).convert("RGBA")

# Create a black background image with the same size
black_bg = Image.new("RGBA", image.size)

# Composite the original image onto the black background
new_image = Image.alpha_composite(black_bg, image)

# Save the result
output_path = 'gripx_logo_black_background.png'
new_image.save(output_path)
output_path
