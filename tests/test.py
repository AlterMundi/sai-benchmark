from PIL import Image

# Open the image
input_image = "camera-1284459_1920.jpg"  # Replace with your image path
img = Image.open(input_image)

# Get original dimensions
width, height = img.size

# Calculate new dimensions (2x)
new_width = width * 2
new_height = height * 2

# Resize image
resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

# Save the resized image
output_image = "camera-1284459_3840.jpg"
resized_img.save(output_image, quality=95)

print(f"Image resized and saved as {output_image}")