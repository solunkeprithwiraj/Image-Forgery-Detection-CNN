from PIL import Image, ImageDraw, ImageFont
import os

def create_placeholder_image():
    # Create a new image with white background
    width, height = 400, 300
    image = Image.new('RGB', (width, height), color=(255, 255, 255))
    
    # Get a drawing context
    draw = ImageDraw.Draw(image)
    
    # Draw a border
    border_color = (200, 200, 200)
    border_width = 2
    draw.rectangle(
        [(border_width, border_width), (width - border_width, height - border_width)],
        outline=border_color,
        width=border_width
    )
    
    # Draw a TIFF icon or symbol
    icon_color = (100, 100, 100)
    draw.rectangle([(width//2 - 50, height//2 - 60), (width//2 + 50, height//2 + 20)], fill=icon_color)
    draw.polygon([(width//2 - 30, height//2 + 20), (width//2 + 30, height//2 + 20), (width//2, height//2 + 50)], fill=icon_color)
    
    # Add text
    text_color = (50, 50, 50)
    # Try to use a system font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    
    text = "TIFF Image Preview"
    text_width = draw.textlength(text, font=font)
    draw.text(
        (width//2 - text_width//2, height//2 + 60),
        text,
        fill=text_color,
        font=font
    )
    
    subtext = "Converting for display..."
    subtext_width = draw.textlength(subtext, font=font)
    draw.text(
        (width//2 - subtext_width//2, height//2 + 90),
        subtext,
        fill=text_color,
        font=font
    )
    
    # Save the image
    output_path = os.path.join('static', 'placeholder-tiff.png')
    image.save(output_path)
    print(f"Placeholder image created at {output_path}")

if __name__ == "__main__":
    create_placeholder_image() 