from PIL import Image, ImageDraw, ImageFont
import textwrap

img = Image.new("RGB", (1280, 720), color=(50, 50, 100))
draw = ImageDraw.Draw(img)

try:
    font = ImageFont.truetype("Arial Bold.ttf", 36)
except:
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
    except:
        font = ImageFont.load_default()

text = "This is a very long text that needs to be wrapped properly on the screen so it looks like a subtitle."
lines = textwrap.wrap(text, width=60)
y = 720 - len(lines)*40 - 60
for line in lines:
    try:
        bbox = font.getbbox(line)
        w = bbox[2] - bbox[0]
    except:
        w = len(line)*20
    x = (1280 - w) // 2
    draw.text((x, y), line, font=font, fill="white")
    y += 40

img.save("test_out.png")
print("Done")
