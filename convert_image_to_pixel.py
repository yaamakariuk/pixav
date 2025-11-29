"""
Convert an image to a single pixel-art frame.

Dependencies:
  pip install pillow numpy
  pip install mediapipe  # Optional, for --fill face

Usage:
  python convert_image_to_pixel.py --input input.png --out output --size 64 64 --scale 8 --palette photo --fill face --outline 30

Examples:
  # Basic conversion with default palette and face detection:
  python convert_image_to_pixel.py --input input.png --out results --size 64 96 --scale 6 --palette natural --fill face

  # With dithering, outline, and custom adjustments:
  python convert_image_to_pixel.py --input input.png --out pixel_art --size 128 128 --scale 4 --palette photo --dither --outline 40 --contrast 1.5 --saturation 1.3 --sharpness 1.4 --blur 0.5

  # Contain mode with custom background color:
  python convert_image_to_pixel.py --input input.jpg --out output --size 100 100 --scale 5 --fill contain --bg-color 255 255 255 255 --palette undertale
"""
import os
import sys
import argparse
import warnings
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from typing import Tuple, List

# Disable decompression bomb warnings
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

try:
    import mediapipe.python.solutions.face_mesh as face_mesh
    print("Mediapipe imported successfully.")
except ImportError as e:
    print(f"Failed to import Mediapipe: {e}. Face detection disabled.")
    face_mesh = None

# Palettes as a dictionary for easy extension
PALETTES = {
    "undertale": [
        (18, 7, 36), (58, 10, 85), (121, 45, 162), (196, 82, 255),
        (255, 185, 0), (255, 120, 120), (128, 80, 180), (60, 40, 70),
        (240, 200, 180), (200, 150, 120), (120, 80, 60), (0, 0, 0), (255, 255, 255)
    ],
    "natural": [
        (15, 15, 20), (60, 56, 54), (110, 90, 80), (185, 140, 120),
        (240, 200, 180), (255, 230, 210), (100, 70, 60), (40, 30, 30),
        (200, 160, 140), (255, 200, 200), (0, 0, 0), (255, 255, 255)
    ],
    "photo": [
        (90, 110, 80), (117, 137, 108), (141, 160, 131),
        (20, 50, 90), (49, 81, 122), (63, 100, 150),
        (100, 65, 45), (150, 115, 90), (200, 163, 135), (240, 206, 185),
        (145, 110, 68), (168, 131, 88),
        (0, 0, 0), (25, 25, 25), (220, 220, 220), (255, 255, 255)
    ]
}


def make_palette_image(palette: List[Tuple[int, int, int]]) -> Image.Image:
    """Create a palette image for quantization."""
    pal_img = Image.new("P", (16, 16))
    pal_data = []
    for r, g, b in palette:
        pal_data.extend((r, g, b))
    pal_data.extend((0, 0, 0) * (256 - len(palette)))  # Pad with black
    pal_img.putpalette(pal_data)
    return pal_img


def prepare_base_aspect(img_path: str, target_size: Tuple[int, int],
                        fill_mode: str = "cover", pad_top_frac: float = 0.18,
                        bg_color: Tuple[int, int, int, int] = (0, 0, 0, 0)) -> Image.Image:
    """Prepare image to fit target size and aspect ratio."""
    img = Image.open(img_path).convert("RGBA")
    w0, h0 = img.size
    target_w, target_h = target_size
    target_ratio = target_w / target_h
    src_ratio = w0 / h0

    if fill_mode == "contain":
        if src_ratio > target_ratio:
            new_w, new_h = target_w, round(target_w / src_ratio)
        else:
            new_w, new_h = round(target_h * src_ratio), target_h
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (target_w, target_h), bg_color)
        canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
        return canvas

    if fill_mode == "face" and face_mesh:
        try:
            rgb_image = img.convert("RGB")
            arr = np.array(rgb_image)
            with face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            ) as fm:
                res = fm.process(arr)
                multi_face_landmarks = getattr(res, 'multi_face_landmarks', None)  # type: ignore
                if multi_face_landmarks:
                    lms = multi_face_landmarks[0].landmark
                    xs = [lm.x * w0 for lm in lms]
                    ys = [lm.y * h0 for lm in lms]
                    fx1, fx2, fy1, fy2 = min(xs), max(xs), min(ys), max(ys)

                    crop_h = min(h0, int(w0 / target_ratio))
                    pad_top = int((fy2 - fy1) * pad_top_frac)
                    crop_top = max(0, int(fy1) - pad_top)
                    if crop_top + crop_h > h0:
                        crop_top = h0 - crop_h
                    crop_w = int(crop_h * target_ratio)
                    face_cx = (fx1 + fx2) / 2
                    crop_left = max(0, min(int(face_cx - crop_w / 2), w0 - crop_w))

                    cropped = img.crop((crop_left, crop_top, crop_left + crop_w, crop_top + crop_h))
                    return cropped.resize(target_size, Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Face detection failed: {e}. Falling back to 'cover'.")
    elif fill_mode == "face":
        print("Mediapipe not available. Falling back to 'cover'. Install with: pip install mediapipe")

    # Default to "cover"
    if src_ratio > target_ratio:
        crop_w, crop_h = int(round(h0 * target_ratio)), h0
    else:
        crop_w, crop_h = w0, int(round(w0 / target_ratio))
    crop_left = (w0 - crop_w) // 2
    crop_top = (h0 - crop_h) // 2
    cropped = img.crop((crop_left, crop_top, crop_left + crop_w, crop_top + crop_h))
    return cropped.resize(target_size, Image.Resampling.LANCZOS)


def adjust_characteristics(img: Image.Image, contrast: float = 1.0, saturation: float = 1.0,
                           sharpness: float = 1.0, blur: float = 0.0) -> Image.Image:
    """Adjust contrast, saturation, sharpness, and optional blur."""
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(saturation)
    if sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return img


def add_outline(img: Image.Image, sensitivity: int = 0, outline_color: Tuple[int, int, int, int] = (0, 0, 0, 255)) -> Image.Image:
    """Add black outline based on edges. Sensitivity: 0=off, 100=max details."""
    if sensitivity <= 0:
        return img
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    # Linear threshold: 0 -> 255 (no edges), 100 -> 50 (strong edges)
    threshold = int(255 - (sensitivity / 100 * 205))  # 255 to 50
    lut = [255 if i > threshold else 0 for i in range(256)]
    mask = edges.point(lut)
    outline_layer = Image.new("RGBA", img.size, outline_color)
    outlined = img.copy()
    outlined.paste(outline_layer, (0, 0), mask=mask)
    return outlined


def quantize_with_palette(img: Image.Image, palette: List[Tuple[int, int, int]], dither: bool = True) -> Image.Image:
    """Quantize image to the given palette."""
    palette_img = make_palette_image(palette)
    rgb_img = img.convert("RGB")
    dither_method = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    quantized = rgb_img.quantize(palette=palette_img, dither=dither_method)
    return quantized.convert("RGBA")


def upscale_nearest(img: Image.Image, scale: int) -> Image.Image:
    """Upscale using nearest neighbor to preserve pixels."""
    return img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)


def save_pixel_art(image: Image.Image, out_dir: str, basename: str, scale: int) -> str:
    """Save the final pixel art image."""
    os.makedirs(out_dir, exist_ok=True)
    upscaled = upscale_nearest(image, scale)
    output_path = os.path.join(out_dir, f"{basename}_pixel.png")
    upscaled.save(output_path, optimize=False)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert a photo to pixel-art.")
    parser.add_argument("--input", "-i", required=True, help="Input photo path.")
    parser.add_argument("--out", "-o", default="output", help="Output directory.")
    parser.add_argument("--size", nargs=2, type=int, default=[64, 64], metavar=("WIDTH", "HEIGHT"), help="Target size (e.g., 64 64).")
    parser.add_argument("--scale", type=int, default=6, help="Upscale factor for final PNG.")
    parser.add_argument("--palette", choices=list(PALETTES.keys()), default="undertale", help="Color palette.")
    parser.add_argument("--dither", action="store_true", help="Enable dithering.")
    parser.add_argument("--fill", choices=["cover", "contain", "face"], default="face", help="Fit mode.")
    parser.add_argument("--bg-color", nargs=4, type=int, default=[0, 0, 0, 0], metavar=("R", "G", "B", "A"), help="BG color for 'contain' (RGBA).")
    parser.add_argument("--contrast", type=float, default=1.2, help="Contrast (1.0=original).")
    parser.add_argument("--saturation", type=float, default=1.2, help="Saturation (1.0=original).")
    parser.add_argument("--sharpness", type=float, default=1.3, help="Sharpness (1.0=original).")
    parser.add_argument("--blur", type=float, default=0.0, help="Gaussian blur radius (0=off).")
    parser.add_argument("--outline", type=int, default=0, help="Outline sensitivity (0-100).")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if min(args.size) < 1:
        print("Error: Size must be positive integers.", file=sys.stderr)
        sys.exit(1)

    palette = PALETTES.get(args.palette.lower(), PALETTES["undertale"])
    target_size = tuple(args.size)
    bg_color = tuple(args.bg_color)

    # Extract basename from input
    basename = os.path.splitext(os.path.basename(args.input))[0]

    # Step 1: Prepare base image
    base_image = prepare_base_aspect(args.input, target_size, fill_mode=args.fill, bg_color=bg_color)

    # Step 2: Adjust characteristics
    base_image = adjust_characteristics(
        base_image,
        contrast=args.contrast,
        saturation=args.saturation,
        sharpness=args.sharpness,
        blur=args.blur
    )

    # Step 3: Add outline
    base_image = add_outline(base_image, sensitivity=args.outline)

    # Step 4: Quantize to palette
    pixel_art_image = quantize_with_palette(base_image, palette, dither=args.dither)

    # Step 5: Save
    output_path = save_pixel_art(pixel_art_image, args.out, basename, args.scale)
    print(f"Pixel art saved to: {output_path}")


if __name__ == "__main__":
    main()
