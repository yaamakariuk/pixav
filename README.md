# Pixel Art Converter

This script converts photos into pixel art using predefined palettes, edge outlining, and optional face detection for cropping.

## Features
- Resize and crop images to fit target dimensions with modes like "cover", "contain", or "face" (using MediaPipe).
- Adjust image characteristics: contrast, saturation, sharpness, and blur.
- Apply color palettes (Undertale, Natural, Photo).
- Add outlines for a more defined pixel art look.
- Quantize colors with optional dithering.
- Upscale the final image for better visibility.

## Installation
1. Clone the repository:  https://github.com/yaamakariuk/pixav.git
2. Install dependencies: pip install -r requirements.txt
3. (Optional) For face detection: pip install mediapipe

## Usage
Run the script with the following command: 
python convert_image_to_pixel.py --input input.jpg --out output --size 64 64 --scale 8 --palette photo --fill face --outline 30

### Arguments
- `--input`: Path to the input photo (required).
- `--out`: Output directory (default: "output").
- `--size`: Target width and height (default: 64 64).
- `--scale`: Upscale factor (default: 6).
- `--palette`: Color palette ("undertale", "natural", "photo"; default: "undertale").
- `--dither`: Enable dithering (flag).
- `--fill`: Fit mode ("cover", "contain", "face"; default: "face").
- `--bg-color`: Background color for "contain" mode (RGBA, default: 0 0 0 0).
- `--contrast`: Contrast adjustment (default: 1.2).
- `--saturation`: Saturation adjustment (default: 1.2).
- `--sharpness`: Sharpness adjustment (default: 1.3).
- `--blur`: Gaussian blur radius (default: 0.0).
- `--outline`: Outline sensitivity (0-100, default: 0).

## Examples
- Basic conversion:
python convert_image_to_pixel.py --input photo.jpg --out results --size 64 96 --scale 6 --palette natural --fill face
- With dithering and outline:
python convert_image_to_pixel.py --input image.png --out pixel_art --size 128 128 --scale 4 --palette photo --dither --outline 40 --contrast 1.5 --saturation 1.3 --sharpness 1.4 --blur 0.5
- Contain mode with custom background:
python convert_image_to_pixel.py --input landscape.jpg --out output --size 100 100 --scale 5 --fill contain --bg-color 255 255 255 255 --palette undertale


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## Security
If you find a security vulnerability, please report it via the guidelines in [SECURITY.md](SECURITY.md).
