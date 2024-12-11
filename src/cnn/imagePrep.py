import os
from PIL import Image
import random
from tqdm import tqdm

celebA_dir = "../../real_images_test"  
stylegan_dir = "../../fake_images_test"

resized_celebA_dir = "test_realImages"
resized_stylegan_dir = "test_fakeImages"

os.makedirs(resized_celebA_dir, exist_ok=True)
os.makedirs(resized_stylegan_dir, exist_ok=True)

target_size = (128, 128)

def resize_images(input_dir, output_dir, max_images=None):
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]

    if max_images and len(image_files) > max_images:
        image_files = random.sample(image_files, max_images)

    print(f"Resizing {len(image_files)} images from {input_dir}...")
    for filename in tqdm(image_files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
           
            with Image.open(input_path) as img:
               
                img = img.convert("RGB")
                
                img_resized = img.resize(target_size, Image.ANTIALIAS)
               
                img_resized.save(output_path)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")

resize_images(celebA_dir, resized_celebA_dir, max_images=1000)

resize_images(stylegan_dir, resized_stylegan_dir, max_images=1000)