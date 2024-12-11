import os
import requests
from time import sleep

output_dir = "fake_images_test"
os.makedirs(output_dir, exist_ok=True)

def save_image_locally(index, output_dir):
    url = "https://thispersondoesnotexist.com"
    image_path = os.path.join(output_dir, f"fake_face_{index}.jpg")
    
    if os.path.exists(image_path):
        print(f"Image {index} already exists. Skipping.")
        return
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(image_path, "wb") as file:
                file.write(response.content)
            print(f"Saved locally: {image_path}")
        else:
            print(f"Failed to fetch image {index}, HTTP status: {response.status_code}")
    except requests.RequestException as e:
        print(f"Error fetching image {index}: {e}")


total_images = 10004
for i in range(10000,total_images):
    save_image_locally(i, output_dir)
    sleep(0.3)  