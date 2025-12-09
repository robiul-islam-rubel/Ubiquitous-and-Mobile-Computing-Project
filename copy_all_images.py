import os
import shutil
import glob

def collect_images_two_levels(root_dir="Images", output_dir="glaredataset", extensions=("*.png", "*.jpg", "*.jpeg")):
    os.makedirs(output_dir, exist_ok=True)
    print(root_dir)

    count = 0
    for ext in extensions:
        # Match: Images/vid*/<subfolder>/*.png
        all_images = glob.glob(os.path.join(root_dir, "vid*", "*", ext))         # one level
        all_images += glob.glob(os.path.join(root_dir, "vid*", "*", "*.png"))   # two levels

        for img in all_images:
            filename = os.path.basename(img)
            dest = os.path.join(output_dir, filename)
            shutil.copy(img, dest)
            count += 1

    print(f"Copied {count} images into {output_dir}")

collect_images_two_levels(root_dir="../Sun_Glare_Adversarial_Attacks_on_Traffic_Sign/Images", output_dir="glaredataset")