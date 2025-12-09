import os
import csv
import xml.etree.ElementTree as ET

def parse_voc_annotation(xml_path):
    """Parse a single VOC XML file and extract object name + bbox."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_name = root.find('filename').text.strip()
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text.strip()
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        # Convert to normalized coordinates (optional)
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height

        objects.append({
            "filename": img_name,
            "sign_name": name,
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "x_center": round(x_center, 6),
            "y_center": round(y_center, 6),
            "width_norm": round(bw, 6),
            "height_norm": round(bh, 6)
        })
    return objects


def xml_folder_to_csv(xml_folder, csv_output):
    """Parse all XML files in a folder and write them into a CSV file."""
    all_data = []

    for file in os.listdir(xml_folder):
        if file.endswith(".xml"):
            xml_path = os.path.join(xml_folder, file)
            all_data.extend(parse_voc_annotation(xml_path))

    # Write to CSV
    if all_data:
        keys = all_data[0].keys()
        with open(csv_output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_data)
        print(f"CSV file saved successfully: {csv_output}")
    else:
        print("No XML files found in the folder.")


# -----------------------------
# Example usage
# -----------------------------
xml_folder = "./1_Datasets/dataset3/annotations"
csv_output = "./1_Datasets/dataset3/traffic_sign_labels.csv"

xml_folder_to_csv(xml_folder, csv_output)
