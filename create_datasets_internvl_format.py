
#download data from : https://www.kaggle.com/datasets/bhavikardeshna/visual-question-answering-computer-vision-nlp/data


'''
this is the data format:
{
  "id": 0,
  "image": "images/00000000.jpg",
  "width": 897,
  "height": 1152,
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nCan you extract any readable text from the image?"
    },
    {
      "from": "gpt",
      "value": "Dares Wins Vol. 5 Tommy's Heroes Vol. 6: For Tomorrow Vol. 7: Closing Time miniseries. Clark Kent is being interviewed about Superman's connection to notorious killer Tommy Monaghan. Taking the conversation..."
    }
  ]
}'''
import os
import pandas as pd
from PIL import Image
import json
import shutil
import random

os.system('mkdir annotations')
os.system('mkdir images/train')
os.system('mkdir images/val')


def split_dataset(image_dir, output_dir, train_ratio=0.8):
    # Get all image files
    images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)  # Shuffle to ensure randomness

    # Split into train and test
    split_idx = int(len(images) * train_ratio)
    train_images, test_images = images[:split_idx], images[split_idx:]

    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Move images to respective folders
    for img in train_images:
        shutil.copy(os.path.join(image_dir, img), os.path.join(train_dir, img))

    for img in test_images:
        shutil.copy(os.path.join(image_dir, img), os.path.join(test_dir, img))

    print(f"Total images: {len(images)}")
    print(f"Train set: {len(train_images)} images")
    print(f"Test set: {len(test_images)} images")

    return train_images, test_images


def convert_df(train_images):
    data_df = pd.read_csv('data.csv')
    image_ids = [i.replace('.png', '') for i in train_images]
    image_ids = set(image_ids)
    annos = []
    for ids in image_ids:
        temp = {}
        temp['id'] = int(ids.replace('image', ''))
        temp['image'] = "images/image%s.png"%ids
        img = "images/%s.png"%ids
        width, height = Image.open(img).size
        temp['width'] = width
        temp['height'] = height
        qas = data_df[data_df['image_id']==ids].values
        conversations = []
        for qa in qas:
            ques = qa[0]
            ans = qa[1]
            ques_format = {
                "from": "human",
                "value": "<image>\n%s"%ques
                }
            ans_format = {
                "from": "gpt",
                "value": ans
                }
            conversations.append(ques_format)
            conversations.append(ans_format)
        temp['conversations'] = conversations
        annos.append(temp)
    return annos


def main():
    image_directory = "images"
    output_directory = "images" 
    train_images, test_images = split_dataset(image_directory, output_directory)
    with open("annotations/vqa_train.jsonl", "w") as f:
        train_annos = convert_df(train_images)
        for entry in train_annos:
            f.write(json.dumps(entry) + "\n")
        f.close()


    with open("annotations/vqa_val.jsonl", "w") as f:
            val_annos = convert_df(test_images)
            for entry in val_annos:
                f.write(json.dumps(entry) + "\n")
            f.close()   


main()