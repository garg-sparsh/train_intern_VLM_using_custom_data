
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

def save_annotations_comparison(annos):
    annotations = []
    for entry in annos:
        temp = {'image_id': int(entry['image'].split('/')[-1].replace('image', '').replace('.png', '')), 'answer_type': 'other', 'question_type': 'other', 'question_id': entry['question_id']}
        answers = []
        ans1 = {'answer_id': 0, 'answer': entry['answer'], 'answer_confidence': 'yes'}
        ans2 = {'answer_id': 1, 'answer': entry['answer'].replace('_', ' '), 'answer_confidence': 'yes'}
        answers.append(ans1)
        answers.append(ans2)
        temp['answers'] = answers
        annotations.append(temp)
    return annotations

def save_questions_comparison(ques):
    questions = []
    for entry in ques:
        temp =  {'image':entry['image'], 'image_id': int(entry['image'].split('/')[-1].replace('image', '').replace('.png', '')),  'question_id': entry['question_id'], 'question' :entry['question']}
        questions.append(temp)
    return questions


def convert_df(train_images):
    data_df = pd.read_csv('data.csv')
    image_ids = [i.replace('.png', '') for i in train_images]
    image_ids = set(image_ids)
    annos = []
    count = 0
    for ids in image_ids:
        # temp = {}
        # temp['id'] = int(ids.replace('image', ''))
        # temp['image'] = "images/image%s.png"%ids
        # img = "images/%s.png"%ids
        # width, height = Image.open(img).size
        # temp['width'] = width
        # temp['height'] = height
        qas = data_df[data_df['image_id']==ids].values
        for qa in qas:
            temp = {}
            temp['image'] = "data/custom_vqa/images/train/%s.png"%ids
            temp['question'] = qa[0]
            temp['question_id'] = count
            temp['answer'] = qa[1]
            annos.append(temp)
            count+=1
    return annos

def main():
    image_dir = 'images'
    output_dir = 'images'
    train_images, val_images = split_dataset(image_dir, output_dir, train_ratio=0.8)
    train_images = [f for f in os.listdir(image_dir+'/train') if f.endswith(('.jpg', '.png', '.jpeg'))]
    val_directory = "images/val"
    val_images = [f for f in os.listdir(image_dir+'/val') if f.endswith(('.jpg', '.png', '.jpeg'))]
    #test_images = split_dataset(image_directory, output_directory)
    with open("annotations/vqa_train.jsonl", "w") as f:
        train_annos = convert_df(train_images)
        for entry in train_annos:
            f.write(json.dumps(entry) + "\n")
        f.close()


    with open("annotations/vqa_val.jsonl", "w") as f:
            val_annos = convert_df(val_images)
            for entry in val_annos:
                f.write(json.dumps(entry) + "\n")
            f.close()   

    '''
    This is the format to save annotations file:
    {'image_id': 0, 'answer_type': 'other', 'question_type': 'other', 'question_id': 34602, 'answers': [{'answer_id': 0, 'answer': 'nous les gosses', 'answer_confidence': 'yes'}, 
    {'answer_id': 1, 'answer': 'dakota', 'answer_confidence': 'yes'}, {'answer_id': 2, 'answer': 'clos culombu', 'answer_confidence': 'yes'},
    {'answer_id': 3, 'answer': 'dakota digital', 'answer_confidence': 'yes'}, {'answer_id': 4, 'answer': 'dakota', 'answer_confidence': 'yes'}, 
    {'answer_id': 5, 'answer': 'dakota', 'answer_confidence': 'yes'}, {'answer_id': 6, 'answer': 'dakota digital', 'answer_confidence': 'yes'}, 
    {'answer_id': 7, 'answer': 'dakota digital', 'answer_confidence': 'yes'}, {'answer_id': 8, 'answer': 'dakota', 'answer_confidence': 'yes'}, 
    {'answer_id': 9, 'answer': 'dakota', 'answer_confidence': 'yes'}]}


    This is the format to save questions file:
    {'image': 'data/textvqa/train_images/003a8ae2ef43b901.jpg', 'image_id': 0, 'question': 'what is the brand of this camera?', 'question_id': 34602}

    '''
    print('train_annos:', train_annos)
    with open("custom_qa_train_annotations.json", "w") as f:
        train_answers = save_annotations_comparison(train_annos)
        train_answers = {'annotations':train_answers}
        f.write(json.dumps(train_answers))
        f.close()
    
    with open("custom_qa_val_annotations.json", "w") as f:
        val_answers = save_annotations_comparison(val_annos)
        val_answers = {'annotations':val_answers}
        f.write(json.dumps(val_answers))
        f.close()

    with open("custom_qa_train_questions.json", "w") as f:
        train_questions = save_questions_comparison(train_annos)
        train_answers = {'questions':train_questions, 'info':'', 'license':'', 'data_subtype':'', 'data_type':''}
        f.write(json.dumps(train_answers))
        f.close()
    
    with open("custom_qa_val_questions.json", "w") as f:
        val_questions = save_questions_comparison(val_annos)
        val_answers = {'questions':val_questions, 'info':'', 'license':'', 'data_subtype':'', 'data_type':''}
        f.write(json.dumps(val_answers))
        f.close()
    


       


main()