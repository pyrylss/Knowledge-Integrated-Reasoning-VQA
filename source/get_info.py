import os
import torch
import pandas as pd 
import numpy as np 
from ast import literal_eval
import numpy as np
from PIL import Image
from tqdm import tqdm
from source.utils import sort_captions_based_on_similarity, get_context_examples

import json

def val_in_context_learning_ok_vqa_with_prophet(
                                             
                                      blip_model, 
                                      blip_processor,
                                      train_annotations_df, 
                                      val_annotations_df, 
                                      train_images_dir, 
                                      test_images_dir, 
                                      n_shots=10, 
                                      k_ensemble=5, 
                                      MAX_CAPTION_LEN=30, 
                                      NO_OF_CAPTIONS_AS_CONTEXT=10, 
                                      path_to_save_preds = None,
                                      device = "cpu"):
  
  """
  Performs n-shot in context learning using the mcan-based shot selection
  :param llama_model: The llama huggingface model
  :param llama_tokenizer: The llama huggingface tokenizer
  :param blip_model: The blip huggingface model
  :param blip_processor: The blip huggingface processor
  :param train_annotations_df: Dataframe containing the ok_vqa train annotations
  :param val_annotations_df: Dataframe containing the ok_vqa val annotations
  :param context_examples_df: Dataframe containing the mcan examples
  :param train_captions: Dataframe containing the train question-informative captions
  :param val_captions: Dataframe containing the val question-informative captions
  :param train_images_dir: The path of the folder containing the training images
  :param val_images_dir: The path of the folder containing the val images
  :param n_shots: The number of shots for the in-context few-shot learning
  :param k_ensemble: The number of ensembles
  :param MAX_CAPTION_LEN: The number of maximum words to keep for each caption
  :param NO_OF_CAPTIONS_AS_CONTEXT: The number of captions to use as context for each shot
  :param path_to_save_preds: Path to save the predictions as a csv file
  :param device: Cpu or gpu device
  :returns llama_preds_df: Dataframe containing the final predictions
  """



   ### Caption with knowledge
  # Load captions into dictionaries
  # with open("/scratch/project_462000472/pyry/prophet/PromptCAP/okvqa_train_captions_w_knowledge.json", 'r') as f:
  #       train_captions_w_knowledge = json.load(f)
  # with open("/scratch/project_462000472/pyry/prophet/PromptCAP/okvqa_val_captions_w_knowledge.json", 'r') as f:
  #       val_captions_w_knowledge = json.load(f)
  train_captions_knowledge_dict = {}
  with open("/scratch/project_462000472/pyry/prophet/PromptCAP/aokvqa_train_captions_w_knowledge_step605.json", 'r') as f:
      train_captions_w_knowledge = json.load(f)
      train_captions_knowledge_dict.update({entry["question_id"]: entry["captions"] for entry in train_captions_w_knowledge})

  with open("/scratch/project_462000472/pyry/prophet/PromptCAP/aokvqa_val_captions_w_knowledge_step605.json", 'r') as f:
        val_captions_w_knowledge = json.load(f)
        train_captions_knowledge_dict.update({entry["question_id"]: entry["captions"] for entry in val_captions_w_knowledge})

  with open("/scratch/project_462000472/pyry/prophet/PromptCAP/aokvqa_test_captions_w_knowledge_step605.json", 'r') as f:
        test_captions_w_knowledge = json.load(f)

  

  #train_captions_knowledge_dict = {entry["question_id"]: entry["captions"] for entry in train_captions_w_knowledge}
  val_captions_knowledge_dict = {entry["question_id"]: entry["captions"] for entry in val_captions_w_knowledge}

  result_file = {}

  # for i in tqdm(range(train_annotations_df.shape[0])):
   
  #   train_sample = train_annotations_df.iloc[i]
  #   raw_image = Image.open(train_images_dir+train_sample.image_path)
    
    
  #   train_sample_captions_w_knowledge = train_captions_knowledge_dict.get(train_sample.question_id, [])
  #   train_sample_captions_w_knowledge = [c["caption"] for c in train_sample_captions_w_knowledge]
  #   train_sample_captions_w_knowledge, cos_scores = sort_captions_based_on_similarity(train_sample_captions_w_knowledge,raw_image=raw_image,model=blip_model,processor=blip_processor,device=device, ascending=False)

  #   result_file[train_sample.question_id] = {
  #       'caption': train_sample_captions_w_knowledge[0]
  #   }

  #   json.dump(result_file, open('aokvqa_captions_w_knowledge', 'w'))

  for i in tqdm(range(val_annotations_df.shape[0])):
   
    test_sample = val_annotations_df.iloc[i]
    raw_test_image = Image.open(test_images_dir+test_sample.image_path)
    
    
    test_sample_captions_w_knowledge = val_captions_knowledge_dict.get(test_sample.question_id, [])
    test_sample_captions_w_knowledge = [c["caption"] for c in test_sample_captions_w_knowledge]
    test_sample_captions_w_knowledge, cos_scores = sort_captions_based_on_similarity(test_sample_captions_w_knowledge,raw_image=raw_test_image,model=blip_model,processor=blip_processor,device=device, ascending=False)
    result_file[test_sample.question_id] = {
        'caption': test_sample_captions_w_knowledge[0]
    }
    json.dump(result_file, open('aokvqa_val_captions_w_knowledge', 'w'))

