import torch
import numpy as np
import pandas as pd
from ast import literal_eval
import os, sys
import json
import random

from source import evaluation, eval_MC
from source.ok_vqa_in_context_learning import val_in_context_learning_ok_vqa_with_beit
from source.a_ok_vqa_in_context_learning import  val_in_context_learning_a_ok_vqa_with_beit
from source.a_ok_vqa_in_context_learning import test_in_context_learning_a_ok_vqa_with_beit
from source.a_ok_vqa_in_context_learning_MC import val_in_context_learning_a_ok_vqa_with_beit_MC, test_in_context_learning_a_ok_vqa_with_beit_MC

from config import get_config
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoProcessor, BlipForImageTextRetrieval
from transformers import set_seed


#get confing variables 
cnf = get_config(sys.argv)

#set up device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#unfolding common params
dataset_to_use = cnf.dataset
train_images_dir = cnf.train_images_dir
val_images_dir = cnf.val_images_dir
test_images_dir = cnf.test_images_dir

n_shots = cnf.n_shots
k_ensemble = cnf.k_ensemble
no_of_captions = cnf.no_of_captions
path_to_save_preds = cnf.path_to_save_preds 

#load Llama model
# "meta-llama/Llama-2-13b-hf"
if dataset_to_use != 'get_info':
    llama_model = LlamaForCausalLM.from_pretrained(cnf.llama_path)
    llama_tokenizer = LlamaTokenizer.from_pretrained(cnf.llama_path)
    llama_model = llama_model.to(device, dtype=torch.float16)

#load the blip model 
blip_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
blip_model = blip_model.to(device)

#load annotations
train_annotations_df = pd.read_csv(cnf.train_annotations_path)
if cnf.evaluation_set == "val":
    val_annotations_df = pd.read_csv(cnf.val_annotations_path)
else:
    test_annotations_df = pd.read_csv(cnf.test_annotations_path)

#if cnf.use_beit_examples == "True":
    #load the in-context examples 
with open(cnf.examples_path, "rb") as input:
    examples = json.load(input)
beit_examples_df = pd.DataFrame({'question_id' : examples.keys(), 'similar_examples' : examples.values()})
# else:
#     train_q_embedds = pd.read_csv(cnf.blip_train_question_embedds_path)
#     train_i_embedds = pd.read_csv(cnf.blip_train_image_embedds_path)
#     train_q_embedds.question_embedd = train_q_embedds.question_embedd.apply(literal_eval)
#     train_i_embedds.image_embedd = train_i_embedds.image_embedd.apply(literal_eval)

#     if cnf.evaluation_set == "val":
#         val_q_embedds = pd.read_csv(cnf.blip_val_question_embedds_path)
#         val_i_embedds = pd.read_csv(cnf.blip_val_image_embedds_path)
#         val_q_embedds.question_embedd = val_q_embedds.question_embedd.apply(literal_eval)
#         val_i_embedds.image_embedd = val_i_embedds.image_embedd.apply(literal_eval)
#     else: #test (only for A-OK-VQA)
#         test_q_embedds = pd.read_csv(cnf.blip_test_question_embedds_path)
#         test_i_embedds = pd.read_csv(cnf.blip_test_image_embedds_path)
#         test_q_embedds.question_embedd = test_q_embedds.question_embedd.apply(literal_eval)
#         test_i_embedds.image_embedd = test_i_embedds.image_embedd.apply(literal_eval)


#load captions 
promptcap_captions_dir_train = cnf.promptcap_captions_dir_train###
train_captions = pd.read_csv(cnf.train_captions_path)
train_captions.captions = train_captions.captions.apply(literal_eval)

if cnf.evaluation_set == 'val':
    val_captions = pd.read_csv(cnf.val_captions_path)
    val_captions.captions = val_captions.captions.apply(literal_eval)
else:
    test_captions = pd.read_csv(cnf.test_captions_path)
    test_captions.captions = test_captions.captions.apply(literal_eval)


if __name__ == "__main__":
    if dataset_to_use == "ok_vqa":
        #apply literal eval to the answers
        train_annotations_df.answers = train_annotations_df.answers.apply(literal_eval)
        val_annotations_df.answers = val_annotations_df.answers.apply(literal_eval)

        beit_examples_df['question_id'] = beit_examples_df['question_id'].astype('int')
        results_df = val_in_context_learning_ok_vqa_with_beit(llama_model=llama_model, 
                                                                llama_tokenizer=llama_tokenizer,
                                                                blip_model=blip_model,
                                                                blip_processor=blip_processor,
                                                                train_annotations_df=train_annotations_df,
                                                                val_annotations_df=val_annotations_df,
                                                                train_captions=train_captions, 
                                                                val_captions=val_captions,
                                                                context_examples_df=beit_examples_df, 
                                                                train_images_dir=train_images_dir, 
                                                                val_images_dir=val_images_dir,
                                                                promptcap_captions_dir_train=promptcap_captions_dir_train,
                                                                n_shots=n_shots, 
                                                                k_ensemble=k_ensemble,
                                                                MAX_CAPTION_LEN=30,
                                                                NO_OF_CAPTIONS_AS_CONTEXT=no_of_captions,
                                                                path_to_save_preds=path_to_save_preds,
                                                                device=device)
    
    elif cnf.dataset == "a_ok_vqa":
        #apply literal eval to the answers
        train_annotations_df.direct_answers = train_annotations_df.direct_answers.apply(literal_eval)

        if cnf.evaluation_set == "val":
            #apply literal eval to the answers
            val_annotations_df.direct_answers = val_annotations_df.direct_answers.apply(literal_eval)
            if cnf.multiple_choice:
                results_df = val_in_context_learning_a_ok_vqa_with_beit_MC(llama_model=llama_model,
                                                                    llama_tokenizer=llama_tokenizer,
                                                                    blip_model=blip_model,
                                                                    blip_processor=blip_processor,
                                                                    train_annotations_df=train_annotations_df,
                                                                    val_annotations_df=val_annotations_df, 
                                                                    train_captions=train_captions, 
                                                                    val_captions=val_captions,
                                                                    context_examples_df=beit_examples_df, 
                                                                    train_images_dir=train_images_dir, 
                                                                    val_images_dir=val_images_dir,
                                                                    n_shots=n_shots, 
                                                                    k_ensemble=k_ensemble, 
                                                                    MAX_CAPTION_LEN=30, 
                                                                    NO_OF_CAPTIONS_AS_CONTEXT=no_of_captions,
                                                                    path_to_save_preds=path_to_save_preds,
                                                                    device=device)
            else:
                results_df = val_in_context_learning_a_ok_vqa_with_beit(llama_model=llama_model,
                                                                        llama_tokenizer=llama_tokenizer,
                                                                        blip_model=blip_model,
                                                                        blip_processor=blip_processor,
                                                                        train_annotations_df=train_annotations_df,
                                                                        val_annotations_df=val_annotations_df, 
                                                                        train_captions=train_captions, 
                                                                        val_captions=val_captions,
                                                                        context_examples_df=beit_examples_df, 
                                                                        train_images_dir=train_images_dir, 
                                                                        val_images_dir=val_images_dir,
                                                                        n_shots=n_shots, 
                                                                        k_ensemble=k_ensemble, 
                                                                        MAX_CAPTION_LEN=30, 
                                                                        NO_OF_CAPTIONS_AS_CONTEXT=no_of_captions,
                                                                        path_to_save_preds=path_to_save_preds,
                                                                        device=device)
        elif cnf.evaluation_set == "test":
            if cnf.multiple_choice:
                results_df = test_in_context_learning_a_ok_vqa_with_beit_MC(llama_model=llama_model,
                                                                    llama_tokenizer=llama_tokenizer,
                                                                    blip_model=blip_model,
                                                                    blip_processor=blip_processor,
                                                                    train_annotations_df=train_annotations_df,
                                                                    test_annotations_df=test_annotations_df, 
                                                                    train_captions=train_captions, 
                                                                    test_captions=test_captions,
                                                                    context_examples_df=beit_examples_df, 
                                                                    train_images_dir=train_images_dir, 
                                                                    test_images_dir=test_images_dir,
                                                                    n_shots=n_shots, 
                                                                    k_ensemble=k_ensemble, 
                                                                    MAX_CAPTION_LEN=30, 
                                                                    NO_OF_CAPTIONS_AS_CONTEXT=no_of_captions,
                                                                    path_to_save_preds=path_to_save_preds,
                                                                    device=device)
            else:
                results_df = test_in_context_learning_a_ok_vqa_with_beit(llama_model=llama_model,
                                                                    llama_tokenizer=llama_tokenizer,
                                                                    blip_model=blip_model,
                                                                    blip_processor=blip_processor,
                                                                    train_annotations_df=train_annotations_df,
                                                                    test_annotations_df=test_annotations_df, 
                                                                    train_captions=train_captions, 
                                                                    test_captions=test_captions,
                                                                    context_examples_df=beit_examples_df, 
                                                                    train_images_dir=train_images_dir, 
                                                                    test_images_dir=test_images_dir,
                                                                    n_shots=n_shots, 
                                                                    k_ensemble=k_ensemble, 
                                                                    MAX_CAPTION_LEN=30, 
                                                                    NO_OF_CAPTIONS_AS_CONTEXT=no_of_captions,
                                                                    path_to_save_preds=path_to_save_preds,
                                                                    device=device)
                                        
    
    #evaluate the predictions (only for val sets)
    if cnf.evaluation_set == "val": 
        results_df = pd.merge(val_annotations_df, results_df, on = 'question_id')
        if cnf.dataset == "ok_vqa":
            results_df['acc'] = results_df.apply(lambda row: evaluation.okvqa_ems(row['llama_answer'], row['answers']), axis = 1)
            print("\n========")
            print("VQA acc: ", np.round(results_df.acc.mean(),4))
            print("=========")
        else:
            if cnf.multiple_choice:
                eval_MC.eval_MC(results_df)
            else:
                results_df['acc'] = results_df.apply(lambda row: evaluation.okvqa_ems(row['llama_answer'], row['direct_answers']), axis = 1)
                print("\n========")
                print("VQA acc: ", np.round(results_df.acc.mean(),4))
                print("==========")
    


    





                   


  