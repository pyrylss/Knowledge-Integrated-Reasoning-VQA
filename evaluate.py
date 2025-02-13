import pandas as pd   # For reading CSV and DataFrame operations
import numpy as np    # For numerical operations like rounding
from source import evaluation  # Replace with the actual path to your evaluation module
from ast import literal_eval


dataset='ok_vqa'


results_df = pd.read_csv('/scratch/project_462000472/pyry/A-Simple-Baseline-For-Knowledge-Based-VQA/results/OKVQA_val_gpt3.csv')

results_df['llama_answer'] = results_df['llama_answer'].fillna("").astype(str)

if dataset == "ok_vqa":
    val_annotations_df = pd.read_csv('/scratch/project_462000472/pyry/A-Simple-Baseline-For-Knowledge-Based-VQA/annotations/ok_vqa/val_annots_fixed.csv.zip')
    val_annotations_df.answers = val_annotations_df.answers.apply(literal_eval)
    results_df = pd.merge(val_annotations_df, results_df, on = 'question_id')
    results_df['acc'] = results_df.apply(lambda row: evaluation.okvqa_ems(row['llama_answer'], row['answers']), axis = 1)
else:
    val_annotations_df = pd.read_csv('/scratch/project_462000472/pyry/A-Simple-Baseline-For-Knowledge-Based-VQA/annotations/a_ok_vqa/a_ok_vqa_val_fixed_annots.csv.zip')
    val_annotations_df.direct_answers = val_annotations_df.direct_answers.apply(literal_eval)
    results_df = pd.merge(val_annotations_df, results_df, on = 'question_id')
    results_df['acc'] = results_df.apply(lambda row: evaluation.okvqa_ems(row['llama_answer'], row['direct_answers']), axis = 1)
print("\n========")
print("VQA acc: ", np.round(results_df.acc.mean(),4))
print("==========")
    