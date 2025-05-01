# MultiFinBenOCR
Repo for MultiFinBen OCR task


# How to use 
1. Before running gpt, please go to main/lib/agent.py and put in your openai_api_key
   
3. Run main/main.py for models to generate OCR output. 
The model is default to be gpt-4o, and language default to be English. If want to change languagr or model, or only run model on small sample, update this part:
```
def main():
    evaluate(model_name="gpt-4o",language = "en", sample = 20)
```

3. After running main.py, run bar_plot.py to both output evaluation metrics (BLEU and BERTScore), and plot corresponding violin plots. 
To control input output path, or change models, csv names etc., please update this part:
```
def main():
    run_eval_and_plot(
        parquet_path="hyr_ocr_process/spanish_output_parquet/spanish_batch_0000.parquet",
        pred_dir="hyr_results/predictions_spanish/gpt-4o_zero-shot_financial",
        model_name="gpt-4o",
        output_csv="hyr_results/eval_spanish_gpt_4o.csv"
    )
```

# Dataset
- Dataset are available on HuggingFace: [TheFinAI/MultiFinBen_OCR_Task](https://huggingface.co/datasets/TheFinAI/MultiFinBen_OCR_Task)
