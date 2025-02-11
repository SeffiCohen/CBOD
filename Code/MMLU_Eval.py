import os
import sys
import json
import argparse
import re
import torch
from auto_round import AutoHfQuantizer  ##must import
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm


def extract_answer(generated_text):
    # This function tries to extract the answer choice from the generated text.
    # We'll assume the answer is indicated by a letter A-D. 
    # First, try a direct "Answer: X" pattern:
    match = re.search(r'Answer:\s*([A-D])', generated_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    else:
        # If not found, try to find a trailing A-D in the text.
        letters = re.findall(r'\b([A-D])\b', generated_text.upper())
        if letters:
            return letters[-1].upper()
        else:
            return None

def format_prompt(question, choices):
    # Formats the prompt for the LLM
    # We limit ourselves to A-D answers.
    choice_letters = ['A', 'B', 'C', 'D'][:len(choices)]
    prompt = (
        "You are a helpful assistant. Read the question and the provided options. "
        "Select the best answer from the given options. Respond with just the letter of the correct choice.\n\n"
    )
    prompt += f"Question: {question}\n"
    for i, option in enumerate(choices):
        prompt += f"{choice_letters[i]}. {option}\n"
    prompt += "Answer:"
    return prompt

def batch_predict(model, tokenizer, device, prompts, batch_size=1):
    # Runs prediction in batches over the given prompts
    predictions = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Running inference in batches"):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=10,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id
            )
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for text in generated_texts:
            predicted_answer = extract_answer(text)
            predictions.append(predicted_answer)
    return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=True, help="Model identifier (Hugging Face Hub name or local path)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    args = parser.parse_args()

    model_name = args.model
    batch_size = args.batch_size

    # Load the questions from the provided JSON file
    input_filename = "rephrased_mmlu_test.json"
    with open(input_filename, 'r') as f:
        data = json.load(f)

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #quantization_config = BitsAndBytesConfig(load_in_4bit=True) ##########
   
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Some models don't have a pad token, set it if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map='auto', trust_remote_code=True)
    model.to(device)
    model.eval()

    # Prepare all prompts
    original_prompts = []
    rephrased_prompts = []
    for entry in data:
        original_prompts.append(format_prompt(entry["original_question"], entry["choices"]))
        rephrased_prompts.append(format_prompt(entry["rephrased_question"], entry["choices"]))

    # Run inference in batches
    original_predictions = batch_predict(model, tokenizer, device, original_prompts, batch_size=batch_size)
    rephrased_predictions = batch_predict(model, tokenizer, device, rephrased_prompts, batch_size=batch_size)

    results = []
    for entry, orig_pred, reph_pred in zip(data, original_predictions, rephrased_predictions):
        correct_answer_index = entry["answer"]
        correct_answer_letter = ['A', 'B', 'C', 'D'][correct_answer_index]

        original_correct = (orig_pred == correct_answer_letter)
        rephrased_correct = (reph_pred == correct_answer_letter)

        result_entry = {
            "subject": entry["subject"],
            "original_question": entry["original_question"],
            "rephrased_question": entry["rephrased_question"],
            "choices": entry["choices"],
            "answer": correct_answer_index,
            "original_question_is_correct": original_correct,
            "rephrased_question_is_correct": rephrased_correct
        }
        results.append(result_entry)

    # Save the results
    output_filename = f"results_{model_name.replace('/', '_')}.json"
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
