import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import requests
from concurrent.futures import ThreadPoolExecutor
import json

# Function to clean and rephrase a single question
def rephrase_question_deepseek(question, api_key):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Rephrase the following question without changing its context or correct answer:\n\n{question}"}
        ],
        "temperature": 1.0,
        "stream": False
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            output = result['choices'][0]['message']['content'].strip()
            
            # Strip instructions from the output
            if "Rephrase the following" in output:
                output = output.split("\n\n", 1)[-1].strip()
            return output
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    except Exception as e:
        return f"Error: {e}"

# Process questions in parallel
def process_questions_parallel(dataset, api_key, max_workers=10):
    rephrased_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for item in dataset:
            question = item['question']
            futures.append(
                executor.submit(rephrase_question_deepseek, question, api_key)
            )
        for item, future in zip(dataset, futures):
            try:
                rephrased_question = future.result()
                rephrased_data.append({
                    'subject': item['subject'],          # Subject
                    'original_question': item['question'],  # Original Question
                    'rephrased_question': rephrased_question,  # Rephrased Question
                    'choices': item['choices'],          # Answer Choices
                    'answer': item['answer']             # Correct Answer
                })
            except Exception as e:
                print(f"Error processing question: {e}")
    return rephrased_data

# Load the MMLU dataset
api_key = "XXXXXXXXXXXXXX"  # Replace with your actual API key 
subject = 'all'  # Specify the subject or use 'all' for the complete dataset
dataset = load_dataset('cais/mmlu', subject, split='test')

# Process the dataset in parallel
max_workers = 2
rephrased_data = process_questions_parallel(dataset, api_key, max_workers=max_workers)

# Save the rephrased dataset
output_file = 'rephrased_mmlu_test_parallel_temp1_0.json'
with open(output_file, 'w') as f:
    json.dump(rephrased_data, f, indent=4)

print(f"Rephrased data saved to {output_file}")
