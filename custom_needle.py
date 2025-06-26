import base64
import json
import pickle
import random
import os
from PIL import Image
import io
import time
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
import torch

def map_to_string_output(generated_text_1, generated_text_2):
    # Initialize the first number
    first_number = 1

    generated_text_1 = generated_text_1.strip().lower()
    if generated_text_1 == "top":
        second_number = 1
    elif generated_text_1 == "bottom":
        second_number = 2
    else:
        second_number = -1

    generated_text_2 = generated_text_2.strip().lower()
    if generated_text_2 == "left":
        third_number = 1
    elif generated_text_2 == "right":
        third_number = 2
    else:
        third_number = -1

    if second_number == -1 or third_number == -1:
        second_number = -1
        third_number = -1

    return f"{first_number}, {second_number}, {third_number}"

def load_finetuned_model(base_model_name, adapter_path, new_weights_path=None):
    # Load the processor/tokenizer
    processor = AutoProcessor.from_pretrained(base_model_name)

    # Step 1: Load the base model from its pretrained weights
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(base_model_name)
    
    # Step 2: If you have additional fine-tuned weights, load them into the base model
    if new_weights_path is not None:
        # hard set the lambda parameters
        for name, param in base_model.named_parameters():
            if ('lambda_q1' in name or
            'lambda_k1' in name or
            'lambda_q2' in name or
            'lambda_k2' in name or
            'subln' in name):
                print(f"Setting custom values for {name}")
                param.data = nn.Parameter(torch.empty(512 // 2, dtype=torch.bfloat16, device="cuda").normal_(mean=0.0, std=0.1))  # Example: Set random values

        # new_weights should be a dict of parameter_name -> tensor
        new_weights = torch.load(new_weights_path, map_location="cpu")

        # Merge any new weights with existing model parameters
        # If the new_weights only contain the updated parameters, you can load them directly:
        adjusted_new_weights = {}
        for k, v in new_weights.items():
            # Remove the "base_model.model." prefix if present
            new_key = k.replace("base_model.model.", "")
            adjusted_new_weights[new_key] = v

        # Now load the adjusted weights into the base model
        base_model.load_state_dict(adjusted_new_weights, strict=False)


    # Step 3: Load the adapter (LoRA or similar) on top of the base model
    # The adapter_path should contain the adapter configuration and weights
    model_with_adapter = PeftModel.from_pretrained(base_model, adapter_path)

    # Move model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_with_adapter.to(device)

    return model_with_adapter, processor

def needle_test(images, instruction, model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        base_model_name = "paligemma-3b-pt-224"
        model_path = "C:/Users/vulte/Documents/CS228/paligemma-3b-pt-224"
        adapter_path = ""
        new_weights_path = ""
        if model_name == 'Base_Paligemma':
            processor = AutoProcessor.from_pretrained(model_path)
            model = PaliGemmaForConditionalGeneration.from_pretrained(model_path)
            model.to(device).eval()
        elif model_name == "Finetuned_Paligemma":
            model, processor = load_finetuned_model(base_model_name, adapter_path, new_weights_path)

        pil_images = [Image.open(io.BytesIO(base64.b64decode(img))) for img in images]
        model_inputs = processor(text=[instruction], images=pil_images, return_tensors="pt").to(device)

        input_len = model_inputs["input_ids"].shape[-1]
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]

        decoded_response = processor.batch_decode(generation, skip_special_tokens=True)[0]
        return decoded_response

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open('annotations_trainval/file_to_caption.pkl', "rb") as image_file:
        file_to_caption = pickle.load(image_file)

    if N_NEEDLES == 1:
        meta_path = 'annotations_' + str(SEQ_LENGTH) + '_' + res_dir + '.json'
        meta_path = os.path.join('metadata_stitched', meta_path)
    else:
        meta_path = str(N_NEEDLES) + '_' + 'annotations_' + str(SEQ_LENGTH) + '_' + res_dir + '.json'
        meta_path = os.path.join('metadata_stitched', meta_path)

    with open(meta_path, 'r') as f:
        meta_data = json.load(f)

    results = []

    for id in range(BEGIN, BEGIN + N_SEQ):
        t0 = time.time()
        image_paths = meta_data[id]['image_ids']

        if N_NEEDLES == 1:
            idx = meta_data[id]['index']
            row = meta_data[id]['row']
            col = meta_data[id]['col']
            target_path = meta_data[id]['target'].split('/')[-1]
        else:
            idx_list = meta_data[id]['index']
            row_list = meta_data[id]['row']
            col_list = meta_data[id]['col']
            target_path = meta_data[id]['target']
            target_path = [tt.split('/')[-1] for tt in target_path]

        images = []
        for path in image_paths:
            with open(path, 'rb') as f:
                image = f.read()
            base64_image = base64.b64encode(image).decode('utf-8')
            images.append(base64_image)

        if N_NEEDLES == 1:
            caption = file_to_caption[target_path]
        else:
            captions = [file_to_caption[path] for path in target_path]

        img_str = SEQ_LENGTH > 1 and 'images' or 'image'
        subimage_str = N_ROW > 1 and 'subimages' or 'subimage'

        if N_NEEDLES == 1:
            # prompt = f"Given {SEQ_LENGTH} {img_str} indexed from 1 to {SEQ_LENGTH}, each divided into {N_ROW}*{N_COL} {subimage_str}, identify the subimage that best matches the provided caption. Respond with 'index, row, column' and nothing else. For example, '1, 2, 3' indicates the subimage in the first image, second row, and third column. If no match is found, respond only with '-1'."

            instruction_top_bottom = f"Caption: {caption}. Where is the caption? Top or Bottom?"
            instruction_left_right = f"Caption: {caption}. Where is the caption? Left or Right?"

        if N_NEEDLES == 1:
            print(f'{idx+1}, {row+1}, {col+1}')
        response_tb = needle_test(images, instruction_top_bottom, MODEL_NAME)
        print(instruction_top_bottom)
        print(f"Top/Bottom Response: {response_tb}\n")
        response_lr = needle_test(images, instruction_left_right, MODEL_NAME)
        print(instruction_left_right)
        print(f"Right/Left Response: {response_lr}\n")
        response = map_to_string_output(response_tb, response_lr)

        if N_NEEDLES == 1:
            gt = f'{idx+1}, {row+1}, {col+1}'
        else:
            gt = '; '.join([f'{idx_list[i]+1}, {row_list[i]+1}, {col_list[i]+1}' for i in range(N_NEEDLES)])

        print(f"Response: {response}  | GT: {gt}")

        results.append({
            "gt": gt,
            "pred": response
        })

    # === New logic to save all results into one JSON file, keyed by model name ===
    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Append or create model results
    if MODEL_NAME in all_results:
        all_results[MODEL_NAME].extend(results)
    else:
        all_results[MODEL_NAME] = results

    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Saved results for model '{MODEL_NAME}' to {output_json}")

if __name__ == "__main__":
    N_ROW = int(os.getenv('N_ROW', '2'))
    N_COL = int(os.getenv('N_COL', '2'))
    SEQ_LENGTH = int(os.getenv('SEQ_LENGTH', '1'))
    BEGIN = int(os.getenv('BEGIN', '0'))
    N_SEQ = int(os.getenv('N_SEQ', '1'))
    N_NEEDLES = int(os.getenv('N_NEEDLES', '1'))
    random.seed(0)

    data_dir = os.getenv('DATA_DIR', 'images_stitched')
    res_dir = f"{N_ROW}_{N_COL}"
    output_dir = 'response'
    os.makedirs(output_dir, exist_ok=True)

    output_suffix = f'_{BEGIN}_{BEGIN + N_SEQ - 1}'
    output_dir = os.path.join(output_dir, 'COCO_val2014' + output_suffix)
    os.makedirs(output_dir, exist_ok=True)

    output_name = f"results_all_models.json"  # single file for all models
    output_json = os.path.join(output_dir, output_name)

    MODEL_NAME = 'Base_Paligemma'  # default model name, must set env var when running

    print('Output file:', output_json)
    print('Model name:', MODEL_NAME)
    main()
