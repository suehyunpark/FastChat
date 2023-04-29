"""
Usage:
python3 -m fastchat.serve.huggingface_api --model ~/model_weights/vicuna-7b/
"""
import argparse
import json

import torch

from fastchat.conversation import get_default_conv_template
from fastchat.serve.inference import load_model, add_model_args


default_retrieved_user_template = """Respond to this question using the given context.
Question:
{question}

Context:
{context}
"""

default_orig_user_template = """Respond to this question.
Question:
{question}
"""

RESERVED_NEW_TOKENS = 512

def get_input(input_file: str):
    with open(input_file) as f:
        items = json.load(f)
    for item in items:
        yield item["question_id"], item["question"], item["context"]
        

default_retrieved_user_template = """Respond to this question using the given context.
Question:
{question}

Context:
{context}
"""

default_orig_user_template = """Respond to this question.
Question:
{question}
"""

RESERVED_NEW_TOKENS = 512

def get_input(input_file: str):
    with open(input_file) as f:
        items = json.load(f)
    for item in items:
        yield item["question_id"], item["question"], item["context"]
        

@torch.inference_mode()
def main(args):
    print("Loading model...")
    model, tokenizer = load_model(
        args.model_path,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        args.cpu_offloading,
        debug=args.debug,
    )
    
    max_context_len = tokenizer.model_max_length
    print(f"Max context length: for {args.model_path}: {max_context_len} tokens")

    responses = []
    for id, question, context in get_input(args.input_file):
        if context:
            msg = default_retrieved_user_template.format(question=question, context=context)
        else:
            msg = default_orig_user_template.format(question=question)
        
        conv = get_default_conv_template(args.model_path).copy()
        conv.append_message(conv.roles[0], msg)  # user prompt
        conv.append_message(conv.roles[1], None)  # assistant's turn
        prompt = conv.get_prompt()

        input_ids = tokenizer(
            [prompt],
            truncation=True,
            max_length=max_context_len - RESERVED_NEW_TOKENS
        )
        print("Generating response...").input_ids
        output_ids = model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=RESERVED_NEW_TOKENS,
            # top_k=4,
            # penalty_alpha=0.6,
        )
    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(input_ids[0]):]
        outputs = tokenizer.decode(output_ids, skip_special_tokens=True,
                                       spaces_between_special_tokens=False)

        print(f"{conv.roles[0]}: {msg}")
        print(f"{conv.roles[1]}: {outputs}")
        responses.append({
            "id": id,
            "prompt": prompt,
            "response": outputs
        })
    
    with open(args.output_file, 'w') as f:
        json.dump(responses, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-file", type=str)
    # parser.add_argument("--message", type=str, default="Hello! Who are you?")
    args = parser.parse_args()

    main(args)
