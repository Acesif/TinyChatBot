from model_utils import load_model, load_tokenizer
from setup import setup
from config import DEVICE

setup()

model = load_model()
tokenizer = load_tokenizer()


def chat():
    """Run chatbot interaction in a loop."""
    while True:
        prompt = input("Ask: ")

        if prompt.lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break

        messages = [
            {"role": "system", "content": "You are a helpful university lecturer. Explain all the jargon"},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(DEVICE)

        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=512
        )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print("\nAI:", response, "\n")

if __name__ == "__main__":
    chat()