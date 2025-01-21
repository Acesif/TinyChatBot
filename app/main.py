import uvicorn

from model_utils import load_model, load_tokenizer
from setup import setup
from config import DEVICE
from fastapi import FastAPI
from app.model import Request, Response

setup()

model = load_model()
tokenizer = load_tokenizer()

app = FastAPI()

@app.post("/chat")
def chat(request: Request) -> Response:
    """
    Run chatbot interaction in a loop.
    """
    try:
        if request.prompt.lower() in ["exit", "quit"]:
            print("Exiting chat.")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful university lecturer. Explain all the jargon"
            },
            {
                "role": "user",
                "content": request.prompt
            }
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(DEVICE)

        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=512
        )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(model_inputs.input_ids, generated_ids)]

        response = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return Response(
            status=200,
            message="Success",
            data=response
        )

    except Exception as e:
        return Response(
            status=500,
            message=str(e),
            data={}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)