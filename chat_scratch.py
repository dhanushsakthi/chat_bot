import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_DIR = "scratch_chatbot"  # folder where we saved the trained model

# ✅ Load the trained model & tokenizer
print("📂 Loading trained chatbot...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

print("✅ Chatbot ready! Type 'quit' to exit.\n")

# ✅ Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("👋 Goodbye!")
        break
   
    # Format the input like training
    prompt = f"User: {user_input}\nBot:"
   
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
   
    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=100,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
   
    # Decode only the new text after the prompt
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
   
    # Extract only bot response
    bot_response = output_text.split("Bot:")[-1].strip()
    print(f"🤖 {bot_response}\n")
