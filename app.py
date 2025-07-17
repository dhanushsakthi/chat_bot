import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

@st.cache_resource
def load_model():
    MODEL_DIR = "scratch_chatbot"
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

def get_bot_response(user_input):
    prompt = f"User: {user_input}\nBot:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

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

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    bot_response = output_text.split("Bot:")[-1].strip()
    return bot_response

st.set_page_config(page_title="ChatGPT Scratch Bot", page_icon="🤖")
st.title("🤖 ChatGPT Scratch - Web UI")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    st.markdown(f"**👤 You:** {chat['user']}")
    st.markdown(f"**🤖 Bot:** {chat['bot']}")

user_input = st.text_input("You:", placeholder="Type your message here...")

if st.button("Send") and user_input:
    bot_response = get_bot_response(user_input)
    st.session_state.chat_history.append({"user": user_input, "bot": bot_response})
    st.experimental_rerun()

if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()
