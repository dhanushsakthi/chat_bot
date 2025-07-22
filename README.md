# Chatbot 🤖

A custom-trained conversational AI chatbot built from scratch using GPT-2 architecture. Features both command-line interface and web-based UI powered by Streamlit.

## 🌟 Features

- **Custom Trained Model**: Fine-tuned GPT-2 model for conversational responses
- **Dual Interface**: Both CLI and web-based chat interfaces
- **GPU Acceleration**: CUDA support for faster inference
- **Real-time Chat**: Interactive conversation with context awareness
- **Modern UI**: Clean, user-friendly Streamlit web interface
- **Session Management**: Chat history preservation in web UI

## 🏗️ Project Structure

```
chatbot-scratch/
├── app.py                 # Streamlit web interface
├── chat_scratch.py        # Command-line interface
├── scratch_chatbot/       # Trained model directory
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── vocab.json
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- PyTorch
- Transformers library
- Streamlit (for web UI)
- CUDA-compatible GPU (optional, for faster performance)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/chatgpt-scratch-chatbot.git
   cd chatgpt-scratch-chatbot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files are present:**
   Make sure the `scratch_chatbot/` directory contains your trained model files.

## 🖥️ Usage

### Command Line Interface

Run the CLI version for a simple terminal-based chat:

```bash
python chat_scratch.py
```

**Features:**
- Direct terminal interaction
- Type `quit` or `exit` to end conversation
- Lightweight and fast

**Example:**
```
📂 Loading trained chatbot...
✅ Chatbot ready! Type 'quit' to exit.

You: Hello, how are you?
🤖 Hi there! I'm doing well, thank you for asking. How can I help you today?

You: quit
👋 Goodbye!
```

### Web Interface

Launch the Streamlit web application:

```bash
streamlit run app.py
```

**Features:**
- Clean, modern web interface
- Chat history preservation
- Real-time responses
- Clear chat functionality
- Mobile-friendly design

**Interface Elements:**
- 📱 Responsive chat interface
- 👤 User message display
- 🤖 Bot response display
- 🔄 Send button for message submission
- 🧹 Clear chat history option

## ⚙️ Technical Details

### Model Configuration

- **Base Model**: GPT-2 (Fine-tuned)
- **Framework**: PyTorch + Transformers
- **Generation Parameters**:
  - `max_length`: 100 tokens
  - `temperature`: 0.7
  - `top_k`: 50
  - `top_p`: 0.95
  - `do_sample`: True

### Performance Optimization

- **GPU Acceleration**: Automatic CUDA detection
- **Model Caching**: Streamlit resource caching for faster loading
- **Efficient Inference**: `torch.no_grad()` for memory optimization

## 🔧 Configuration

### Generation Parameters

You can modify the response generation by adjusting these parameters in both files:

```python
output_ids = model.generate(
    input_ids,
    max_length=100,      # Maximum response length
    temperature=0.7,     # Creativity (0.1-1.0)
    top_k=50,           # Top-k sampling
    top_p=0.95,         # Nucleus sampling
    do_sample=True      # Enable sampling
)
```

### Model Directory

Update the `MODEL_DIR` variable to point to your trained model:

```python
MODEL_DIR = "scratch_chatbot"  # Your model directory
```

## 📋 Requirements

Create a `requirements.txt` file with:

```txt
torch>=1.9.0
transformers>=4.21.0
streamlit>=1.25.0
numpy>=1.21.0
```

## 🏋️ Model Training

This project assumes you have a pre-trained model in the `scratch_chatbot/` directory. If you need to train your own model, ensure you have:

- Training dataset in conversational format
- Proper tokenization setup
- Fine-tuning script for GPT-2

**Expected Model Files:**
- `config.json` - Model configuration
- `pytorch_model.bin` - Model weights
- `tokenizer.json` - Tokenizer configuration
- `vocab.json` - Vocabulary mapping

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py --server.port 8501
```

### Production Deployment

**Streamlit Cloud:**
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Deploy directly from repository

**Docker Deployment:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

## 🎯 Use Cases

- **Personal Assistant**: Answer questions and provide assistance
- **Educational Tool**: Learn about conversational AI
- **Prototype Development**: Base for more complex chatbot applications
- **Research**: Experiment with different generation parameters

## 🔍 Troubleshooting

### Common Issues

1. **Model Not Found Error:**
   ```bash
   # Ensure scratch_chatbot directory exists with model files
   ls scratch_chatbot/
   ```

2. **CUDA Out of Memory:**
   ```python
   # Use CPU instead
   device = "cpu"
   ```

3. **Streamlit Port Issues:**
   ```bash
   streamlit run app.py --server.port 8502
   ```

## 📈 Performance Metrics

- **Response Time**: ~1-3 seconds (GPU) / ~5-10 seconds (CPU)
- **Memory Usage**: ~2-4 GB (depending on model size)
- **Concurrent Users**: Suitable for single-user applications

## 🔮 Future Enhancements

- [ ] **Context Memory**: Implement conversation history retention
- [ ] **Multi-turn Conversations**: Better context understanding
- [ ] **Custom Training Pipeline**: End-to-end training script
- [ ] **API Endpoint**: REST API for integration
- [ ] **Voice Interface**: Speech-to-text and text-to-speech
- [ ] **Personality Customization**: Different chatbot personalities
- [ ] **Database Integration**: Store conversation logs
- [ ] **Multi-language Support**: Support for multiple languages

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face Transformers** - For the excellent NLP library
- **OpenAI** - For the GPT-2 architecture
- **Streamlit** - For the intuitive web framework
- **PyTorch** - For the deep learning framework

## 👨‍💻 Author

Created by [Dhanush.S] - [s46dhanush2005@gmail.com]

## 📞 Support

If you encounter any issues or have questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the code comments for implementation details

