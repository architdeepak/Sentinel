from llama_cpp import Llama

# Load model once at startup
llm = Llama(
    model_path="/home/archi/Sentinel/modls/granite-3.0-1b-a400m-instruct.Q4_K_M.gguf",
    n_ctx=2048,      # Context window
    n_threads=3,     # Use 2 threads (save 2 for camera!)
    n_gpu_layers=0,  # CPU only on Pi
    verbose=False
)

# Generate response (same interface as OpenAI!)
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant..."},
        {"role": "user", "content": "I'm feeling drowsy"}
    ],
    temperature=0.7,
    max_tokens=100
)

print(response['choices'][0]['message']['content'])