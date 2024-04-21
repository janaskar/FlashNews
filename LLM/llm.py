# From
# https://medium.com/predict/a-simple-comprehensive-guide-to-running-large-language-models-locally-on-cpu-and-or-gpu-using-c0c2a8483eee

import random
import threading
import webbrowser

from flask import Flask, request, render_template_string
# Imports
# from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Define model name and file name
# model_name = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"
# model_file = "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
# https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf?download=true

# Use the following model_name and model_file if you have 8gb ram or less
# model_name = "TheBloke/Mistral-7B-OpenOrca-GGUF"
# model_file = "mistral-7b-openorca.Q4_K_M.gguf"

# Use the following model_name and model_file if you have 16gb ram or less
# model_name = "TheBloke/vicuna-13B-v1.5-16K-GGUF"
# model_file = "vicuna-13b-v1.5-16k.Q4_K_M.gguf"

# https://huggingface.co/TheBloke/vicuna-13B-v1.5-16K-GGUF/resolve/main/vicuna-13b-v1.5-16k.Q4_K_M.gguf?download=true

model_path = "Model/mistral-7b-openorca.Q4_K_M.gguf"

model_kwargs = {
    "n_ctx": 4096,  # Context length to use
    "n_threads": 6,  # Number of CPU threads to use
    "n_gpu_layers": 0,  # Number of model layers to offload to GPU. Set to 0 if only using CPU
}

# Instantiate model from downloaded file
llm = Llama(model_path=model_path, **model_kwargs)

# Generation kwargs
generation_kwargs = {
    "max_tokens": 256,  # Max number of new tokens to generate
    "stop": ["<|endoftext|>", "</s>"],  # Text sequences to stop generation on
    "echo": False,  # Echo the prompt in the output
    "top_k": 5
    # This is essentially greedy decoding, since the model will always return the highest-probability token. Set this
    # value > 1 for sampling decoding
}

app = Flask(__name__)

# HTML template for the form
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Message Form</title>
</head>
<body>
  <h2>Enter your message:</h2>
  <form method="post" action="/submit">
    <input type="text" name="message" placeholder="Type your message here" required>
    <input type="submit" value="Submit">
  </form>
  {% if prompt %}
    <h3>Prompt:</h3>
    <p>{{ prompt }}</p>
  {% endif %}
  {% if response %}
    <h3>Response:</h3>
    <p>{{ response }}</p>
  {% endif %}
</body>
</html>
"""


@app.route('/', methods=['GET'])
def index():
    # Display the form with the prompt message
    prompt_message = "Type a message and submit to get a response."
    return render_template_string(HTML_TEMPLATE, prompt=prompt_message)


@app.route('/submit', methods=['POST'])
def submit():
    # Get the message from the submitted form
    message = request.form['message']
    prompt = f"{message}"

    # Generate response using llm
    res = llm(prompt, **generation_kwargs)

    # Render the form again, along with the response and prompt
    return render_template_string(HTML_TEMPLATE, prompt=prompt, response=res["choices"][0]["text"])


if __name__ == '__main__':
    port = 5000 + random.randint(0, 999)
    url = "http://127.0.0.1:{0}".format(port)
    threading.Timer(1.25, lambda: webbrowser.open(url) ).start()
    app.run(port=port, debug=False)

"""
# Start an infinite loop to continuously prompt for user input
while True:
    try:
        # Get input from the user
        user_input = input("Enter your input (or press CTRL+C to exit): ")
        prompt = f"{user_input}"
        # prompt = "Does Eliot hate mexicans?"

        # Run inference with the language model
        res = llm(prompt, **generation_kwargs)  # Res is a dictionary

        # Print the prompt and the generated text
        print("Prompt: " + prompt)
        print(res["choices"][0]["text"])

    except KeyboardInterrupt:
        # Handle the CTRL+C keyboard interrupt and exit gracefully
        print("\nExiting the program. Goodbye!")
        break
"""
