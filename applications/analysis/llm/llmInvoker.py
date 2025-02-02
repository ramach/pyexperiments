#import sqlite3
import json
import configparser
import requests


# Define the LLM endpoint choices
llm_choices = {
    1: "Local LLama",
    2: "Remote LLama",
    3: "Remote Gemini",
    4: "Remote OpenAI",
    5: "Remote Anthropic"
}

def generate_text(prompt, llm_config):
    """Generates text using a local LLama instance through the Ollama tool.

    Args:
        prompt: The text prompt to use for generating text.
        llm_config: The configuration of the llm to use.

    Returns:
        The generated text.
    """

    temperature=0.1
    max_tokens=512
    headers = {"Content-Type": "text/event-stream"}

    url = llm_config['url']
    model_path = llm_config['model_path']

    data = {
        "model": model_path,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()  # Raise an exception for error responses

    responses = ""
    lines = response.text.splitlines()
    for line in lines:
        try:
            data = json.loads(line)
            response = data.get('response')  # Use .get() to handle missing keys safely
            if response:
                responses = responses +  response
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line: {line}")
    print(responses)



def main():
    # Read the config file
    config = configparser.ConfigParser()
    config.read('llm_config.ini')  # Replace with your config file name

    # Get the LLM choices from the config
    llm_choices = {}
    for section in config.sections():
        llm_choices[section] = {
            'url': config.get(section, 'url'),
            'model_path': config.get(section, 'model_path')
        }

    # Prompt the user for their choice
    print("Choose an LLM model:")
    for choice, name in llm_choices.items():
        print(f"{choice}. {name}")

    selected_choice = input("Enter your choice: ")

    # Validate the user's choice
    if selected_choice not in llm_choices:
        print("Invalid choice. Please select a valid option.")
        exit()

    # Get the LLM configuration
    llm_config = llm_choices[selected_choice]

    # Get the user prompt
    prompt = input("Enter your prompt: ")

    # Generate text using the selected LLM
    generate_text(prompt, llm_config)

if __name__ == "__main__":
    main()