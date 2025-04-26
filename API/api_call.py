import openai
from config.params import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT

# Set up the OpenAI API client
openai.api_key = AZURE_OPENAI_API_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT

def runner(prompt):
    """Send a prompt to the Azure OpenAI API and return the response."""
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500
    )
    return response["choices"][0]["text"].strip()