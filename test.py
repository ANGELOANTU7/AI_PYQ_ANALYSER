import openai

# Set up your OpenAI API key
openai.api_key = "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"

# Use the Completion API to generate text
response = openai.Completion.create(
  engine="text-davinci-002",
  prompt="Hello, world!",
  max_tokens=5
)

# Print the generated text
print(response.choices[0].text)
