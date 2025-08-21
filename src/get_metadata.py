import json
import os
from openai import OpenAI

# CONFIGURATION
input = 'data/chunks/chunked.jsonl'
output = 'data/chunks/chunked_wMetadata.jsonl'

# Ensure the OpenAI API key is set in your environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPEN_AI_MODEL = "gpt-4o-mini-2024-07-18"
client = OpenAI(api_key=OPENAI_API_KEY)
tool = [{
    "type": "function",
    "name": "extract_metadata",
    "description": "Extract topics and entities from a text chunk.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "topics": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Key themes or topics in the text"
            },
            "entities": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Specific named entities (organizations, dates, etc.)"
            }
        },
        "required": ["topics", "entities"],
        "additionalProperties": False
    }
}]

def call_extraction_model(chunk_id: str, text: str):
    prompt = [
        {
            "role": "system",
            "content": (
                "You’re an assistant that extracts topics and entities from a text chunk extracted from a pdf."
                "Always respond by calling the function."
            )
        },
        {
            "role": "user",
            "content": f"Chunk id: {chunk_id}\nText:\n{text}"
        }
    ]

    response = client.responses.create(
        model=OPEN_AI_MODEL,
        input=prompt,
        tools=tool,
    )

    call = response.output[0]
    args = json.loads(call.arguments)
    return args["topics"], args["entities"]

if __name__ == "__main__":
    with open(input, 'r') as infile, open(output, 'w') as outfile:
        for i, line in enumerate(infile):
            if i < 2832:
                continue
            data = json.loads(line)
            data['topics'], data['entities'] = call_extraction_model(data['chunk_id'], data['text'])
            json.dump(data, outfile)
            outfile.write('\n')
        print(f"Metadata extraction completed. Output saved to {output}.")
