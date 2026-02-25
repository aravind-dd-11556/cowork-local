from openai import OpenAI
import json

def main():
    # client = OpenAI(
    #     base_url="https://integrate.api.nvidia.com/v1",
    #     api_key="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"
    #
    # )
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="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"

    )

    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
            Return a JSON object with 'title' and 'summary' keys.
            For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
            For the summary: Create a concise summary of the main points in this chunk.
            Keep both title and summary concise but informative."""

    ##qwen
    completion = client.chat.completions.create(
        model="qwen/qwen2.5-7b-instruct",
        messages=[{"role": "system", "content": system_prompt},{"role": "user", "content": "Write a limerick about the wonders of GPU computing."}],
        stream=False
    )



    # response = client.embeddings.create(
    #
    #
    #     input=["What is the capital of France?"],
    #     model="baai/bge-m3",
    #     encoding_format="float",
    #     extra_body={"truncate": "NONE","input_type": "query"}
    # )

    # completion = client.chat.completions.create(
    #     model="nvidia/llama-3.1-nemotron-70b-instruct",
    #     messages=[{"role": "system", "content": system_prompt}, {"role": "user",
    #                "content": "Can you speculate on the potential impact of a recession on ABCs business?"}],
    #     # temperature=0.5,
    #     # top_p=1,
    #     # max_tokens=1024
    # )
    if isinstance(completion, tuple):
        response_text = "".join(chunk for chunk in completion if chunk)
    else:
        response_text = completion.choices[0].message.content
    #print(completion.choices[0].message)
    try:
       print(json.loads(response_text))
    except json.JSONDecodeError:
        return {"title": "Error parsing JSON", "summary": "Response not in JSON format"}

if __name__ == "__main__":
    main()