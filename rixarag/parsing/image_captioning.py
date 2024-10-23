from rixarag import settings
import os
from tqdm import tqdm

def caption_images(processed_chunks):
    process_kwargs = {}
    if settings.IMAGE_CAPTIONING_BACKEND == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        process_kwargs["openai_client"] = client
        get_image_transcription = get_image_transcription_openai
    else:
        raise NotImplementedError("Only OpenAI is currently supported as a backend for image captioning")

    print("Starting transcription of document images")
    count = 0
    total_token_count = 0
    for chunk in tqdm(processed_chunks, desc="Processing Chunks"):
        if "images" in chunk:
            for image in chunk["images"]:
                transcript, token_count = get_image_transcription(image["base64"], image["start_context"],
                                                                  image["end_context"])
                image["transcription"] = transcript
                image["transcript_tokens"] = token_count
                count+=1
                total_token_count += token_count


def get_image_transcription_openai(b64image_str, start_context, end_context, process_kwargs):
    client = process_kwargs["openai_client"]
    response = client.chat.completions.create(
        model=settings.OPENAI_IMAGE_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": settings.IMAGE_CAPTIONING_INSTRUCTION_TEMPLATE.format(
                            start_context=start_context, end_context=end_context
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64image_str}",
                            "detail": "low"
                        },
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content, response.usage.total_tokens
