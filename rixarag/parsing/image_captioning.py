from rixarag import settings
import os
from tqdm import tqdm

def caption_images(processed_chunks):
    process_kwargs = {}
    if settings.IMAGE_CAPTIONING_BACKEND == "openai":
        from openai import OpenAI
        if settings.OPENAI_API_KEY:
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
        else:
            client = OpenAI()
        process_kwargs["openai_client"] = client
        get_image_transcription = get_image_transcription_openai
    else:
        #https://llama-cpp-python.readthedocs.io/en/latest/#multi-modal-models
        raise NotImplementedError("Only OpenAI is currently supported as a backend for image captioning")

    print("Starting transcription of document images")
    count = 0
    total_token_count = 0
    for chunk in tqdm(processed_chunks, desc="Processing images"):
        if "images" in chunk:
            for image in chunk["images"]:
                transcript, token_count = get_image_transcription(image["base64"], image["start_context"],
                                                                  image["end_context"], process_kwargs)
                image["transcription"] = transcript
                image["transcript_tokens"] = token_count
                count+=1
                total_token_count += token_count
    skip_count = sum([1 for chunk in processed_chunks for image in chunk.get("images", []) if image["transcription"] == "SKIP"])
    for chunk in processed_chunks:
        if "images" in chunk:
            chunk["images"] = [image for image in chunk["images"] if image["transcription"] != "SKIP"]
        if "images" in chunk and len(chunk.get("images", [])) == 0:
            del chunk["images"]
    print(f"Transcribed {count} images while using a total of {total_token_count} tokens")
    if skip_count > 0:
        print(f"{skip_count} images were discarded due to being marked as irrelevant")
    return processed_chunks, count, total_token_count


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
