from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict
from PIL import Image
import io

from slimgest.cli.process.image_process import DeepseekOCRProcessor
from slimgest.cli.local import PROMPT, CROP_MODE, clean_formula
from vllm import SamplingParams

app = FastAPI(
    title="slimgest FastAPI VLM Server",
    description="OpenAI-compatible VLM server endpoint using slimgest local pipeline.",
)

def extract_images_and_prompts(payload: Dict) -> List[Dict]:
    """
    Parses the OpenAI-compatible VLM chat API payload to extract images and associate prompts.
    """
    messages = payload.get("messages", [])
    results = []
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", [])
            prompt_text = ""
            images = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = part.get("image_url", {})
                    image_b64 = url.get("data", None)
                    if image_b64:
                        images.append(image_b64)
                    # If the reference is a URL, unsupported in offline/slimgest local mode.
                elif isinstance(part, dict) and part.get("type") == "text":
                    prompt_text += part.get("text", "")
                elif isinstance(part, str):
                    prompt_text += part
            if images:
                results.append({
                    "prompt": prompt_text.strip() or PROMPT,
                    "images_b64": images
                })
    return results

import base64

def decode_images_b64(images_b64: List[str]) -> List[Image.Image]:
    images = []
    for imgstr in images_b64:
        try:
            img_bytes = base64.b64decode(imgstr)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            images.append(image)
        except Exception as e:
            continue
    return images

@app.post("/v1/chat/completions")
async def vlm_chat_completions(request: Request):
    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    # Parse messages to get image(s) and prompt
    parsed_list = extract_images_and_prompts(payload)
    if not parsed_list:
        return JSONResponse({"error": "No user-supplied image found in messages."}, status_code=400)
    responses = []
    for entry in parsed_list:
        images = decode_images_b64(entry["images_b64"])
        if not images:
            responses.append({"error": "Failed to decode image."})
            continue
        # Use only the first image for now (extend to batch if needed)
        image = images[0]
        # Prepare input for processor
        cache_item = {
            "prompt": entry["prompt"],
            "multi_modal_data": {
                "image": DeepseekOCRProcessor().tokenize_with_images(images=[image], bos=True, eos=True, cropping=CROP_MODE)
            }
        }
        # NOTE: This is a synchronous simplification: No batching, sequential call.
        # Choose default sampling params -- for OpenAI-style API, make temperature, max_tokens, etc. configurable via POST
        sampling_params = SamplingParams(temperature=0.0, max_tokens=2048, skip_special_tokens=True)
        # Infer: use your local pipeline. Here, for simplicity, we invoke just the processor and say the answer is a dummy string,
        # but adapt to match your pipeline.
        # [Insert real OCR logic here as done in local.py, perhaps calling llm.generate([...], ...)]
        try:
            # This is only the tokenization and prompt prep; inference would require access to the loaded llm and actual run.
            # If available globally, use it; otherwise, return dummy for this example.
            from slimgest.cli.local import llm  # loaded on startup in local.py
            output = llm.generate([cache_item], sampling_params)[0].outputs[0].text
            output = clean_formula(output)
        except Exception as e:
            output = f"[ERROR running OCR engine: {e}]"
        responses.append({
            "role": "assistant",
            "content": output
        })
    # Respond in OpenAI Chat Completions format
    # In OpenAI API, the 'choices' field is a list of responses; we map each entry as a choice
    # The id/created/model fields can be dummies
    resp_json = {
        "id": "chatcmpl-slimgest",
        "object": "chat.completion",
        "created": int(__import__("time").time()),
        "model": payload.get("model", "deepseek-ocr"),
        "choices": [
            {
                "index": idx,
                "message": resp,
                "finish_reason": "stop"
            }
            for idx, resp in enumerate(responses)
        ]
    }
    return JSONResponse(content=resp_json)

