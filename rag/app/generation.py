from __future__ import annotations

import asyncio
import json
import re

import httpx

from app.schemas import GeneratedAnswer

SYSTEM_PROMPT = """You are a grounded retrieval-augmented answering system. Answer only using the supplied CONTEXT. Do not invent facts or use outside knowledge. If the context does not contain enough evidence to answer the question, reply with JSON {\"answer\": \"संदर्भ में पर्याप्त जानकारी नहीं है\", \"grounded\": false}. Otherwise return a short factual answer in Hindi and set grounded=true. Return valid JSON only. Do not wrap the JSON in markdown fences or prose."""


class Generator:
    def __init__(self, base_url: str | None, api_key: str | None, model: str | None, timeout: float):
        self.base_url, self.api_key, self.model, self.timeout = base_url, api_key, model, timeout

    @staticmethod
    def _extract_json(content: str) -> dict:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError("No JSON object found in model output")

    async def answer(self, query: str, context: str) -> GeneratedAnswer:
        if not all([self.base_url, self.api_key, self.model]):
            return GeneratedAnswer(answer="संदर्भ में पर्याप्त जानकारी नहीं है।", grounded=False)

        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 180,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"QUESTION:\n{query}\n\nCONTEXT:\n{context}"},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(f"{self.base_url.rstrip('/')}/chat/completions", json=payload, headers=headers)
                    response.raise_for_status()

                content = response.json()["choices"][0]["message"].get("content", "")
                if not content:
                    raise ValueError("Empty model response content")

                result = GeneratedAnswer.model_validate(self._extract_json(content))
                if not result.grounded and not result.answer:
                    result.answer = "संदर्भ में पर्याप्त जानकारी नहीं है।"
                return result
            except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                if attempt == 0:
                    await asyncio.sleep(0.15)

        return GeneratedAnswer(answer="संदर्भ में पर्याप्त जानकारी नहीं है।", grounded=False)
