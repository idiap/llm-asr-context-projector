# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Esaú Villatoro-Tello <esau.villatoro@idiap.ch>
# SPDX-License-Identifier: MIT
#
# Step 3: extract the salient keywords of every utterance with an LLM (Section 2.4 of the paper).
#
# The paper uses Gemma 3 27B served locally by Ollama, through the sdialog library:
#   pip install sdialog==0.4.4
#   ollama serve &          # in another shell, or as a background job
#   ollama pull gemma3:27b
#   python data_prep/extract_keywords.py --inp train.jsonl --out train.kw.jsonl
#
# Every output entry gets a "keywords" field (upper-cased, in order of appearance).

import argparse
import json

import sdialog
from jinja2 import Template
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sdialog.config import config
from sdialog.util import get_llm_model
from tqdm import tqdm


SYSTEM_PROMPT = """You are a concise assistant that extracts important keywords (single- and multi-word keyphrases) from conversation transcripts to power a memory mechanism.

Requirements:
- Return ONLY valid JSON. Do not include Markdown code fences, explanations, or extra text.
- Output format (exactly): {"keywords": ["keyword1", "keyword2", ...]}.
- Order: preserve the order of first appearance in the transcript.
- Granularity: short, specific noun phrases and named entities (1-4 words). Keep multi-word phrases intact.
- Include: participant names, organizations, products, locations, dates/times, amounts/IDs, tasks, decisions, issues, intents, salient topics.
- Exclude: stopwords and fillers (e.g., "uh", "um"), function words, generic terms without context (e.g., "thing", "stuff"), pleasantries/greetings, disfluencies, punctuation-only tokens.
- Deduplicate: case-insensitive; keep only the first occurrence.
- Normalization: preserve original wording and casing; do not invent, expand, or infer information not present in the transcript.
- Quantity: only the most relevant keywords; if none are present, return {"keywords": []}.

Output JSON schema:
{"keywords": ["keyword1", "keyword2", ...]}
"""

INSTRUCTION = "Extract all relevant keywords as they appear, and in order, from the following conversation transcript:\n\n{{transcript}}\n\nReturn only the JSON object."


class LLMKeywordExtractor:
    def __init__(self, instruction: str = None, system_prompt: str = None, model_name: str = None, output_format: BaseModel = None, **llm_kwargs):

        llm_config_params = {k: v for k, v in config["llm"].items() if k != "model" and v is not None}
        llm_kwargs = {**llm_config_params, **llm_kwargs}
        if model_name is None:
            model_name = config["llm"]["model"]

        if output_format is None:
            class KeywordExtractionOutput(BaseModel):
                keywords: list[str] = Field(..., description="List of extracted keywords")
            output_format = KeywordExtractionOutput

        self.output_format = output_format
        self.instruction = instruction or INSTRUCTION
        self.llm = get_llm_model(model_name, output_format=output_format, **llm_kwargs)
        self.memory = [SystemMessage(content=system_prompt or SYSTEM_PROMPT), HumanMessage(content="")]

    def __call__(self, transcript: str) -> BaseModel:
        self.memory[1].content = Template(self.instruction).render(transcript=transcript)
        response = self.llm.invoke(self.memory)
        return self.output_format.model_validate(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add LLM-extracted keywords to every utterance of a JSONL file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inp", required=True, help="input JSONL (output of add_dialogue_context.py)")
    parser.add_argument("--out", required=True, help="output JSONL")
    parser.add_argument("--llm", default="ollama:gemma3:27b", help="sdialog LLM identifier")
    args = parser.parse_args()

    sdialog.config.llm(args.llm)
    keyword_extractor = LLMKeywordExtractor()

    with open(args.inp) as f_in, open(args.out, "w") as f_out:
        for line in tqdm(f_in, desc="Extracting keywords"):
            data = json.loads(line)
            try:
                data["keywords"] = [k.upper() for k in keyword_extractor(data["target"]).keywords]
            except Exception as e:
                print(f"Error extracting keywords from {data['key']}: {e}")
                data["keywords"] = []
            f_out.write(json.dumps(data) + "\n")
            f_out.flush()
