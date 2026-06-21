import asyncio
import os
import sys
import json
import re
import logging
from typing import List, Dict, Any
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference import InferenceModule, InferenceType
from agents.wiki_agent import ReActAgent

class ValidationResult(BaseModel):
    """
    Result of the LLM validation containing a score and reason.
    """
    score: int
    reason: str

async def evaluate_answer(
    validator_module: InferenceModule,
    query: str,
    reference: str,
    agent_answer: str
) -> ValidationResult:
    """
    Evaluate the agent's answer against the reference answer and return structured feedback.
    """
    prompt = (
        f"Сравни ответ агента с эталонным ответом на вопрос и поставь оценку от 1 до 5.\n"
        f"Вопрос: {query}\n"
        f"Эталонный ответ:\n{reference}\n\n"
        f"Ответ агента:\n{agent_answer}\n\n"
        f"Верни результат в формате JSON:\n"
        f"{{\n"
        f"  \"score\": <оценка от 1 до 5>,\n"
        f"  \"reason\": \"<краткое пояснение>\"\n"
        f"}}\n"
    )
    
    try:
        response = await validator_module.run(query=prompt)
        
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                json_str = response[start:end+1]
            else:
                return ValidationResult(score=1, reason="Failed to parse JSON from response")
                
        data = json.loads(json_str)
        return ValidationResult(
            score=int(data.get("score", 1)),
            reason=data.get("reason", "Parsed successfully")
        )
    except Exception as e:
        return ValidationResult(score=1, reason=f"Validation error: {str(e)}")

async def run_validation():
    """
    Load test cases, execute agent runs for each model, evaluate output, and save results.
    """
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    os.makedirs(log_dir, exist_ok=True)
    for filename in os.listdir(log_dir):
        if filename.endswith(".log"):
            try:
                os.remove(os.path.join(log_dir, filename))
            except OSError:
                pass
    val_log_path = os.path.join(log_dir, "validation.log")
    val_logger = logging.getLogger("ValidationLog")
    if not val_logger.handlers:
        fh = logging.FileHandler(val_log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        val_logger.addHandler(fh)
        val_logger.setLevel(logging.INFO)
    val_logger.propagate = False

    filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "validation_QA.jsonl"))
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
        
    decoder = json.JSONDecoder(strict=False)
    pos = 0
    test_cases = []
    while pos < len(content):
        while pos < len(content) and content[pos].isspace():
            pos += 1
        if pos >= len(content):
            break
        try:
            obj, index = decoder.raw_decode(content, pos)
            test_cases.append(obj)
            pos = index
        except json.JSONDecodeError:
            pos = content.find('{', pos + 1)
            if pos == -1:
                break
                
    models = ["qwen/qwen3.7-plus"]
    
    results = {}
    for model_name in models:
        print(f"Validating model: {model_name}")
        inference = InferenceModule(model_name=model_name)
        agent = ReActAgent(inference_module=inference)
        
        model_results = []
        for case in test_cases:
            query = case.get("query")
            reference = case.get("answer")
            print(f"Query: {query}")
            
            agent_answer = await agent.run(query)
            print(f"Agent Answer: {agent_answer}")
            
            val_result = await evaluate_answer(inference, query, reference, agent_answer)
            print(f"Score: {val_result.score}, Reason: {val_result.reason}\n")
            
            ans_short = agent_answer.replace('\n', ' ')
            val_logger.info(f"Ans: {ans_short} | Score: {val_result.score}")
            
            model_results.append({
                "query": query,
                "agent_answer": agent_answer,
                "score": val_result.score,
                "reason": val_result.reason
            })
        results[model_name] = model_results
        
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "validation_results.json"))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Validation completed. Results saved to {output_path}")

if __name__ == "__main__":
    try:
        asyncio.run(run_validation())
    except BaseException as e:
        import traceback
        print("CRITICAL ERROR: Exception caught at top level:")
        traceback.print_exc()
        raise e
