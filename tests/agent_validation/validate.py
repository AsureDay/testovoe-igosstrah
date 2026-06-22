import asyncio
import os
import sys
import json
import re
import logging
from typing import List, Dict, Any
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.core.inference import InferenceModule, InferenceType
from app.agents.wiki_agent import ReActAgent

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
        response_format = None
        model_name = validator_module.model_name.lower()
        if "gpt" in model_name:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "validation_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "score": {"type": "integer"},
                            "reason": {"type": "string"}
                        },
                        "required": ["score", "reason"],
                        "additionalProperties": False
                    }
                }
            }
        elif "gemini" in model_name or "gemma" in model_name or "deepseek" in model_name or "kimi" in model_name:
            response_format = {"type": "json_object"}

        response = await validator_module.run(query=prompt, response_format=response_format)
        
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
                
    models = [
        "google/gemma-4-31b-it",
        "openai/gpt-5.4-mini",
        "moonshotai/kimi-k2.6",
    ]
    
    PROMPT_BASELINE = (
        "Вы — агент по решению задач. Вы решаете задачу по шагам, используя доступные инструменты.\n"
        "На каждом шаге вы должны вывести строго в формате:\n"
        "Thought: <ваши рассуждения о следующем шаге>\n"
        "Action: <название_инструмента>: <аргументы в формате JSON>\n\n"
        "Доступные инструменты:\n"
        "{tools_text}\n\n"
        "Когда у вас будет окончательный ответ, выведите:\n"
        "Thought: <ваши рассуждения>\n"
        "Action: final_answer: <ваш окончательный ответ пользователю>\n"
    )

    PROMPT_IMPROVED = (
        "Вы — продвинутый аналитический агент по решению сложных многошаговых задач. Вы решаете задачу по шагам, используя доступные инструменты.\n\n"
        "Важные правила:\n"
        "1. Внимательно анализируйте условие задачи: обращайте внимание на скрытые подвохи, крайние случаи и все аспекты вопроса.\n"
        "2. Учитывайте реальные физические, биологические и логические ограничения при расчетах и поиске информации.\n"
        "3. Ваша цель — дать максимально точный и полный ответ на исходный вопрос, доводя рассуждения до логического конца.\n\n"
        "На каждом шаге вы должны вывести строго в формате:\n"
        "Thought: <ваши детальные рассуждения о следующем шаге, критическая оценка найденной информации и промежуточных результатов>\n"
        "Action: <название_инструмента>: <аргументы в формате JSON>\n\n"
        "Доступные инструменты:\n"
        "{tools_text}\n\n"
        "Когда у вас будет окончательный ответ, выведите:\n"
        "Thought: <ваша финальная проверка ответа на соответствие всем условиям задачи>\n"
        "Action: final_answer: <ваш полный и окончательный ответ пользователю>\n"
    )

    prompts = {
        "baseline": PROMPT_BASELINE,
        "improved": PROMPT_IMPROVED
    }
    
    results = {}
    semaphore = asyncio.Semaphore(5)
    
    async def process_test_case(case, agent, inference, prompt_name):
        import uuid
        run_id = uuid.uuid4().hex[:8]
        async with semaphore:
            query = case.get("query")
            reference = case.get("answer")
            print(f"[{run_id}][{prompt_name}] Query: {query}")
            
            agent_answer, used_tools = await agent.run(query, return_tools=True, run_id=run_id)
            print(f"[{run_id}][{prompt_name}] Agent Answer for '{query}': {agent_answer}")
            print(f"[{run_id}][{prompt_name}] Used Tools for '{query}': {used_tools}")
            
            val_result = await evaluate_answer(inference, query, reference, agent_answer)
            print(f"[{run_id}][{prompt_name}] Score: {val_result.score}, Reason: {val_result.reason}")
            
            expected_tools = case.get("expected_tools", [])
            missing_tools = []
            for t in expected_tools:
                if t not in used_tools and not (t=="search_wikipedia" and "wikipedia_search" in used_tools):
                    missing_tools.append(t)
            
            tool_score = 1 if not missing_tools else 0
            tool_reason = "All expected tools used" if not missing_tools else f"Missing tools: {missing_tools}"
            print(f"[{run_id}][{prompt_name}] Tool Score: {tool_score}, Tool Reason: {tool_reason}\n")
            
            ans_short = agent_answer.replace('\n', ' ')
            val_logger.info(f"[{run_id}][{prompt_name}] Ans: {ans_short} | Score: {val_result.score} | ToolScore: {tool_score}")
            
            return {
                "query": query,
                "agent_answer": agent_answer,
                "score": val_result.score,
                "reason": val_result.reason,
                "used_tools": used_tools,
                "expected_tools": expected_tools,
                "tool_score": tool_score,
                "tool_reason": tool_reason
            }

    for model_name in models:
        for prompt_name, prompt_template in prompts.items():
            run_key = f"{model_name}_{prompt_name}"
            print(f"Validating model: {model_name} with prompt: {prompt_name}")
            inference = InferenceModule(model_name=model_name)
            agent = ReActAgent(inference_module=inference, system_prompt_template=prompt_template)
            
            tasks = [process_test_case(case, agent, inference, prompt_name) for case in test_cases]
            model_results = await asyncio.gather(*tasks)
            
            results[run_key] = model_results
        
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
