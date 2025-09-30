from openai import OpenAI
import json
import os
# Get the API key from environment variable
api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)
def check_answer(question: str, potential_solution: str, correct_solution: str):
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",  
        messages=[
            {"role": "system", "content": "You are a strict answer checker. Only check correctness."},
            {"role": "user", "content": f"Question: {question}\nAnswer: {potential_solution}\nCorrect: {correct_solution}"}
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "evaluate_answer",
                    "description": "Check if the given answer matches the correct solution",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "is_correct": {
                                "type": "boolean",
                                "description": "True if the potential solution is correct, otherwise False"
                            },
                            "feedback": {
                                "type": "string",
                                "description": "Brief explanation why the answer is correct or not"
                            }
                        },
                        "required": ["is_correct"]
                    }
                }
            }
        ],
        tool_choice="auto"
    )

    # Extract function call arguments
    tool_call = completion.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    return args['is_correct']  # dictionary with is_correct + feedback


result = check_answer(
    question="What is 2 + 2?",
    potential_solution="4",
    correct_solution="4"
)

print(result)
