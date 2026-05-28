# token budgeting       --context
#



class TokenBudget:

    def __init__(self, max_tokens_per_request: int=4000):
        self.max_tokens_per_request = max_tokens_per_request
        self.usage = {
            "total_tokens": 0,
            "total_input_tokens": 0,
            "requests_tokens": 0,
        }

    def estimate_tokens(self, text: str) -> int:
        """ rough token estimation (actual would use tiktoken)"""
        return int(len(text.split()) * 1.3)

    def check_budget(self, text: str) -> tuple[bool, int]:
        """ check if budget is within budget."""
        tokens = self.estimate_tokens(text)
        return tokens <= self.max_tokens_per_request, tokens

    def record_usage(self, input_tokens: int, output_tokens: int):
        """record token usage"""
        self.usage["total_input_tokens"] += input_tokens
        self.usage["total_output_tokens"] += output_tokens
        self.usage["requests"] += 1

    def get_stats(self) -> dict:

        pass



class BudgetedLLM:
    def __init__(self, max_budget: int=4000):
        self.llm = None
        self.budget = TokenBudget(max_tokens_per_request=max_budget)

    @traceable(name="budget_model")
    def invoke(self, query: str) -> str:
        # check budget
        within_budget, tokens = self.budget.check_budget(query)

        if not within_budget:
            raise ValueError(f"xxx")



    def get_stats(self) -> dict:
        return self.budget.get_stats()



    















