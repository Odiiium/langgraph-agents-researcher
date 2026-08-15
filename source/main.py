from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler

from .models import ResearchState
from .pipeline import build_research_pipeline


def main():
    load_dotenv()

    langfuse_handler = CallbackHandler()

    pipeline = build_research_pipeline().build()
    pipeline.save_image()

    return pipeline.run(
        ResearchState(
            user_query=(
                "Research possible bitcoin fluctuations and market moves in "
                "meantime. Give an answer in which side is better to invest right now"
            ),
            current_time=None,
            iteration_count=0,
            plan=None,
            research=None,
            analysis=None,
            check_result=None,
            final_answer=None,
            guardrail_violation=None,
        ),
        callbacks=[langfuse_handler],
    )


if __name__ == "__main__":
    main()
