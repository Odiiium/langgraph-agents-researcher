"""Central configuration for the research pipeline."""

# Models
GPT4_O_MODEL = "gpt-4o"
GPT5_MINI_MODEL = "gpt-5-mini"
GPT5_MODEL = "gpt-5"
O4_MINI_MODEL = "o4-mini"
DEEP_AGENT_MODEL = "openai:gpt-5-mini"

# Pipeline behaviour
TIMEZONE = "Europe/Kyiv"

# Output paths
RESULTS_DIR = "results"
GRAPH_IMAGE_PATH = "graph.png"
LOG_DIR = "logs"

#Guardrails
MAX_QUERY_LENGTH = 500
PLAN_TASKS_SIZE = 5
MAX_ITERATIONS = 3