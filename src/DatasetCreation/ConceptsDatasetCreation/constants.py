import logging
import os

logger = logging.getLogger(__name__)

# dataset generator
generation_model_name = "gpt-4o"

# dataset classifier
classification_model_name = "gpt-4o"
classification_model_temperature = 1e-8
# classification_model_top_p = 1e-8

# number of examples to be generated and classified
estimated_total_num_examples = 1000

# maximum amount spent per API call
max_API_cost = 50


try:
    openai_api_key = os.environ["OPENAI_API_KEY"]
    openai_org_id = os.environ["OPENAI_ORG_ID"]
except KeyError:
    logger.error("Either OPENAI_API_KEY or OPENAI_ORG_ID not found in the environment")
