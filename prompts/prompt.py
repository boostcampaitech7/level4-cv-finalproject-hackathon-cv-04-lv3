from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

def load_template(template_name):
    template_path = f"prompts/{template_name}.md"
    with open(template_path, "r", encoding="utf-8") as file:
        template = file.read()
    return template

def one(input_text):
    prompt = load_template("test1")
    return prompt.format(input=input_text)

def extract_rss_content_prompt(input_text):
    prompt = load_template("extract_rss_content_prompt")
    return prompt.format(content=input_text)