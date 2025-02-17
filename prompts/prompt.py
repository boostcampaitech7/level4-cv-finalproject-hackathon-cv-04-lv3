from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

def load_template(template_name, input_text):
    template_path = f"prompts/{template_name}.md"
    with open(template_path, "r", encoding="utf-8") as file:
        template = file.read()
    return template.format(input=input_text)