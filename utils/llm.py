import os
from langchain_upstage import ChatUpstage

def get_solar_pro(max_token, temperature):
    return ChatUpstage(api_key=os.environ['Upstage_API'], model="solar-pro", max_tokens=max_token, temperature=temperature)