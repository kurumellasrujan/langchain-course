from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
import httpx

load_dotenv()
def main():
    print("Hello from langchain-course!")

    information = """
    Ratan Naval Tata[a] (28 December 1937 – 9 October 2024) was an Indian industrialist and philanthropist. He served as the chairman of Tata Group and Tata Sons from 1991 to 2012, and he held the position of interim chairman from October 2016 to February 2017.[3][4] In 2000, he received the Padma Bhushan, the third-highest civilian honour in India, followed by the Padma Vibhushan, the country's second-highest civilian honour, in 2008.[5]

Ratan Tata was the son of Naval Tata, who was adopted by Ratanji Tata, son of Jamsetji Tata, the founder of the Tata Group. He graduated from Cornell University College of Architecture with a bachelor's degree in architecture.[6] He had also attended the Harvard Business School (HBS) Advanced Management Program in 1975.[2] He joined the Tata Group in 1962,[7] starting on the shop floor of Tata Steel. He later succeeded JRD Tata as chairman of Tata Sons upon the latter's retirement in 1991. During his tenure, the Tata Group acquired Tetley, Jaguar Land Rover, and Corus, in an attempt to turn Tata from a largely India-centric group into a global business.

Throughout his life, Tata invested in over 40 start-ups, primarily in a personal capacity, with additional investments through his firm, RNT Capital Advisors.
    """
    
    system_template = """
    Given the information {information} about a person I want you to create:
    1. A short summary
    2. Two interesting facts about them
    """

    http_client = httpx.Client(verify=False)
    prompt_template = PromptTemplate(input_variables=["information"], template=system_template)

    #llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, http_client=http_client)
    llm = ChatOllama(model="gemma3:270m", temperature=0, http_client=http_client)
    chain = prompt_template | llm # LCEL - LangChain Expression Language
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
