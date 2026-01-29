import os
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent

# Ensure the API key is set
if "GEMINI_API_KEY" not in os.environ:
    print("Error: GEMINI_API_KEY environment variable not found.")
    exit(1)

class ChatGoogleGenerativeAIWrapper(ChatGoogleGenerativeAI):
    model_config = {"extra": "allow"}

    @property
    def provider(self):
        return "google"

    @property
    def model_name(self):
        return self.model

# Initialize the Gemini model
llm = ChatGoogleGenerativeAIWrapper(
    model="gemini-2.0-flash",
    google_api_key=os.environ["GEMINI_API_KEY"]
)

async def main():
    agent = Agent(
        task="Find the cheapest one-way flight from New York (JFK) to London (LHR) for next Wednesday.",
        llm=llm,
    )
    result = await agent.run()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
