import os
import asyncio
from browser_use.llm.google.chat import ChatGoogle
from browser_use import Agent

# Ensure the API key is set
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY environment variable not found.")
    exit(1)

# Initialize the Native Browser-Use Gemini model
llm = ChatGoogle(
    model="gemini-2.0-flash",
    api_key=api_key
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
