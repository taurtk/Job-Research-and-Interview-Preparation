import os
import streamlit as st
from crewai import Task, Crew, Agent
from langchain_groq import ChatGroq
from crewai_tools import ScrapeWebsiteTool, SerperDevTool

# Set up environment variables for API keys
os.environ["SERPER_API_KEY"] = "7c748181570a86f0b2f5d26c92dd4c92ad35e7ae"
os.environ["GROQ_API_KEY"] = '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'  # Replace with your actual Groq API key

# Initialize the tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Initialize LLM with Groq provider and model
llm = ChatGroq(temperature=0, model_name="groq/gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"])

# Create the Suggestion Agent
suggestion_agent = Agent(
    role="Interview Tips and Roadmap Advisor",
    goal="Prepare suggestions and a roadmap for interview preparation based on user interests.",
    backstory="You provide tailored advice and resources to help candidates excel in their interview preparations.",
    allow_delegation=False,
    tools=[search_tool,scrape_tool],
    llm=llm,
    verbose=True,
)

def prepare_suggestions_task(user_interest):
    return Task(
        description=f"Prepare a roadmap and suggestions for interview preparation based on {user_interest}.",
        agent=suggestion_agent,
        expected_output=f"A tailored roadmap and interview tips for {user_interest}.",
        async_execution=True
    )

# Main Streamlit app function
def main():
    st.title("Job Research and Interview Preparation")
    
    user_interest = st.text_input("Enter your area of interest:")
    
    if st.button("Start Research"):
        # Show a spinner while processing
        with st.spinner("Processing... Please wait."):
            # Initialize Crew with tasks
            crew = Crew(
                agents=[suggestion_agent],
                tasks=[prepare_suggestions_task(user_interest)],
                verbose=True
            )
            
            # Start the crew execution
            crew_output = crew.kickoff()
        
        # Displaying outputs using Streamlit after processing is complete
        if crew_output.raw:
            st.subheader("Raw Output")
            st.write(crew_output.raw)
        
        if crew_output.json_dict:
            st.subheader("JSON Output")
            st.json(crew_output.json_dict)

if __name__ == "__main__":
    main()