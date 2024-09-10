import pandas as pd
from langchain import LLMChain, PromptTemplate
from langchain.llms import Ollama

# Load your CSV data into a pandas DataFrame
def load_csv_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Generate insights using LangChain and Ollama
def generate_insights(df):
    # Convert the dataframe to a string for input into the model
    data_str = df.to_string()

    # Set up the Ollama LLM and the prompt template
    ollama_llm = Ollama()

    # Define the template for the prompt
    prompt_template = PromptTemplate(
        template="""
        Given the following dataset:
        
        {data_str}
        
        Provide key insights and analysis about the data:
        """
    )

    # Create the LLM chain with the prompt
    chain = LLMChain(llm=ollama_llm, prompt=prompt_template)

    # Run the chain with the CSV data
    insights = chain.run(data_str=data_str)

    return insights

# Main function to load CSV and generate insights
if __name__ == "__main__":
    file_path = "data.csv"  # Replace with your CSV file path
    df = load_csv_data(file_path)
    insights = generate_insights(df)
    print("Generated Insights:\n", insights)