import streamlit as st
import pandas as pd
from transformers import pipeline

# Load a smaller TAPAS model for table-based question answering
tqa = pipeline(task="table-question-answering", model="google/tapas-small-finetuned-wtq")

# Load table from CSV
table = pd.read_csv("C:\\Users\\U.S\\Desktop\\ChatMate\\ChatMate-Your-Personal-Assistant-Fueled-by-Web-Scraped-Insights\\Companies.csv").astype(str)

# Streamlit app
def main():
    st.set_page_config(page_title="ChatMate - Your Personal Assistant", page_icon="🤖", layout="wide")
    
    st.title("InfoNinja 🤖")
    st.subheader("Your Personal Assistant Fueled by Web-Scraped Insights")

    # Sidebar for user input and example questions
    st.sidebar.header("Ask a Question")
    query = st.sidebar.text_input("Enter your question here:")

    st.sidebar.markdown("### Example Questions")
    st.sidebar.write("""
    - Which companies provide healthcare?
    - What is the revenue of Apple?
    - Headquarters of Apple and which place?
    """)
    
    # Button to submit the question
    if st.sidebar.button("Get Answer"):
        if query:
            try:
                # Get the answer using TAPAS
                answer = tqa(table=table, query=query)["answer"]
                st.sidebar.success(f"Answer: {answer}")
            except Exception as e:
                st.sidebar.error(f"An error occurred: {str(e)}")
        else:
            st.sidebar.warning("Please enter a question.")

    # Main content area to display the table
    st.write("### Scraped Data from Wikipedia")
    st.write("Here is the table of company data retrieved and converted to a structured format:")
    st.dataframe(table, height=500)

    # Customize the footer
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            color: gray;
            background-color: #f1f1f1;
            padding: 10px;
        }
        </style>
        <div class="footer">
            Developed by Pallav chavda
        </div>
        """,
        unsafe_allow_html=True
    )

# Run the Streamlit app
if __name__ == "__main__":
    main()
