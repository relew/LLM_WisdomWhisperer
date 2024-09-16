import os
import time
import uuid
import pandas as pd
import streamlit as st

from assistant import get_answer
from db import save_conversation, save_feedback, get_recent_conversations, get_feedback_stats, init_table_if_needed

def print_log(message):
    print(message, flush=True)

def main():
    # Initialize tables if needed
    init_table_if_needed()

    # Set up the Streamlit app
    st.set_page_config(page_title="Philosophy Assistant", layout="wide")
    st.title("Philosophy Assistant")

    # Sidebar for settings
    st.sidebar.header("Settings")
    ideology = st.sidebar.selectbox("Select an ideology:", ["stoicism", "zen-buddhism"])
    model_choice = st.sidebar.selectbox("Select a model:", ["ollama/phi3", "openai/gpt-4o", "openai/gpt-4o-mini"])
    search_type = st.sidebar.radio("Select search type:", ["Text", "Vector"])

    # User input form
    with st.form(key='question_form'):
        user_input = st.text_input("Enter your question:")
        submit_button = st.form_submit_button(label='Ask')

    if submit_button:
        print_log(f"User asked: '{user_input}'")
        with st.spinner('Processing...'):

            # Session state initialization
            st.session_state.conversation_id = str(uuid.uuid4())
            print_log(f"New conversation started with ID: {st.session_state.conversation_id}")

            print_log(f"Getting answer from assistant using {model_choice} model and {search_type} search")
            start_time = time.time()
            answer_data = get_answer(user_input, ideology, model_choice, search_type)
            end_time = time.time()
            print_log(f"Answer received in {end_time - start_time:.2f} seconds")

            st.success("Completed!")
            st.markdown(f"**Answer:** {answer_data['answer']}")
            st.markdown(f"**Response Time:** {answer_data['response_time']:.2f} seconds")
            st.markdown(f"**Relevance:** {answer_data['relevance']}")
            st.markdown(f"**Model Used:** {answer_data['model_used']}")
            st.markdown(f"**Total Tokens:** {answer_data['total_tokens']}")
            if answer_data['openai_cost'] > 0:
                st.markdown(f"**OpenAI Cost:** ${answer_data['openai_cost']:.4f}")

            # Save conversation to database
            print_log("Saving conversation to database")
            save_conversation(st.session_state.conversation_id, user_input, answer_data, ideology)
            print_log("Conversation saved successfully")

    # Feedback buttons
    st.sidebar.subheader("Feedback")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("👍"):
            if 'count' not in st.session_state:
                st.session_state.count = 0
            st.session_state.count += 1
            print_log(f"Positive feedback received. New count: {st.session_state.count}")
            save_feedback(st.session_state.conversation_id, 1)
            print_log("Positive feedback saved to database")
    with col2:
        if st.button("👎"):
            if 'count' not in st.session_state:
                st.session_state.count = 0
            st.session_state.count -= 1
            print_log(f"Negative feedback received. New count: {st.session_state.count}")
            save_feedback(st.session_state.conversation_id, -1)
            print_log("Negative feedback saved to database")

    if 'count' in st.session_state:
        st.sidebar.markdown(f"**Current Count:** {st.session_state.count}")

    # Display recent conversations
    st.subheader("Recent Conversations")
    relevance_filter = st.selectbox("Filter by relevance:", ["All", "RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"])
    recent_conversations = get_recent_conversations(limit=5, relevance=relevance_filter if relevance_filter != "All" else None)

    if recent_conversations:
        # Convert the recent conversations into a DataFrame
        asd_df = pd.DataFrame(recent_conversations)
        print_log(asd_df.columns)

        df_recent_conversations = pd.DataFrame(recent_conversations, columns=[
            "id", "question", "answer", "ideology", "model_used","completion_time","relevance", "relevance_explained",
            "prompt_tokens", "completion_tokens", "total_tokens", "eval_prompt_tokens", "eval_completion_tokens", "eval_total_tokens", 
            "openai_cost","timestamp","feedback"
        ])

        # Drop unnecessary columns
        df_recent_conversations = df_recent_conversations.drop(
            columns=["id","completion_time","prompt_tokens", "completion_tokens", "total_tokens",
                    "eval_prompt_tokens", "eval_completion_tokens", "eval_total_tokens",
                    "openai_cost","timestamp","feedback"],
            axis=1
        )
        
        # Display the DataFrame
        st.dataframe(df_recent_conversations)
    else:
        st.write("No recent conversations to display.")

    # Display feedback stats
    st.subheader("Feedback Statistics")
    feedback_stats = get_feedback_stats()

    if feedback_stats:
        # Convert feedback stats into a DataFrame
        df_feedback_stats = pd.DataFrame([feedback_stats], columns=["thumbs-up", "thumbs-down"])
        st.dataframe(df_feedback_stats)
    else:
        st.write("No feedback statistics available.")


if __name__ == "__main__":
    print_log("Philosophy Assistant application started")
    main()
