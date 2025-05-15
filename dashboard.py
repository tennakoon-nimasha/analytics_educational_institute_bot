import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import altair as alt
from langchain_openai import OpenAI
from dotenv import load_dotenv
from supabase import create_client, Client
import openai
import ast
import re
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Page configuration
st.set_page_config(
    page_title="Chat System Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional color scheme and styling
st.markdown("""
<style>
    /* Overall styling */
    .main .block-container {
        padding-top: 1rem;
    }
    
    /* Headers */
    h1 {
        color: #0f2b5a;
        font-weight: 600;
        font-size: 2.2rem;
    }
    h2 {
        color: #0f2b5a;
        font-weight: 500;
        font-size: 1.8rem;
        margin-top: 1rem;
    }
    h3 {
        color: #2c5282;
        font-weight: 500;
        font-size: 1.4rem;
    }
    
    /* Metrics styling */
    .stMetric {
        background-color: #f8fafc;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stMetric label {
        color: #4a5568;
        font-weight: 500;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
        color: #0f2b5a;
    }
    
    /* Message styling */
    .user-message {
        background-color: #f1f5f9;
        border-left: 4px solid #2c5282;
        border-radius: 4px;
        padding: 12px;
        margin: 8px 0;
    }
    .bot-message {
        background-color: #f8fafc;
        border-left: 4px solid #7c3aed;
        border-radius: 4px;
        padding: 12px;
        margin: 8px 0;
    }
    
    /* Panel styling */
    .panel {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border: 1px solid #e2e8f0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
    }
    
    /* Button styling */
    button[kind="primary"] {
        background-color: #2c5282;
    }
    
    /* Dataframe styling */
    [data-testid="stTable"] {
        border: 1px solid #e2e8f0;
        border-radius: 6px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        background-color: #f1f5f9;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2c5282 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Supabase connection
@st.cache_resource
def get_supabase_client():
    """Get a connection to Supabase"""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        st.error("Missing Supabase credentials. Please add SUPABASE_URL and SUPABASE_KEY to your .env file")
        st.stop()
    
    return create_client(supabase_url, supabase_key)

# Topic analysis functions
def analyze_topics_with_openai(messages, api_key=None):
    """
    Analyze the main topics discussed in a chat using OpenAI's API.

    Args:
        messages (list of str): The chat messages.
        api_key (str): Optional OpenAI API key.

    Returns:
        list of str: A list of 3–5 main topics.
    """
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        return ["Topic analysis unavailable (no API key)"]
    
    # openai.api_key = api_key
    
    full_content = " ".join(msg.strip() for msg in messages if msg.strip())

    # Truncate to avoid exceeding token limits
    if len(full_content) > 4000:
        full_content = full_content[:4000] + "..."
    
    prompt = f"""
        You are given a chat conversation. Analyze it carefully and extract the main topics being discussed.

        Instructions:
        - Return only a Python list containing 3 to 5 concise topic names.
        - Do not include any explanations, formatting, or text outside the list.
        - Each topic should be a short phrase or keyword.

        Chat content:
        {full_content}

        Main topics:
        """

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model="gpt-4o-mini",  
            input=prompt
        )

        raw_output = response.output[0].content[0].text

        # Try safe parsing
        try:
            topics = ast.literal_eval(raw_output)
            if isinstance(topics, list):
                return topics
        except (ValueError, SyntaxError):
            pass

        # Fallback: extract quoted strings
        topics = re.findall(r'"([^"]+)"', raw_output)
        if topics:
            return topics

        return ["Topic extraction failed", "Try manual analysis"]

    except Exception as e:
        print(f"Error analyzing topics: {str(e)}")
        return ["Error in topic analysis"]

def generate_faqs_with_openai(messages, api_key=None, num_faqs=5):
    """
    Generate FAQs based on chat history using OpenAI
    
    Args:
        messages: List of message content
        api_key: OpenAI API key (optional)
        num_faqs: Number of FAQs to generate
        
    Returns:
        List of FAQs with questions and answers
    """
    if not api_key and os.environ.get("OPENAI_API_KEY"):
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        return [{"question": "FAQ generation unavailable", "answer": "No API key provided"}]
    
    full_content = " ".join([msg for msg in messages if msg.strip()])
    
    # Truncate if too long
    if len(full_content) > 8000:
        full_content = full_content[:8000] + "..."
    
    # llm = OpenAI(temperature=0.2, model="gpt-3.5-turbo-instruct", api_key=api_key)
    client = OpenAI(api_key=api_key)
    prompt = f"""
    Analyze the following chat conversation history and create {num_faqs} helpful FAQs for administrators 
    managing this RAG system. Identify common user questions, issues, or patterns, and provide clear answers.
    
    The FAQs should help administrators better understand user behavior, improve the system, 
    and address common challenges or questions. These FAQs are for internal use by administrators.
    
    Format your response as a JSON array of objects, each with 'question' and 'answer' fields.
    Don't include any explanations or other text outside the JSON array.
    
    Chat content for analysis:
    {full_content}
    
    Example format:
    [
        {{"question": "What are users asking about most frequently?", "answer": "Based on the conversations, users are most frequently asking about..."}},
        {{"question": "How can we improve response quality?", "answer": "The data suggests that..."}}
    ]
    """
    
    try:
        response = client.responses.create(
            model="gpt-4o-mini",  
            input=prompt
        )

        # Clean and parse the response
        response = response.output[0].content[0].text
        
        # Extract JSON if wrapped in other text
        import re
        import json
        
        # Try to find JSON array in the response
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                faqs = json.loads(json_str)
                return faqs
            except:
                pass
        
        # If not found or couldn't parse, try to parse the whole response
        try:
            faqs = json.loads(response)
            return faqs
        except:
            # Manual extraction as fallback
            questions = re.findall(r'"question"\s*:\s*"([^"]*)"', response)
            answers = re.findall(r'"answer"\s*:\s*"([^"]*)"', response)
            
            if questions and answers and len(questions) == len(answers):
                return [{"question": q, "answer": a} for q, a in zip(questions, answers)]
            
            return [
                {"question": "Couldn't parse FAQs properly", 
                 "answer": "Please check the API response format or try again."}
            ]
    except Exception as e:
        st.error(f"Error generating FAQs: {str(e)}")
        return [{"question": "Error in FAQ generation", "answer": str(e)}]

def extract_keywords(text, top_n=10):
    """Extract the top keywords from text"""
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    
    # Filter out stop words and non-alphabetic tokens
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
    
    # Count word frequencies
    word_freq = Counter(filtered_words)
    
    # Return top N keywords
    return dict(word_freq.most_common(top_n))

# Data visualization functions
def plot_message_timeline(df):
    """Plot message timeline"""
    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract date
    df['date'] = df['timestamp'].dt.date
    
    # Group by date and sender, count messages
    daily_counts = df.groupby(['date', 'sender']).size().reset_index(name='count')
    
    # Create the chart
    chart = alt.Chart(daily_counts).mark_line(point=True).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('count:Q', title='Number of Messages'),
        color=alt.Color('sender:N', title='Sender', 
                        scale=alt.Scale(domain=['user', 'bot'], 
                                        range=['#2c5282', '#7c3aed'])),
        tooltip=['date', 'sender', 'count']
    ).properties(
        height=300
    ).interactive()
    
    return chart

def plot_user_activity(df, candidates_df):
    """Plot user activity by date added"""
    # Ensure created_at is datetime
    if not pd.api.types.is_datetime64_any_dtype(candidates_df['created_at']):
        candidates_df['created_at'] = pd.to_datetime(candidates_df['created_at'])
    
    # Extract date
    candidates_df['date'] = candidates_df['created_at'].dt.date
    
    # Count users by date
    user_counts = candidates_df.groupby('date').size().reset_index(name='new_users')
    
    # Create cumulative sum
    user_counts['cumulative_users'] = user_counts['new_users'].cumsum()
    
    # Create chart
    base = alt.Chart(user_counts).encode(
        x=alt.X('date:T', title='Date')
    )
    
    bar = base.mark_bar(color='#2c5282').encode(
        y=alt.Y('new_users:Q', title='New Users'),
        tooltip=['date', 'new_users']
    )
    
    line = base.mark_line(color='#7c3aed', strokeWidth=3).encode(
        y=alt.Y('cumulative_users:Q', title='Cumulative Users'),
        tooltip=['date', 'cumulative_users']
    )
    
    chart = alt.layer(bar, line).resolve_scale(
        y='independent'
    ).properties(
        height=300
    ).interactive()
    
    return chart

def analyze_message_sentiment(df, top_n=20):
    """Get a list of the most common terms/phrases in messages"""
    # Extract user messages
    user_messages = df[df['sender'] == 'user']['message'].str.cat(sep=' ')
    
    # Extract keywords
    keywords = extract_keywords(user_messages, top_n)
    
    # Convert to DataFrame for visualization
    keywords_df = pd.DataFrame({
        'keyword': list(keywords.keys()),
        'frequency': list(keywords.values())
    }).sort_values('frequency', ascending=False)
    
    # Create chart
    chart = alt.Chart(keywords_df).mark_bar(color='#2c5282').encode(
        x=alt.X('frequency:Q', title='Frequency'),
        y=alt.Y('keyword:N', title='Keyword', sort='-x'),
        tooltip=['keyword', 'frequency']
    ).properties(
        height=400
    )
    
    return chart

def identify_common_topics(df, api_key=None):
    """Identify common topics across all conversations"""
    if len(df) == 0:
        return ["No messages available for analysis"]
    
    # Extract all user messages
    user_messages = df[df['sender'] == 'user']['message'].tolist()
    
    if not user_messages:
        return ["No user messages available for analysis"]
    
    # Use OpenAI to identify topics
    return analyze_topics_with_openai(user_messages, api_key)

def plot_response_times(df):
    """Plot average response times by hour of day"""
    if len(df) < 2:
        return None
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add hour column
    df['hour'] = df['timestamp'].dt.hour
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Calculate response times
    response_times = []
    
    for user_id in df['candidate_id'].unique():
        user_df = df[df['candidate_id'] == user_id].copy()
        
        # Create a shifted column for the previous message timestamp
        user_df['prev_timestamp'] = user_df['timestamp'].shift(1)
        user_df['prev_sender'] = user_df['sender'].shift(1)
        
        # Calculate response time where current sender is 'bot' and previous sender is 'user'
        user_df['response_time'] = None
        mask = (user_df['sender'] == 'bot') & (user_df['prev_sender'] == 'user')
        user_df.loc[mask, 'response_time'] = (user_df.loc[mask, 'timestamp'] - 
                                             user_df.loc[mask, 'prev_timestamp']).dt.total_seconds()
        
        # Add valid response times to our list
        valid_times = user_df[user_df['response_time'].notna()]
        response_times.extend(zip(valid_times['hour'], valid_times['response_time']))
    
    if not response_times:
        return None
    
    # Create DataFrame from response times
    response_df = pd.DataFrame(response_times, columns=['hour', 'response_time'])
    
    # Calculate average response time by hour
    avg_response = response_df.groupby('hour')['response_time'].mean().reset_index()
    avg_response['response_time'] = avg_response['response_time'].round(2)
    
    # Create chart
    chart = alt.Chart(avg_response).mark_bar(color='#7c3aed').encode(
        x=alt.X('hour:O', title='Hour of Day', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('response_time:Q', title='Avg. Response Time (seconds)'),
        tooltip=['hour', 'response_time']
    ).properties(
        title='Average Bot Response Time by Hour of Day',
        height=300
    )
    
    return chart

# Main application
def main():
    # Set up the sidebar
    st.sidebar.title("RAG Analytics Dashboard")
    
    # Initialize Supabase client
    supabase = get_supabase_client()
    
    # Load data from Supabase
    with st.spinner("Loading data from Supabase..."):
        # Get candidates data
        candidates_response = supabase.table('candidates_education').select('*').execute()
        if hasattr(candidates_response, 'error') and candidates_response.error:
            st.error(f"Error fetching candidates: {candidates_response.error}")
            st.stop()
        
        candidates_df = pd.DataFrame(candidates_response.data)
        
        # Get chat history data
        chat_response = supabase.table('chat_history_education').select('*').execute()
        if hasattr(chat_response, 'error') and chat_response.error:
            st.error(f"Error fetching chat history: {chat_response.error}")
            st.stop()
        
        chat_df = pd.DataFrame(chat_response.data)
        
    # Handle empty dataframes
    if len(candidates_df) == 0:
        st.warning("No candidate data found in Supabase")
        st.stop()
    
    if len(chat_df) == 0:
        st.warning("No chat history found in Supabase")
        st.stop()
    
    # Convert timestamps to datetime
    for df in [candidates_df, chat_df]:
        for col in df.columns:
            if 'time' in col.lower() or col.lower() == 'created_at':
                df[col] = pd.to_datetime(df[col])
    
    # Data filtering options
    st.sidebar.markdown("## Filters")
    
    # Get user IDs for filtering
    candidate_ids = ['All Users'] + candidates_df['candidate_id'].unique().tolist()
    selected_user = st.sidebar.selectbox("Select User", candidate_ids)
    
    # Date range selection
    min_date = chat_df['timestamp'].min().date() if len(chat_df) > 0 else datetime.now().date() - timedelta(days=30)
    max_date = chat_df['timestamp'].max().date() if len(chat_df) > 0 else datetime.now().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date
    
    # Filter data based on selection
    if selected_user != 'All Users':
        chat_filtered = chat_df[chat_df['candidate_id'] == selected_user]
    else:
        chat_filtered = chat_df.copy()
    
    # Filter by date
    chat_filtered = chat_filtered[
        (chat_filtered['timestamp'].dt.date >= start_date) & 
        (chat_filtered['timestamp'].dt.date <= end_date)
    ]
    
    # Dashboard Header
    st.title("Analytics Dashboard")
    
    # Display applied filters
    st.markdown(
        f"<div style='padding: 10px; background-color: #f8fafc; border-radius: 5px; margin-bottom: 20px;'>"
        f"<strong>Applied Filters:</strong> {'All Users' if selected_user == 'All Users' else f'User ID: {selected_user}'} | "
        f"Date Range: {start_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')}"
        f"</div>", 
        unsafe_allow_html=True
    )
    
    # Create tabs for different sections
    tabs = st.tabs(["Overview", "Conversations", "Topic Analysis", "FAQ Generator", "User Analytics"])
    
    # Tab 1: Overview
    with tabs[0]:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Key Performance Metrics")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            active_users = len(chat_filtered['candidate_id'].unique())
            total_users = len(candidates_df)
            st.metric("Active Users", active_users, 
                     f"{round(active_users/total_users*100, 1)}% of total" if total_users > 0 else "")
        
        with col2:
            total_msgs = len(chat_filtered)
            st.metric("Total Messages", f"{total_msgs:,}")
        
        with col3:
            user_msgs = len(chat_filtered[chat_filtered['sender'] == 'user'])
            bot_msgs = len(chat_filtered[chat_filtered['sender'] == 'bot'])
            msg_ratio = round(user_msgs / bot_msgs, 2) if bot_msgs > 0 else 0
            st.metric("User:Bot Ratio", f"{msg_ratio:.2f}")
        
        with col4:
            avg_msgs_per_user = round(total_msgs / active_users, 1) if active_users > 0 else 0
            st.metric("Avg. Messages/User", avg_msgs_per_user)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Charts
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Activity Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Message Volume")
            if len(chat_filtered) > 0:
                timeline_chart = plot_message_timeline(chat_filtered)
                st.altair_chart(timeline_chart, use_container_width=True)
            else:
                st.info("No message data available for the selected filters.")
        
        with col2:
            st.markdown("### User Growth")
            if len(candidates_df) > 0:
                user_chart = plot_user_activity(chat_filtered, candidates_df)
                st.altair_chart(user_chart, use_container_width=True)
            else:
                st.info("No user data available.")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Response time analysis
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("System Performance")
        
        response_chart = plot_response_times(chat_filtered)
        if response_chart:
            st.altair_chart(response_chart, use_container_width=True)
        else:
            st.info("Insufficient data to calculate response times.")
            
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Tab 2: Conversations
    with tabs[1]:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Conversation Explorer")
        
        # Get unique users who have messages
        users_with_messages = chat_df['candidate_id'].unique().tolist()
        
        if len(users_with_messages) > 0:
            # Get user metadata if available
            user_meta = {}
            for user_id in users_with_messages:
                user_data = candidates_df[candidates_df['candidate_id'] == user_id]
                if len(user_data) > 0:
                    created_at = user_data['created_at'].iloc[0]
                    user_meta[user_id] = {
                        'created_at': created_at.strftime('%Y-%m-%d') if pd.notna(created_at) else 'Unknown',
                        'message_count': len(chat_df[chat_df['candidate_id'] == user_id])
                    }
            
            # Create user selection with metadata
            user_options = []
            for user_id in users_with_messages:
                if user_id in user_meta:
                    meta = user_meta[user_id]
                    option = f"{user_id} - Joined: {meta['created_at']} - Messages: {meta['message_count']}"
                else:
                    option = f"{user_id}"
                user_options.append(option)
            
            selected_chat_user_option = st.selectbox(
                "Select a user to view conversation",
                user_options
            )
            
            # Extract user_id from selection
            selected_chat_user = int(selected_chat_user_option.split(' - ')[0])
            # print(selected_chat_user)
            if selected_chat_user:
                # Get user's messages
                user_chat = chat_df[chat_df['candidate_id'] == selected_chat_user].sort_values('timestamp')
                # print(user_chat)
                if len(user_chat) > 0:
                    # Display conversation metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        first_msg = user_chat['timestamp'].min()
                        st.metric("First Message", first_msg.strftime('%Y-%m-%d'))
                        
                    with col2:
                        last_msg = user_chat['timestamp'].max()
                        st.metric("Last Message", last_msg.strftime('%Y-%m-%d'))
                        
                    with col3:
                        active_days = user_chat['timestamp'].dt.date.nunique()
                        st.metric("Active Days", active_days)
                    
                    # Display conversation
                    st.markdown("### Conversation History")
                    
                    for _, row in user_chat.iterrows():
                        sender = row['sender']
                        content = row['message']
                        time = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                        
                        if sender == 'user':
                            st.markdown(f"<div class='user-message'><strong>User</strong> ({time}):<br>{content}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='bot-message'><strong>Bot</strong> ({time}):<br>{content}</div>", unsafe_allow_html=True)
                    
                    # Topic analysis for this conversation
                    with st.expander("Analyze This Conversation", expanded=False):
                        api_key = st.text_input("OpenAI API Key (optional)", 
                                              type="password", 
                                              help="Enter your OpenAI API key to analyze topics",
                                              key="convo_api_key")
                        
                        if st.button("Analyze Topics", key="analyze_convo"):
                            # Get message content for topic analysis
                            messages_content = user_chat['message'].tolist()
                            
                            with st.spinner("Analyzing topics..."):
                                # Only run if we have messages
                                if messages_content:
                                    topics = analyze_topics_with_openai(messages_content, api_key)
                                    st.markdown("### Detected Topics")
                                    for topic in topics:
                                        st.markdown(f"- {topic}")
                                else:
                                    st.warning("No messages to analyze.")
                else:
                    st.info("No messages found for this user.")
        else:
            st.info("No conversations found in the database.")
            
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Tab 3: Topic Analysis
    with tabs[2]:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Content Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Popular Keywords")
            if len(chat_filtered) > 0 and 'message' in chat_filtered.columns:
                keyword_chart = analyze_message_sentiment(chat_filtered)
                st.altair_chart(keyword_chart, use_container_width=True)
            else:
                st.info("No message data available for keyword analysis.")
        
        with col2:
            st.markdown("### Common Topics")
            api_key = st.text_input("OpenAI API Key (optional)", 
                                  type="password", 
                                  help="Enter your OpenAI API key to analyze topics",
                                  key="topics_api_key")
            
            if st.button("Identify Common Topics", key="analyze_topics"):
                with st.spinner("Analyzing topics with AI..."):
                    topics = identify_common_topics(chat_filtered, api_key)
                    
                    st.markdown("### Detected Topics")
                    for topic in topics:
                        st.markdown(f"- {topic}")
            else:
                st.info("Click the button to analyze common topics using OpenAI.")
                
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Tab 4: FAQ Generator
    with tabs[3]:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("FAQ Generator for Administrators")
        
        st.markdown("""
        This tool analyzes your chat history to automatically generate FAQs that can help administrators understand:
        
        - Common user questions and patterns
        - Potential improvements to the RAG system
        - Issues users might be facing
        - How to better manage the system
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            api_key = st.text_input("OpenAI API Key (optional)", 
                                  type="password", 
                                  help="Enter your OpenAI API key to generate FAQs",
                                  key="faq_api_key")
            
            num_faqs = st.slider("Number of FAQs to generate", min_value=3, max_value=10, value=5)
        
        with col2:
            if st.button("Generate Admin FAQs", key="generate_faqs"):
                # Get all messages for FAQ generation
                all_messages = chat_filtered['message'].tolist()
                
                if len(all_messages) < 10:
                    st.warning("Not enough messages to generate meaningful FAQs. Please select a wider date range or more users.")
                else:
                    with st.spinner("Generating FAQs with AI..."):
                        faqs = generate_faqs_with_openai(all_messages, api_key, num_faqs)
                        
                        st.markdown("### Generated FAQs for Administrators")
                        
                        for i, faq in enumerate(faqs, 1):
                            with st.expander(f"Q{i}: {faq['question']}", expanded=True):
                                st.markdown(faq['answer'])
            else:
                st.info("Click the button to generate administrator FAQs using OpenAI.")
                
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Tab 5: User Analytics
    with tabs[4]:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("User Analytics")
        
        if len(candidates_df) > 0:
            # Create user metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                new_users = len(candidates_df[candidates_df['created_at'].dt.date >= start_date])
                st.metric("New Users in Period", new_users)
            
            with col2:
                active_users = len(chat_filtered[chat_filtered['timestamp'].dt.date >= start_date]['candidate_id'].unique())
                st.metric("Active Users in Period", active_users)
            
            with col3:
                retention = round(active_users / new_users * 100, 1) if new_users > 0 else 0
                st.metric("Retention Rate", f"{retention}%")
            
            # User engagement metrics
            st.markdown("### User Engagement")
            
            # Calculate user message counts and other metrics
            user_message_counts = chat_df.groupby('candidate_id')['message'].count().reset_index()
            user_message_counts.columns = ['User ID', 'Message Count']
            
            # Calculate messages per day
            chat_df['date'] = pd.to_datetime(chat_df['timestamp']).dt.date
            user_days = chat_df.groupby(['candidate_id', 'date']).size().reset_index()
            user_days = user_days.groupby('candidate_id').size().reset_index()
            user_days.columns = ['User ID', 'Active Days']
            
            # Calculate first and last activity
            user_first_activity = chat_df.groupby('candidate_id')['timestamp'].min().reset_index()
            user_first_activity.columns = ['User ID', 'First Activity']
            
            user_last_activity = chat_df.groupby('candidate_id')['timestamp'].max().reset_index()
            user_last_activity.columns = ['User ID', 'Last Activity']
            
            # Merge all metrics
            user_stats = user_message_counts
            user_stats = user_stats.merge(user_days, on='User ID', how='left')
            user_stats = user_stats.merge(user_first_activity, on='User ID', how='left')
            user_stats = user_stats.merge(user_last_activity, on='User ID', how='left')
            
            # Calculate messages per day
            user_stats['Messages per Day'] = user_stats['Message Count'] / user_stats['Active Days']
            user_stats['Messages per Day'] = user_stats['Messages per Day'].round(1)
            
            # Format datetime columns
            user_stats['First Activity'] = pd.to_datetime(user_stats['First Activity']).dt.strftime('%Y-%m-%d %H:%M')
            user_stats['Last Activity'] = pd.to_datetime(user_stats['Last Activity']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Sort by message count
            user_stats = user_stats.sort_values('Message Count', ascending=False)
            
            # Display the table
            st.dataframe(user_stats, use_container_width=True)
            
            # Show most and least engaged users
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Most Engaged Users")
                most_engaged = user_stats.head(5)[['User ID', 'Message Count', 'Active Days', 'Messages per Day']]
                st.dataframe(most_engaged, use_container_width=True)
            
            with col2:
                st.markdown("### Least Engaged Users")
                # Filter to users with at least 2 messages
                active_users = user_stats[user_stats['Message Count'] >= 2]
                least_engaged = active_users.tail(5)[['User ID', 'Message Count', 'Active Days', 'Messages per Day']]
                st.dataframe(least_engaged, use_container_width=True)
        else:
            st.info("No user data available.")
            
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Add footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #718096;'>"
        "RAG System Analytics Dashboard | Last updated: " + 
        datetime.now().strftime("%Y-%m-%d %H:%M:%S") + 
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()