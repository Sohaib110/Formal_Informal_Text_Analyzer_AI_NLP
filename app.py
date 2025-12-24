import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
st.set_page_config(
    page_title="Text Formality Checker",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4ECDC4;
        color:  white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        border: none;
        font-size: 16px;
    }
    . stButton>button:hover {
        background-color: #45b8b0;
        border: none;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius:  10px;
        margin:  10px 0;
    }
    </style>
""", unsafe_allow_html=True)
@st.cache_resource
def load_model():
    try:
        with open('formality_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e: 
        st.error(f"Error loading model: {e}")
        return None, None
def preprocess_text(text):
    if not text or pd.isna(text):
        return ""
    text = str(text).lower()
    text = ' '.join(text.split())
    return text
def get_formality_score(text, model, vectorizer):
    if not text or not text.strip():
        return 0.5
    processed_text = preprocess_text(text)
    text_vector = vectorizer.transform([processed_text])
    proba = model.predict_proba(text_vector)[0]
    formality_score = proba[1] 
    return formality_score
def get_interpretation(score):
    if score >= 0.8:
        return {
            'level': 'Very Formal',
            'emoji': '🎩',
            'color': '#2E7D32',
            'description': 'Professional and appropriate for formal documents',
            'tips': [
                'Perfect for business emails',
                'Suitable for academic writing',
                'Good for official documents'
            ]
        }
    elif score >= 0.6:
        return {
            'level': 'Moderately Formal',
            'emoji': '👔',
            'color': '#558B2F',
            'description': 'Polite and professional tone',
            'tips': [
                'Good for professional communication',
                'Appropriate for most workplace contexts',
                'Balanced tone'
            ]
        }
    elif score >= 0.4:
        return {
            'level': 'Neutral',
            'emoji': '😐',
            'color': '#F57C00',
            'description': 'Balanced between formal and informal',
            'tips':  [
                'Versatile for various contexts',
                'Consider your audience',
                'May need adjustment based on situation'
            ]
        }
    elif score >= 0.2:
        return {
            'level': 'Moderately Informal',
            'emoji': '😊',
            'color':  '#D84315',
            'description': 'Casual and friendly tone',
            'tips': [
                'Good for casual conversations',
                'Friendly and approachable',
                'May be too casual for professional settings'
            ]
        }
    else:
        return {
            'level': 'Very Informal',
            'emoji':  '🤙',
            'color': '#C62828',
            'description':  'Very casual, conversational style',
            'tips': [
                'Perfect for chatting with friends',
                'Good for social media',
                'Not suitable for professional contexts'
            ]
        }
def create_gauge_chart(score):
    fig = go.Figure(go. Indicator(
        mode="gauge+number+delta",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text':  "Formality Score", 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range':  [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar':  {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#FFCDD2'},
                {'range': [20, 40], 'color': '#FFAB91'},
                {'range': [40, 60], 'color': '#FFF59D'},
                {'range': [60, 80], 'color':  '#C5E1A5'},
                {'range':  [80, 100], 'color': '#A5D6A7'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score * 100
            }
        }
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    return fig
def get_text_statistics(text):
    words = text.split()
    sentences = text.count('.') + text.count('!') + text.count('?')
    sentences = max(sentences, 1)
    return {
        'char_count': len(text),
        'word_count': len(words),
        'sentence_count': sentences,
        'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0
    }
def main():
    st.title("📝 Text Formality Checker")
    st.markdown("### Analyze the formality level of your text using AI")
    st.markdown("---")
    model, vectorizer = load_model()
    if model is None or vectorizer is None:
        st.error("⚠️ Could not load the model. Please ensure 'formality_model.pkl' and 'vectorizer.pkl' exist.")
        return
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            """
            This app uses Machine Learning to analyze text formality. 
            
            **Features:**
            - Real-time formality analysis
            - Confidence scores
            - Visual gauge chart
            - Text statistics
            - Example texts
            
            **Model Info:**
            - Algorithm:  Logistic Regression
            - Dataset: 1,403 pairs
            - Accuracy: 96.09%
            """
        )
        st.header("📊 Formality Levels")
        st.markdown("""
        - 🎩 **Very Formal** (80-100%)
        - 👔 **Moderately Formal** (60-80%)
        - 😐 **Neutral** (40-60%)
        - 😊 **Moderately Informal** (20-40%)
        - 🤙 **Very Informal** (0-20%)
        """)
    col1, col2 = st. columns([2, 1])
    with col1:
        st. subheader("✍️ Enter Your Text")
        examples = {
            "Select an example... ": "",
            "Formal Email": "I am writing to formally request a meeting to discuss the upcoming project deliverables.",
            "Informal Chat": "Hey! What's up?  Wanna grab some coffee later?",
            "Business Letter": "I would like to express my sincere gratitude for your assistance with this matter.",
            "Casual Message": "Thanks a lot for your help! Really appreciate it!",
            "Neutral Request": "Can we schedule a meeting for tomorrow? ",
            "Very Formal":  "It would be greatly appreciated if you could attend the conference."
        }
        example_choice = st. selectbox(
            "Choose an example:",
            options=list(examples.keys()),
            key="example_selector"
        )
        if 'text_input_value' not in st.session_state:
            st.session_state.text_input_value = ""
        if example_choice != "Select an example...":
            st.session_state.text_input_value = examples[example_choice]
        user_input = st.text_area(
            "Type or paste your text here:",
            value=st.session_state.text_input_value,
            height=200,
            placeholder="Enter your text here to analyze its formality level...",
            help="Enter any text to analyze its formality"
        )
        if user_input != st.session_state.text_input_value:
            st. session_state.text_input_value = user_input
        analyze_button = st.button("🔍 Analyze Formality", use_container_width=True)
    with col2:
        st.subheader("💡 Tips")
        st.markdown("""
        **For best results:**
        - Use complete sentences
        - Include at least 5-10 words
        - Avoid excessive special characters
        - Write naturally
        """)
    if analyze_button and user_input. strip():
        with st.spinner("Analyzing text..."):
            formality_score = get_formality_score(user_input, model, vectorizer)
            interpretation = get_interpretation(formality_score)
            st.markdown("---")
            st.header("📊 Analysis Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Classification",
                    value=f"{interpretation['emoji']} {interpretation['level']}"
                )
            with col2:
                st.metric(
                    label="Formality Score",
                    value=f"{formality_score:.1%}"
                )
            with col3:
                confidence = formality_score if formality_score >= 0.5 else 1 - formality_score
                st.metric(
                    label="Confidence",
                    value=f"{confidence:.1%}"
                )
            st.plotly_chart(create_gauge_chart(formality_score), use_container_width=True)
            st.markdown(f"""
            <div style='background-color: {interpretation['color']}; padding: 20px; border-radius: 10px; color: white;'>
                <h3>🎯 Interpretation:  {interpretation['level']}</h3>
                <p style='font-size: 16px;'>{interpretation['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            st.subheader("💡 Usage Tips")
            for tip in interpretation['tips']:
                st.markdown(f"- {tip}")
            st.markdown("---")
            st.subheader("📈 Text Statistics")
            stats = get_text_statistics(user_input)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Characters", stats['char_count'])
            with col2:
                st.metric("Words", stats['word_count'])
            with col3:
                st.metric("Sentences", stats['sentence_count'])
            with col4:
                st. metric("Avg Word Length", f"{stats['avg_word_length']:.1f}")
    elif analyze_button:
        st.warning("⚠️ Please enter some text to analyze.")
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Built with ❤️ using Streamlit and Machine Learning</p>
        <p>Model Accuracy: 96.09% | Dataset: 1,403 pairs</p>
    </div>
    """, unsafe_allow_html=True)
if __name__ == "__main__":
    main()