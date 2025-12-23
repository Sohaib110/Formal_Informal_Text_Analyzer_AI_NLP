import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

class FormalityChecker:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text) or not text: 
            return ""
        text = str(text).lower()
        text = ' '.join(text.split())  # Remove extra whitespace
        return text
    
    def get_formality_score(self, text):
        """Get formality score for a given text"""
        if not text or not text.strip():
            return 0.5
        
        processed_text = self.preprocess_text(text)
        text_vector = self. vectorizer.transform([processed_text])
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(text_vector)[0]
            formality_score = proba[1]
        else: 
            decision = self.model. decision_function(text_vector)[0]
            formality_score = 1 / (1 + np.exp(-decision))
        
        return formality_score
    
    def predict(self, text):
        """Predict whether text is formal or informal"""
        formality_score = self. get_formality_score(text)
        
        if formality_score >= 0.5:
            label = "Formal"
            confidence = formality_score
        else:
            label = "Informal"
            confidence = 1 - formality_score
        
        return {
            'label': label,
            'confidence': confidence,
            'formality_score': formality_score
        }
    
    def load_model(self, model_path='formality_model.pkl', 
                   vectorizer_path='vectorizer.pkl'):
        """Load pre-trained model and vectorizer"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)


@st.cache_resource
def load_model():
    """Load model with caching"""
    checker = FormalityChecker()
    try:
        checker.load_model('formality_model.pkl', 'vectorizer.pkl')
        return checker
    except FileNotFoundError:
        st.error("⚠️ Model files not found!  Please train the model first by running 'python formality_checker.py'")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def create_gauge_chart(score):
    """Create a gauge chart for formality score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text':  "Formality Level (%)", 'font': {'size': 24}},
        gauge={
            'axis': {'range':  [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar':  {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#ffcccb'},
                {'range':  [20, 40], 'color': '#ffd9b3'},
                {'range': [40, 60], 'color':  '#ffffcc'},
                {'range': [60, 80], 'color': '#c1f0c1'},
                {'range':  [80, 100], 'color': '#add8e6'}
            ],
            'threshold': {
                'line': {'color':  "red", 'width': 4},
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

def get_interpretation(score):
    """Get interpretation based on formality score"""
    if score >= 0.8:
        return {
            'level': 'Very Formal',
            'emoji': '🎩',
            'color': 'success',
            'description': 'This text uses highly formal language suitable for professional, academic, or official contexts.',
            'tips': [
                '✓ Perfect for business correspondence',
                '✓ Suitable for academic papers',
                '✓ Appropriate for legal documents'
            ]
        }
    elif score >= 0.6:
        return {
            'level': 'Moderately Formal',
            'emoji':  '👔',
            'color': 'info',
            'description': 'This text has a formal tone but may include some less formal elements.',
            'tips':  [
                '✓ Good for professional emails',
                '✓ Suitable for reports',
                '~ Consider making it more formal for official documents'
            ]
        }
    elif score >= 0.4:
        return {
            'level': 'Neutral',
            'emoji': '📝',
            'color': 'warning',
            'description': 'This text is neither particularly formal nor informal.',
            'tips': [
                '✓ Works for general communication',
                '~ May need adjustment based on context',
                '~ Consider your audience and purpose'
            ]
        }
    elif score >= 0.2:
        return {
            'level':  'Moderately Informal',
            'emoji': '👕',
            'color': 'warning',
            'description': 'This text has a casual tone with some informal elements.',
            'tips': [
                '✓ Good for friendly emails',
                '✓ Suitable for casual blog posts',
                '✗ Not appropriate for formal contexts'
            ]
        }
    else:
        return {
            'level': 'Very Informal',
            'emoji': '😊',
            'color': 'error',
            'description': 'This text uses highly casual language, suitable for informal conversations.',
            'tips': [
                '✓ Perfect for chatting with friends',
                '✓ Good for social media',
                '✗ Avoid in professional settings'
            ]
        }

def main():
    st.set_page_config(page_title="Formality Checker", page_icon="📝", layout="wide")
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color:  white;
            font-size: 18px;
            padding: 10px;
            border-radius: 10px;
        }
        .stTextArea>div>div>textarea {
            font-size: 16px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("📝 Text Formality Checker")
    st.markdown("### Analyze Your Text for Formal and Informal Tone Using NLP")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st. header("ℹ️ About")
        st.info(
            "This application uses **Natural Language Processing** and **Machine Learning** "
            "to analyze text and determine whether it's formal or informal.\n\n"
            "**How it works:**\n"
            "1. Enter your text\n"
            "2. Click 'Analyze'\n"
            "3. Get instant results with confidence scores"
        )
        
        st.markdown("---")
        st.header("📊 Features")
        st.markdown("""
        - ✅ Real-time analysis
        - ✅ Confidence scores
        - ✅ Visual indicators
        - ✅ Contextual tips
        - ✅ Example texts
        """)
        
        st.markdown("---")
        st.header("👨‍💻 Developer")
        st.markdown("**Muhammad Sohaib**  \n22-CS-110")
    
    # Load model
    checker = load_model()
    
    if checker is None:
        st.stop()
    
    # Main content
    col1, col2 = st. columns([3, 2])
    
    with col1:
        st.subheader("📋 Enter Your Text")
        user_text = st.text_area(
            label="Input Text",
            height=250,
            placeholder="Type or paste your text here.. .\n\nExample: 'I would like to request a meeting to discuss the project details.'",
            key="user_input",
            label_visibility="collapsed"
        )
        
        col_btn1, col_btn2, col_btn3 = st. columns([1, 1, 1])
        with col_btn2:
            analyze_button = st.button("🔍 Analyze Formality", type="primary")
    
    with col2:
        st.subheader("💡 Try Examples")
        
        examples = {
            "Select an example... ": "",
            "🎩 Very Formal": "I am writing to formally request your assistance with this matter.  Could you kindly provide the necessary documentation?",
            "👔 Formal": "I would like to schedule a meeting to discuss the project details.",
            "📝 Neutral": "Can we meet tomorrow to talk about the project?",
            "👕 Informal": "Hey! Can we catch up tomorrow about that project?",
            "😊 Very Informal": "Hey! What's up? Wanna hang out and chat about stuff?"
        }
        
        example_choice = st.selectbox(
            label="Choose Example",
            options=list(examples.keys()),
            key="example_select",
            label_visibility="collapsed"
        )
        
        # Show preview of selected example
        if example_choice != "Select an example...":
            st. markdown("**Preview:**")
            st.info(examples[example_choice])
            
            # Button to use this example
            if st.button("📝 Use This Example", key="use_example"):
                st.session_state.user_input = examples[example_choice]
                st.experimental_rerun()  # Compatible with older Streamlit versions
    
    # Analysis - Only analyze what's in the text area
    if analyze_button and user_text and user_text.strip():
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        # Show what text is being analyzed
        with st.expander("📄 Analyzed Text", expanded=False):
            st.write(user_text)
        
        # Get prediction
        with st.spinner("Analyzing text..."):
            result = checker. predict(user_text)
        
        interpretation = get_interpretation(result['formality_score'])
        
        # Display results in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Classification",
                f"{interpretation['emoji']} {result['label']}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Confidence",
                f"{result['confidence']:.1%}",
                delta=None
            )
        
        with col3:
            st.metric(
                "Formality Score",
                f"{result['formality_score']:.1%}",
                delta=None
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Gauge chart
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig = create_gauge_chart(result['formality_score'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown(f"### {interpretation['emoji']} {interpretation['level']}")
            
            if interpretation['color'] == 'success':
                st.success(interpretation['description'])
            elif interpretation['color'] == 'info': 
                st.info(interpretation['description'])
            elif interpretation['color'] == 'warning':
                st.warning(interpretation['description'])
            else:
                st.error(interpretation['description'])
            
            st.markdown("**Tips:**")
            for tip in interpretation['tips']:
                st.markdown(f"- {tip}")
        
        # Text statistics
        st.markdown("---")
        st.subheader("📈 Text Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            char_count = len(user_text)
            st.metric("Characters", char_count)
        
        with col2:
            word_count = len(user_text.split())
            st.metric("Words", word_count)
        
        with col3:
            words = user_text.split()
            avg_word_len = np.mean([len(word) for word in words]) if words else 0
            st.metric("Avg Word Length", f"{avg_word_len:.1f}")
        
        with col4:
            sentences = [s.strip() for s in user_text.split('.') if s.strip()]
            st.metric("Sentences", len(sentences))
    
    elif analyze_button: 
        st.warning("⚠️ Please enter some text to analyze!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Made with ❤️ using Streamlit and Machine Learning | "
        "© 2024 Muhammad Sohaib (22-CS-110)"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__": 
    main()