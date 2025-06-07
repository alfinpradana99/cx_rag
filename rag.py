import streamlit as st

from google import genai
from google.genai import types
from pinecone import Pinecone
import ast
from typing import List, Dict, Tuple, Optional

# Page configuration
st.set_page_config(
    page_title="Balto Customer Service RAG Demo",
    page_icon="🐕",
    layout="wide"
)

# Initialize session state for API keys and prompts
if 'api_keys_set' not in st.session_state:
    st.session_state.api_keys_set = False

# Product ingredients dictionary
product_ingredients = {
    "Joint Care": {
        "active_ingredients": ["Glucosamine", "Chondroitin", "MSM", "Omega-3"],
        "supporting_ingredients": ["Vitamin C", "Vitamin E", "Turmeric"]
    },
    "Anti-itch": {
        "active_ingredients": ["Quercetin", "Omega-3", "Probiotics"],
        "supporting_ingredients": ["Biotin", "Zinc", "Vitamin A"]
    },
    "Calming": {
        "active_ingredients": ["L-Theanine", "Chamomile", "Valerian Root"],
        "supporting_ingredients": ["Melatonin", "Tryptophan", "B-Complex"]
    },
    "Probiotics": {
        "active_ingredients": ["Lactobacillus", "Bifidobacterium", "Enterococcus"],
        "supporting_ingredients": ["Prebiotics", "Digestive Enzymes"]
    },
    "Croquettes au poulet frais": {
        "active_ingredients": ["Fresh Chicken", "Rice", "Vegetables"],
        "supporting_ingredients": ["Vitamins", "Minerals", "Antioxidants"]
    }
}

# Default prompt templates
DEFAULT_BEGINNING_PROMPT = """You are a helpful CX agent for Balto, a premium dog supplement brand.
Please provide a response to the customer's original question based on the Q&A context provided."""

DEFAULT_INSTRUCTIONS_PROMPT = """Instructions:
- Respond in the same language as the customer's original question. Regardless of the language of the Q&A context.
- Use the relevant Q&A context to understand how to address customer's question
- Be concise and professional; follow the tone used in templates and previous answers
- Stay as close as possible to the provided Q&A context. Be helpful, professional, and empathetic.
- Do not make up information or speculate—only answer based on the context and ingredients provided.
- If no relevant answer can be inferred, say so politely and suggest the customer contact our support team."""

# Initialize session state for prompts
if 'beginning_prompt' not in st.session_state:
    st.session_state.beginning_prompt = DEFAULT_BEGINNING_PROMPT
if 'instructions_prompt' not in st.session_state:
    st.session_state.instructions_prompt = DEFAULT_INSTRUCTIONS_PROMPT


# Helper functions
@st.cache_resource
def initialize_clients(google_api_key, pinecone_api_key):
    """Initialize API clients"""
    client = genai.Client(api_key=google_api_key)
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index("socmed-question")
    return client, index

def get_embedding(client, text: str, n_dims: int) -> List[float]:
    """Generate embedding for given text using Gemini."""
    text = text.strip()
    result = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="SEMANTIC_SIMILARITY", 
            outputDimensionality=n_dims
        )
    )
    return result.embeddings[0].values

def retrieve_from_namespace(
    index,
    embedding: List[float],
    namespace: Optional[str] = None,
    product_name: Optional[str] = None,
    question_type: str = "general",
    top_k: int = 10,
    score_threshold: float = 0.9
) -> List[Dict]:
    """Retrieve relevant QnA pairs from a specific namespace."""
    
    # Build filter based on question type
    filter_dict = {}
    if question_type == "product" and product_name:
        filter_dict = {"product_name": {"$in": [product_name, "All"]}}
    
    # Query the index
    results = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
        filter=filter_dict if filter_dict else None
    )
    
    # Filter results by score threshold
    filtered_results = []
    for match in results.matches:
        if match.score >= score_threshold:
            filtered_results.append({
                "id": match.id,
                "score": match.score,
                "question": match.metadata.get("user_question", ""),
                "answer": match.metadata.get("cs_response", ""),
                "product_name": match.metadata.get("product_name", ""),
                "source": namespace
            })
    
    return filtered_results

def combine_results(
    ground_truth_results: List[Dict],
    organic_results: List[Dict],
    max_results: int = 10
) -> List[Dict]:
    """Combine ground truth and organic results, prioritizing ground truth."""
    
    # Start with all ground truth results
    combined = ground_truth_results.copy()
    
    # If we need more results, add from organic
    if len(combined) < max_results:
        # Sort organic results by score descending
        organic_sorted = sorted(organic_results, key=lambda x: x['score'], reverse=True)
        
        # Add organic results until we reach max_results
        for result in organic_sorted:
            if len(combined) >= max_results:
                break
            combined.append(result)
    
    return combined[:max_results]

def get_product_ingredients(product_name: str) -> Optional[Dict]:
    """Get ingredients for a specific product."""
    return product_ingredients.get(product_name, None)

def construct_prompt(
    user_question: str,
    question_type: str,
    qa_context: List[Dict],
    product_name: Optional[str] = None,
    ingredients: Optional[Dict] = None,
    beginning_prompt: str = DEFAULT_BEGINNING_PROMPT,
    instructions_prompt: str = DEFAULT_INSTRUCTIONS_PROMPT
) -> str:
    """Construct prompt for customer service response generation."""
        
    prompt = f"""{beginning_prompt}
    
Customer Original Question: {user_question}
Question Type: {question_type}
"""
        
    if product_name:
        prompt += f"Product Name: {product_name}\n"
        
    prompt += "\nRelevant Q&A Context:"
    for i, qa in enumerate(qa_context, 1):
        prompt += f"\n{i}. Similar Question: {qa['question']}"
        prompt += f"\n   Answer: {qa['answer']}"
        prompt += f"\n   Product: {qa['product_name']}"

    if ingredients:
        prompt += f"\nHere is the composition of this product:\n"
        prompt += f"Active Ingredients: {', '.join(ingredients.get('active_ingredients', []))}\n"
        prompt += f"Supporting Ingredients: {', '.join(ingredients.get('supporting_ingredients', []))}\n"

    prompt += f"\n\n{instructions_prompt}\n"

    if ingredients:
        prompt += "- Use Product composition as a source to help answer the customer's question where relevant.\n"

    prompt += f"\nYour Response:"

    return prompt

def generate_response(client, prompt: str) -> str:
    """Generate customer service response using Gemini."""
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )
    return response.text

def rag_pipeline(
    client,
    index,
    user_query: str,
    product_name: Optional[str] = None,
    question_type: str = "general",
    need_ingredients: bool = False,
    top_k: int = 10,
    ground_truth_threshold: float = 0.84,
    organic_threshold: float = 0.80,
    beginning_prompt: str = DEFAULT_BEGINNING_PROMPT,
    instructions_prompt: str = DEFAULT_INSTRUCTIONS_PROMPT
) -> Dict:
    """Complete RAG pipeline for customer service response generation."""
    # Fixed embedding dimensions
    n_dims = 768
    # Step 1 & 2: Embed user question
    embedding = get_embedding(client, user_query, n_dims)
    
    # Step 3: Retrieve from ground-truth namespace first
    ground_truth_results = retrieve_from_namespace(
        index=index,
        embedding=embedding,
        namespace="ground-truth",
        product_name=product_name,
        question_type=question_type,
        top_k=top_k,
        score_threshold=ground_truth_threshold
    )
    
    # Step 4: If needed, retrieve from organic namespace
    organic_results = []
    if len(ground_truth_results) < top_k:
        organic_results = retrieve_from_namespace(
            index=index,
            embedding=embedding,
            namespace=None,  # Default namespace
            product_name=product_name,
            question_type=question_type,
            top_k=top_k,
            score_threshold=organic_threshold
        )
    
    # Step 5: Combine results
    combined_results = combine_results(ground_truth_results, organic_results, max_results=top_k)
    
    # Step 6: Get ingredients if needed
    ingredients = None
    if need_ingredients and product_name:
        ingredients = get_product_ingredients(product_name)
    
    # Step 7: Construct prompt and generate response
    prompt = construct_prompt(
        user_question=user_query,
        question_type=question_type,
        qa_context=combined_results,
        product_name=product_name,
        ingredients=ingredients,
        beginning_prompt=beginning_prompt,
        instructions_prompt=instructions_prompt
    )
    
    response = generate_response(client, prompt)
    
    return {
        "user_query": user_query,
        "product_name": product_name,
        "question_type": question_type,
        "retrieved_contexts": combined_results,
        "ingredients_used": ingredients is not None,
        "prompt": prompt,
        "response": response,
        "ground_truth_count": len(ground_truth_results),
        "organic_count": len(organic_results),
        "ground_truth_threshold": ground_truth_threshold,
        "organic_threshold": organic_threshold
    }

# Main app
st.title("🐕 Balto Customer Service RAG Demo")
st.markdown("This demo showcases the RAG pipeline for generating customer service responses.")

# Sidebar for API keys and configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Keys section
    with st.expander("🔑 API Keys", expanded=not st.session_state.api_keys_set):
        google_api_key = st.text_input("Google API Key", type="password")
        pinecone_api_key = st.text_input("Pinecone API Key", type="password")
        
        if st.button("Set API Keys"):
            if google_api_key and pinecone_api_key:
                st.session_state.google_api_key = google_api_key
                st.session_state.pinecone_api_key = pinecone_api_key
                st.session_state.api_keys_set = True
                st.success("API keys set successfully!")
                st.rerun()
            else:
                st.error("Please provide both API keys")
    
    # Advanced settings
    st.header("🎛️ Advanced Settings")
    top_k = st.slider("Top K Results", min_value=1, max_value=20, value=10)
    st.subheader("Score Thresholds")
    ground_truth_threshold = st.slider("Ground Truth Threshold", min_value=0.0, max_value=1.0, value=0.84, step=0.01)
    organic_threshold = st.slider("Organic Threshold", min_value=0.0, max_value=1.0, value=0.80, step=0.01)

# Main content
if st.session_state.api_keys_set:
    # Initialize clients
    client, index = initialize_clients(
        st.session_state.google_api_key,
        st.session_state.pinecone_api_key
    )
    
    # Input section
    col1, col2 = st.columns(2)
    
    with col1:
        user_query = st.text_area(
            "Customer Question",
            value="Is Joint Care effective for my dog's arthritis?",
            height=100
        )
        
        question_type = st.selectbox(
            "Question Type",
            options=["general", "product"],
            index=1
        )
    
    with col2:
        product_name = st.selectbox(
            "Product Name",
            options=[None] + list(product_ingredients.keys()),
            index=1
        )
        
        need_ingredients = st.checkbox("Include Product Ingredients", value=True)
    
    # Prompt Customization Section
    st.header("📝 Prompt Customization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Beginning Prompt")
        beginning_prompt = st.text_area(
            "Customize the beginning of the prompt:",
            value=st.session_state.beginning_prompt,
            height=120,
            help="This sets the role and initial context for the AI agent.",
            key="beginning_prompt_input"
        )
        # Update session state when text changes
        st.session_state.beginning_prompt = beginning_prompt
    
    with col2:
        st.subheader("Instructions Prompt")
        instructions_prompt = st.text_area(
            "Customize the instructions section:",
            value=st.session_state.instructions_prompt,
            height=120,
            help="These are the specific instructions for how the AI should respond.",
            key="instructions_prompt_input"
        )
        # Update session state when text changes
        st.session_state.instructions_prompt = instructions_prompt
    
    # Reset buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🔄 Reset Beginning"):
            st.session_state.beginning_prompt = DEFAULT_BEGINNING_PROMPT
            st.rerun()
    with col2:
        if st.button("🔄 Reset Instructions"):
            st.session_state.instructions_prompt = DEFAULT_INSTRUCTIONS_PROMPT
            st.rerun()
    
    # Process button
    if st.button("🚀 Generate Response", type="primary"):
        with st.spinner("Processing..."):
            try:
                # Run the pipeline
                result = rag_pipeline(
                    client=client,
                    index=index,
                    user_query=user_query,
                    product_name=product_name,
                    question_type=question_type,
                    need_ingredients=need_ingredients,
                    top_k=top_k,
                    ground_truth_threshold=ground_truth_threshold,
                    organic_threshold=organic_threshold,
                    beginning_prompt=beginning_prompt,
                    instructions_prompt=instructions_prompt
                )
                
                # Display results
                st.header("📊 Results")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Retrieved Contexts", len(result['retrieved_contexts']))
                with col2:
                    st.metric("Ground Truth Results", result['ground_truth_count'])
                with col3:
                    st.metric("Organic Results", result['organic_count'])
                
                # Response
                st.subheader("🤖 Generated Response")
                st.info(result['response'])
                
                # Debug information
                with st.expander("🔍 Debug Information", expanded=True):
                    # Retrieved contexts
                    st.subheader("Retrieved Contexts")
                    for i, ctx in enumerate(result['retrieved_contexts'], 1):
                        st.markdown(f"**Context {i}** (Score: {ctx['score']:.3f}, Source: {ctx['source']})")
                        st.markdown(f"- **Question:** {ctx['question']}")
                        st.markdown(f"- **Answer:** {ctx['answer']}")
                        st.markdown(f"- **Product:** {ctx['product_name']}")
                        st.divider()
                    
                    # Prompt
                    st.subheader("Generated Prompt")
                    st.code(result['prompt'], language='text')
                
                # Additional info
                with st.expander("ℹ️ Pipeline Information"):
                    st.json({
                        "user_query": result['user_query'],
                        "product_name": result['product_name'],
                        "question_type": result['question_type'],
                        "ingredients_used": result['ingredients_used'],
                        "total_contexts": len(result['retrieved_contexts']),
                        "embedding_dimensions": 768,  # Fixed value
                        "ground_truth_threshold": result['ground_truth_threshold'],
                        "organic_threshold": result['organic_threshold']
                    })
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
else:
    st.warning("⚠️ Please set your API keys in the sidebar to begin.")
    st.info("You need both Google API Key (for Gemini) and Pinecone API Key to run this demo.")

# Footer
st.divider()
st.markdown("Made with ❤️ for Balto Customer Service Team")
