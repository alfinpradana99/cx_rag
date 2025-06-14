import ast
import requests
import streamlit as st
from google import genai
from pinecone import Pinecone
from google.genai import types
from typing import List, Dict, Tuple, Optional

# Page configuration
st.set_page_config(
    page_title="Balto Customer Service RAG Demo",
    page_icon="🐕",
    layout="wide"
)

# Initialize session state for API keys only
if 'api_keys_set' not in st.session_state:
    st.session_state.api_keys_set = False

product_sku_mapping = {
    "Joint Care": "CJF",
    "Anti-itch": "CSI",
    "Calming": "CSR",
    "Probiotics": "CSD"
}

# Default prompt templates
DEFAULT_BEGINNING_PROMPT = """You are a helpful CX agent for Balto, a premium dog supplement brand.
Please provide a response to the customer's original question based on the Q&A context provided."""

DEFAULT_INSTRUCTIONS_PROMPT = """Instructions:
- Respond in the same language as the customer's original question. Regardless of the language of the Q&A context.
- Use the relevant Q&A context to understand how to address customer's question
- Maintain a consistent and professional tone by closely following the style, tone, and structure of past answers and templates.
- Always prioritize relevance to the provided Q&A context and show empathy.
- Respond concisely in one paragraph if possible.
- When referring to the product, use pronouns (it, this, these) instead of repeating the product name.
- Do not make up information or speculate—only answer based on the context and ingredients provided.
- If no relevant answer can be inferred, say so politely and suggest the customer contact our support team.
- NOTE: The contexts are ordered by relevance (cosine similarity score). The earlier contexts are more relevant and should be given more weight in your response.
"""

# Fixed values (no longer editable by users)
FIXED_TOP_K = 10
FIXED_GROUND_TRUTH_THRESHOLD = 0.85
FIXED_ORGANIC_THRESHOLD = 0.85

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

def get_product_by_sku(sku):
    url = "http://cms.balto.fr/api/graphql"
    headers = {
        "Authorization": "users API-Key 8b4a64b7-8de6-4f23-8025-4b0340951e3a",
        "Content-Type": "application/json"
    }
    query = """
    query baltoProducts($skus: [String]) {
      ProductIngredientsAiRags(limit: 100, where: { sku: { in: $skus } }) {
        limit
        totalDocs
        docs {
          sku
          generalInstructions
          dosageGuidelines
          ingredients {
            name
            value
          }
        }
      }
    }
    """
    variables = {
        "skus": [sku]
    }

    response = requests.post(url, headers=headers, json={"query": query, "variables": variables})

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Query failed with status code {response.status_code}: {response.text}")

def retrieve_from_namespace(
    index,
    embedding: List[float],
    namespace: Optional[str] = None,
    product_name: Optional[str] = None,
    question_type: str = "general",
    top_k: int = 10,
    score_threshold: float = 0.85
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
    """Get ingredients for a specific product from API."""
    # Get SKU from product name
    sku = product_sku_mapping.get(product_name)
    if not sku:
        return None
    
    try:
        # Call API
        api_response = get_product_by_sku(sku)
        
        # Extract product data
        docs = api_response.get('data', {}).get('ProductIngredientsAiRags', {}).get('docs', [])
        if not docs:
            return None
        
        product_data = docs[0]  # Get first matching product
        
        # Format ingredients with their values
        ingredients = product_data.get('ingredients', [])
        
        # Create formatted ingredient list with names and values
        formatted_ingredients = []
        for ingredient in ingredients:
            ingredient_name = ingredient.get('name', '')
            ingredient_value = ingredient.get('value', '')
            if ingredient_name and ingredient_value:
                formatted_ingredients.append(f"{ingredient_name} ({ingredient_value})")
            elif ingredient_name:
                formatted_ingredients.append(ingredient_name)
        
        return {
            "ingredients": formatted_ingredients,
            "general_instructions": product_data.get('generalInstructions', ''),
            "dosage_guidelines": product_data.get('dosageGuidelines', ''),
            "raw_ingredients": ingredients,  # Keep raw data for reference
            "sku": product_data.get('sku', '')
        }
        
    except Exception as e:
        st.error(f"Failed to fetch product ingredients: {str(e)}")
        return None

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
        
    prompt += "\nRelevant Q&A Context (ordered by relevance - higher cosine similarity scores indicate more relevant contexts):"
    for i, qa in enumerate(qa_context, 1):
        prompt += f"\n{i}. Similar Question: {qa['question']}"
        prompt += f"\n   Answer: {qa['answer']}"
        prompt += f"\n   Product: {qa['product_name']}"
        prompt += f"\n   Cosine Similarity Score: {qa['score']:.3f}"

    if ingredients:
        prompt += f"\n\nProduct Information for {product_name}:\n"
        
        # Show ingredients with values
        if ingredients.get('ingredients'):
            prompt += f"\nIngredients:\n"
            for ingredient in ingredients.get('ingredients', []):
                prompt += f"- {ingredient}\n"
        
        # Show general instructions if available
        if ingredients.get('general_instructions'):
            prompt += f"\nGeneral Instructions: {ingredients.get('general_instructions')}\n"
        
        # Show dosage guidelines if available
        if ingredients.get('dosage_guidelines'):
            prompt += f"\nDosage Guidelines:\n{ingredients.get('dosage_guidelines')}\n"

    prompt += f"\n\n{instructions_prompt}"

    if ingredients:
        prompt += "- Use product information (ingredients, instructions, dosage) to support the response only where relevant. Do not list all ingredients—mention only those specifically asked about by the customer."
                
    prompt += f"\n\nYour Response:"

    return prompt

def generate_response(client, prompt: str) -> str:
    """Generate customer service response using Gemini."""
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=8192) # can use 0 to 24576, 0 means disable thinking # 1024,2048, 4096, 8192, 16384
        )
    )
    return response.text

def rag_pipeline(
    client,
    index,
    user_query: str,
    product_name: Optional[str] = None,
    question_type: str = "general",
    need_ingredients: bool = False
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
        top_k=FIXED_TOP_K,
        score_threshold=FIXED_GROUND_TRUTH_THRESHOLD
    )
    
    # Step 4: If needed, retrieve from organic namespace
    organic_results = []
    if len(ground_truth_results) < FIXED_TOP_K:
        organic_results = retrieve_from_namespace(
            index=index,
            embedding=embedding,
            namespace=None,  # Default namespace
            product_name=product_name,
            question_type=question_type,
            top_k=FIXED_TOP_K,
            score_threshold=FIXED_ORGANIC_THRESHOLD
        )
    
    # Step 5: Combine results
    combined_results = combine_results(ground_truth_results, organic_results, max_results=FIXED_TOP_K)
    
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
        beginning_prompt=DEFAULT_BEGINNING_PROMPT,  # Always use default
        instructions_prompt=DEFAULT_INSTRUCTIONS_PROMPT  # Always use default
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
        "ground_truth_threshold": FIXED_GROUND_TRUTH_THRESHOLD,
        "organic_threshold": FIXED_ORGANIC_THRESHOLD
    }

# Main app
st.title("🐕 Balto Customer Service RAG Demo")
st.markdown("This demo showcases the RAG pipeline for generating customer service responses.")

# Sidebar for API keys only
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
            options=[None] + list(product_sku_mapping.keys()),
            index=1
        )
        
        need_ingredients = st.checkbox("Include Product Ingredients", value=True)
    
    # Process button
    if st.button("🚀 Generate Response", type="primary"):
        with st.spinner("Processing..."):
            try:
                # Run the pipeline with fixed parameters
                result = rag_pipeline(
                    client=client,
                    index=index,
                    user_query=user_query,
                    product_name=product_name,
                    question_type=question_type,
                    need_ingredients=need_ingredients
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
                        "top_k": FIXED_TOP_K,
                        "ground_truth_threshold": FIXED_GROUND_TRUTH_THRESHOLD,
                        "organic_threshold": FIXED_ORGANIC_THRESHOLD
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
