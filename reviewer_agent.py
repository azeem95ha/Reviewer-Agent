import os
import shutil
import tempfile
import logging
from io import BytesIO
from typing import Dict, List, Optional, Any, Tuple

# External library imports
import streamlit as st
import PIL.Image
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from docx import Document

# Local imports
from classes import State, Title, ChooseBOQ
from helpers import datasheet_content, retrieve_from_vectorstore, generate_report, create_docx_from_markdown

# Setup logging
# Consider adding Streamlit handler if needed, but Cloud logs capture stdout/stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler("submittal_review.log"), # File handler might not persist on Streamlit Cloud ephemeral storage
        logging.StreamHandler() # Logs to console/Streamlit Cloud logs
    ]
)
logger = logging.getLogger("submittal_review")

# --- Constants and Configuration ---

# Load secrets using st.secrets (Preferred for Streamlit Cloud)
# Ensure these secrets are set in the Streamlit Cloud app settings!
GEMINI_MODEL_NAME = st.secrets.get("GEMINI_MODEL_NAME", "gemini-1.5-flash") # Provide a default
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")

VECTORSTORE_PATH = "vectorestores/mzdb" # Assumes this path is in your Git repo
DEFAULT_BOQ_COLLECTION = "boq_FireFighting" # Default if LLM fails or text is empty
DEFAULT_SPECS_COLLECTION = "specifications"
LOGO_PATH = "logo.png" # Assumes this path is in your Git repo

# --- Input Validation ---
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY secret not found. Please set it in Streamlit Cloud secrets.")
    st.error("`GOOGLE_API_KEY` secret is missing. Please configure it in your Streamlit app settings.")
    # Optionally st.stop() here if the app cannot function without it

#####################################
# Model & Tool Initialization
#####################################

# Use st.cache_resource for expensive initializations like models
@st.cache_resource
def initialize_gemini_model(model_name: str, api_key: str) -> Optional[Any]:
    """Initialize and return the Google Gemini model. Cached."""
    if not api_key:
        logger.error("Cannot initialize Gemini model: API key is missing.")
        return None
    if not model_name:
        logger.error("Cannot initialize Gemini model: Model name is missing.")
        return None

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.7, # Example parameter
            # Add safety_settings or other parameters if needed
            # convert_system_message_to_human=True # Uncomment if needed for your model/Langchain version
        )
        logger.info(f"Successfully initialized Gemini model: {model_name}")
        return model
    except ImportError:
        logger.error("Failed to import langchain-google-genai. Please ensure it's in requirements.txt.")
        st.error("Error: Required package `langchain-google-genai` not found.")
        return None
    except Exception as e:
        # Catch potential API errors during initialization (e.g., invalid key)
        logger.error(f"Error initializing Gemini model '{model_name}': {e}")
        st.error(f"Error initializing Google GenAI Model: {e}. Check model name and API key validity.")
        return None

# Initialize session state ONCE at the start
def initialize_session_state():
    """Initialize all necessary session state variables if they don't exist."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True # Flag to run only once per session

        # Report and chat history
        st.session_state.generated_report = None
        st.session_state.chat_history = []
        st.session_state.chat_images = {}

        # --- Initialize Models ---
        st.session_state.gemini_model = initialize_gemini_model(GEMINI_MODEL_NAME, GOOGLE_API_KEY)
        # Use the same instance for chat unless specifically needed otherwise
        st.session_state.google_chat_model = st.session_state.gemini_model

        # --- Placeholder for Vision Model ---
        st.session_state.vision_model = None # Set explicitly to None
        # logger.info("Vision model not initialized (placeholder).") # Optional logging

        # --- Bind Tools (only if base model initialized successfully) ---
        if st.session_state.gemini_model:
            try:
                # Ensure ChooseBOQ is a Pydantic model suitable for tool binding
                tools = [ChooseBOQ]
                # Force the model to use the specified tool
                st.session_state.model_with_tools = st.session_state.gemini_model.bind_tools(
                    tools,
                    tool_choice="ChooseBOQ" # Force calling this specific tool
                    )
                # Ensure Title is a Pydantic model for structured output
                st.session_state.model_with_structured_output = st.session_state.gemini_model.with_structured_output(Title)
                logger.info("Successfully bound tools and structured output to Gemini model")
            except Exception as e:
                logger.error(f"Error binding tools/structured output: {e}")
                st.error(f"Failed to configure model tools: {e}")
                st.session_state.model_with_tools = None
                st.session_state.model_with_structured_output = None
        else:
            logger.error("Base Gemini Model not initialized. Cannot bind tools or structured output.")
            st.error("Base Gemini Model failed to initialize. Tool-based features unavailable.")
            st.session_state.model_with_tools = None
            st.session_state.model_with_structured_output = None

# Call initialization function
initialize_session_state()


#####################################
# LangGraph Node Functions
#####################################

def datasheet_extractor(state: State) -> Dict[str, Any]:
    """Extract text content from the uploaded datasheet."""
    logger.info(f"Node: Extractor - Processing file: {state.get('file_name', 'N/A')}")
    st.info("Extracting text from datasheet...")

    file_name = state.get("file_name")
    if not file_name:
        logger.error("Extractor node received no file_name in state.")
        st.error("Internal error: No file name provided for extraction.")
        return {"submittal_text": "", "error_message": "No file name provided."}

    # Check if model is available for the helper function
    if not st.session_state.gemini_model:
         logger.error("Extractor node: Gemini model not available for datasheet_content helper.")
         st.error("Model not available for text extraction.")
         return {"submittal_text": "", "error_message": "Model unavailable for extraction."}

    try:
        # Assuming datasheet_content handles file reading and returns text or "Error: ..."
        submittal_text = datasheet_content(file_name, st.session_state.gemini_model)

        if submittal_text is None or submittal_text.startswith("Error:"):
            error_msg = submittal_text or "Unknown extraction error."
            error_msg = error_msg.replace('Error: ', '').strip()
            logger.error(f"Datasheet extraction failed: {error_msg}")
            st.error(f"Datasheet extraction failed: {error_msg}")
            return {"submittal_text": "", "error_message": error_msg}

        logger.info(f"Text extraction successful. Length: {len(submittal_text)}")
        st.success("Text extracted.")
        return {"submittal_text": submittal_text, "error_message": None}

    except Exception as e:
        logger.exception(f"Unexpected error during datasheet extraction for {file_name}: {e}")
        st.error(f"An unexpected error occurred during text extraction: {e}")
        return {"submittal_text": "", "error_message": f"Unexpected extraction error: {e}"}


def decide_boq(state: State) -> Dict[str, Any]:
    """Determine the most relevant BOQ discipline using an LLM tool call."""
    logger.info("Node: ChooseBOQ - Deciding relevant BOQ")

    # Skip if previous critical error occurred
    if state.get("error_message"):
        logger.warning("Skipping BOQ decision due to previous error.")
        # Return empty dict to avoid overwriting existing error
        return {}

    # Check if the specific tool-bound model is available
    if not st.session_state.get('model_with_tools'): # Use .get for safety
        logger.error("BOQ decision model (with tools) not available.")
        st.error("Model for BOQ decision is not configured correctly.")
        return {
            "boq_collection": DEFAULT_BOQ_COLLECTION, # Fallback
            "specs_collection": DEFAULT_SPECS_COLLECTION,
            "error_message": "BOQ decision model not available"
        }

    st.info("Deciding relevant BOQ discipline...")
    submittal_text = state.get("submittal_text", "")
    # Initialize fallbacks
    boq_collection_name = DEFAULT_BOQ_COLLECTION
    specs_collection_name = DEFAULT_SPECS_COLLECTION

    try:
        if not submittal_text:
            logger.warning("Submittal text is empty. Using default BOQ.")
            st.warning("Submittal text is empty. Using default BOQ.")
            # No error message needed here, just proceed with default
        else:
            # Prepare the prompt for the tool-using model
            # Ensure the prompt clearly asks the model to USE the tool
            prompt = (
                f"Given the following datasheet text, use the 'ChooseBOQ' tool to select the single most relevant "
                f"BOQ discipline from the allowed options (boq_FireFighting, boq_Electrical, boq_HVAC, boq_Plumbing). "
                f"Datasheet Text:\n```\n{submittal_text[:4000]}\n```" # Limit text length if needed
            )
            logger.info(f"Invoking LLM for BOQ decision with text length: {len(submittal_text)}")

            # Invoke the model bound with the tool
            response = st.session_state.model_with_tools.invoke(prompt)
            logger.info(f"LLM Raw Response for BOQ decision: {response!r}") # Log the raw response

            # --- Robust validation of the response ---
            if not response.tool_calls:
                logger.error("LLM response did NOT contain any tool calls!")
                # Decide how to handle: Use default or raise specific error? Using default here.
                st.warning("LLM failed to select a BOQ discipline using the required tool. Using default.")
                # Keep boq_collection_name as the default
            else:
                # Assume the first tool call is the one we forced with tool_choice
                tool_call = response.tool_calls[0]
                logger.info(f"Tool Call Received: {tool_call!r}")

                # Validate the tool call structure
                if tool_call.get('name') != 'ChooseBOQ':
                     logger.warning(f"LLM called unexpected tool '{tool_call.get('name')}'. Expected 'ChooseBOQ'. Using default.")
                     st.warning("LLM called an unexpected tool. Using default BOQ.")
                elif 'args' not in tool_call or not isinstance(tool_call.get('args'), dict):
                    logger.error(f"Tool call missing 'args' dictionary: {tool_call}")
                    st.warning("LLM tool call structure is invalid (missing 'args'). Using default BOQ.")
                elif 'boq_name' not in tool_call['args']:
                    logger.error(f"Tool call 'args' missing 'boq_name': {tool_call['args']}")
                    st.warning("LLM tool call arguments are invalid (missing 'boq_name'). Using default BOQ.")
                else:
                    # Safely extract the boq_name
                    selected_boq = tool_call['args']['boq_name']
                    # Optional: Validate against allowed list
                    allowed_boqs = ["boq_FireFighting", "boq_Electrical", "boq_HVAC", "boq_Plumbing"]
                    if selected_boq in allowed_boqs:
                        boq_collection_name = selected_boq
                        logger.info(f"LLM selected BOQ: {boq_collection_name}")
                        st.success(f"Selected BOQ discipline: {boq_collection_name}")
                    else:
                        logger.warning(f"LLM selected invalid BOQ '{selected_boq}'. Allowed: {allowed_boqs}. Using default.")
                        st.warning(f"LLM selected an invalid BOQ choice ('{selected_boq}'). Using default.")

        # Return successful state (using determined or default BOQ)
        return {
            "boq_collection": boq_collection_name,
            "specs_collection": specs_collection_name, # Assuming specs collection is always default for now
            "error_message": None # Clear any previous non-critical warnings implicitly
        }

    except Exception as e:
        # Catch API errors, parsing errors, etc.
        logger.exception(f"Error deciding BOQ: {e}. Using default: {DEFAULT_BOQ_COLLECTION}")
        st.error(f"An error occurred during BOQ decision: {e}. Using default: {DEFAULT_BOQ_COLLECTION}")

        # Fallback to default and record the error
        return {
            "boq_collection": DEFAULT_BOQ_COLLECTION,
            "specs_collection": DEFAULT_SPECS_COLLECTION,
            "error_message": f"BOQ decision failed: {e}"
        }


def retriever(state: State) -> Dict[str, Any]:
    """Retrieve relevant documents from vector stores."""
    logger.info("Node: Retriever - Retrieving documents")

    # Skip if previous critical error occurred
    if state.get("error_message"):
        logger.warning("Skipping document retrieval due to previous error.")
        return {}

    st.info("Retrieving documents from vector store...")
    # Get collection names from state, falling back to defaults
    boq_collection_name = state.get("boq_collection", DEFAULT_BOQ_COLLECTION)
    specs_collection_name = state.get("specs_collection", DEFAULT_SPECS_COLLECTION)
    submittal_text = state.get("submittal_text", "")

    if not submittal_text:
        logger.warning("Retriever: Submittal text is empty. Retrieval might be ineffective.")
        st.warning("Submittal text is empty, document retrieval may not find relevant results.")
        # Proceed anyway, maybe retrieve general docs? Or return empty? Returning empty for now.
        # return {"retrieved_docs": [[], []], "error_message": None} # Option 1: Return empty
        # Option 2: Proceed (below)

    logger.info(f"Retrieving documents using: BOQ='{boq_collection_name}', Specs='{specs_collection_name}'")

    # --- Check if vector store path exists (critical for Cloud deployment) ---
    if not os.path.exists(VECTORSTORE_PATH):
        error_msg = f"Vector store path not found: '{VECTORSTORE_PATH}'. Ensure it's included in your repository."
        logger.error(error_msg)
        st.error(error_msg)
        return {
            "retrieved_docs": [[], []], # Empty results
            "error_message": "Vector store path invalid"
        }
    # --- Check complete ---

    try:
        # Assuming retrieve_from_vectorstore handles ChromaDB/FAISS initialization and querying
        retrieved_data = retrieve_from_vectorstore(
            VECTORSTORE_PATH,
            boq_collection_name,
            specs_collection_name,
            submittal_text
        )

        # Basic validation of returned data structure
        if not isinstance(retrieved_data, list) or len(retrieved_data) != 2 or \
           not isinstance(retrieved_data[0], list) or not isinstance(retrieved_data[1], list):
             logger.error(f"Unexpected data structure returned from retrieve_from_vectorstore: {type(retrieved_data)}")
             st.error("Internal error: Received invalid data from document retrieval.")
             return {"retrieved_docs": [[], []], "error_message": "Invalid retrieval data format."}

        logger.info(f"Retrieved {len(retrieved_data[0])} BOQ docs and {len(retrieved_data[1])} spec docs.")
        st.success("Document retrieval complete.")

        return {"retrieved_docs": retrieved_data, "error_message": None} # Clear potential previous non-critical error

    except Exception as e:
        logger.exception(f"Error retrieving documents: {e}") # Log traceback
        st.error(f"An error occurred during document retrieval: {e}")

        return {
            "retrieved_docs": [[], []], # Empty results on failure
            "error_message": f"Retrieval failed: {e}"
        }


def report_generator(state: State) -> Dict[str, Any]:
    """Generate the final report based on retrieved documents and submittal text."""
    logger.info("Node: ReportGenerator - Generating final report")

    # Don't generate report if critical errors occurred earlier
    if state.get("error_message"):
        error_msg = state["error_message"]
        logger.error(f"Report generation skipped due to previous error: {error_msg}")
        st.error(f"Report generation skipped due to previous error: {error_msg}")
        # Return the error message as the report content
        return {"final_report": f"Report Generation Failed.\n\nPrevious Error: {error_msg}"}

    st.info("Generating the final report...")
    retrieved_docs = state.get("retrieved_docs", [[], []]) # Default to empty lists
    submittal_text = state.get("submittal_text", "")

    # Validate retrieved_docs structure again just in case
    boq_retrieved_docs = retrieved_docs[0] if isinstance(retrieved_docs, list) and len(retrieved_docs) > 0 and isinstance(retrieved_docs[0], list) else []
    specs_retrieved_docs = retrieved_docs[1] if isinstance(retrieved_docs, list) and len(retrieved_docs) > 1 and isinstance(retrieved_docs[1], list) else []

    logger.info(f"Generating report with {len(boq_retrieved_docs)} BOQ docs, {len(specs_retrieved_docs)} spec docs.")

    # Check for necessary inputs
    if not submittal_text:
        logger.error("Cannot generate report: Submittal text is missing.")
        st.error("Cannot generate report: Submittal text was not extracted correctly.")
        return {
            "final_report": "Report Generation Failed: Missing submittal text.",
            "error_message": "Missing submittal text for report generation."
            }

    if not st.session_state.gemini_model:
         logger.error("Report generator: Gemini model not available.")
         st.error("Model not available for report generation.")
         return {
            "final_report": "Report Generation Failed: Model unavailable.",
            "error_message": "Model unavailable for report generation."
            }

    try:
        # Assuming generate_report handles LLM calls and formatting
        final_report = generate_report(
            boq_retrieved_docs,
            specs_retrieved_docs,
            submittal_text,
            st.session_state.gemini_model # Pass the initialized model
        )

        if final_report is None or final_report.strip() == "":
             logger.warning("Report generation resulted in an empty report.")
             st.warning("Report generation resulted in an empty report.")
             final_report = "Report generation completed, but the result was empty."
             # Decide if this is an error state or not
             # return {"final_report": final_report, "error_message": "Generated empty report."}

        logger.info("Report generation successful.")
        st.success("Final report generated.")
        return {"final_report": final_report, "error_message": None} # Success

    except Exception as e:
        logger.exception(f"Error generating report: {e}")
        st.error(f"An error occurred during report generation: {e}")
        final_report_content = f"Error during report generation: {e}"
        return {
            "final_report": final_report_content,
            "error_message": f"Report generation failed: {e}"
        }


#####################################
# LangGraph Definition
#####################################

# Use caching for the compiled graph structure
@st.cache_resource
def build_langgraph() -> Optional[Any]:
    """Build and compile the LangGraph. Cached."""
    # Check essential components needed for the graph nodes
    if not st.session_state.get('gemini_model'):
        logger.error("Cannot build graph - Base Gemini model unavailable.")
        st.error("Cannot build analysis graph: Base model is not initialized.")
        return None
    if not st.session_state.get('model_with_tools'):
         logger.error("Cannot build graph - Model with tools unavailable (needed for ChooseBOQ node).")
         st.error("Cannot build analysis graph: Model with tools is not configured.")
         return None
    # Add checks for other models if nodes depend on them explicitly

    logger.info("Building submittal analysis graph...")
    try:
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("Extractor", datasheet_extractor)
        graph_builder.add_node("ChooseBOQ", decide_boq)
        graph_builder.add_node("Retriever", retriever)
        graph_builder.add_node("ReportGenerator", report_generator)

        # Define edges (workflow)
        graph_builder.add_edge(START, "Extractor")
        graph_builder.add_edge("Extractor", "ChooseBOQ")
        graph_builder.add_edge("ChooseBOQ", "Retriever")
        graph_builder.add_edge("Retriever", "ReportGenerator")
        graph_builder.add_edge("ReportGenerator", END)

        # Compile the graph
        # Consider adding interrupt handlers if needed for long runs:
        # from langgraph.interrupt import np, EphemeralInterrupt
        # worker = graph_builder.compile(checkpointer=..., interrupt_before=..., interrupt_after=...)
        worker = graph_builder.compile()
        logger.info("Analysis graph compiled successfully.")
        return worker
    except Exception as e:
        logger.exception(f"Fatal error compiling LangGraph: {e}") # Log traceback
        st.error(f"Fatal Error: Could not compile the analysis workflow graph: {e}")
        return None

#####################################
# Chat Functions (Revised for Clarity & Placeholders)
#####################################

def setup_chat_interface(report_text: str) -> None:
    """Set up and manage the chat interface."""
    if "chat_images" not in st.session_state:
        st.session_state.chat_images = {} # Should be initialized earlier, but safe check

    # Image upload section (Keep simple)
    with st.expander("Upload an image to discuss (Vision Model Not Enabled)", expanded=False):
        st.warning("Note: Image analysis requires a Vision model, which is not currently enabled in this version.")
        uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="chat_image_uploader")

        if uploaded_image is not None:
            image = PIL.Image.open(uploaded_image)
            st.image(image, caption=f"Uploaded: {uploaded_image.name}", use_container_width=True)
            image_id = f"img_{len(st.session_state.chat_images) + 1}"
            # Store image reference, not the full image data repeatedly unless needed
            st.session_state.chat_images[image_id] = {
                # "image": image, # Consider storing only path/bytes if memory becomes an issue
                "filename": uploaded_image.name
            }
            st.info(f"Image '{uploaded_image.name}' noted. You can ask questions, but visual analysis is disabled.")

    # Display chat history
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.write(msg.content) # Use write for potential markdown/formatting

    # Chat input processing
    if user_query := st.chat_input("Ask a question about the report..."):
        # Add user message to history and display
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("user"):
            st.write(user_query)

        logger.info(f"Chat query: {user_query[:50]}...")

        # Define system message (adapt based on whether images are present/relevant)
        has_images = bool(st.session_state.chat_images)
        if has_images:
             system_message = ("You are a helpful assistant. Answer questions based on the provided report context. "
                               "The user has also uploaded images, but you currently cannot analyze their visual content. "
                               "Acknowledge image references if mentioned, but state you cannot see them.")
        else:
            system_message = ("You are a helpful assistant. Answer questions based on the provided report context. "
                              "You can also answer general questions not related to the report. Do not make up information.")

        # Prepare prompt messages
        prompt_messages = [
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history_placeholder"), # Use placeholder name
            ("human", "Report Context:\n```\n{report_context}\n```\n\nQuestion: {question}")
        ]
        prompt_template = ChatPromptTemplate.from_messages(prompt_messages)

        # Define the RAG chain
        # Ensure the chat model instance is available
        if not st.session_state.google_chat_model:
             st.error("Chat model is not available. Cannot process query.")
             logger.error("Chat query failed: Chat model unavailable.")
             return # Exit if model missing

        rag_chain = (
            RunnablePassthrough.assign(
                # Prepare context for the prompt template
                report_context=lambda x: x["report_context"],
                question=lambda x: x["question"],
                # Pass history under the placeholder name used in the template
                chat_history_placeholder=lambda x: x["chat_history"]
            )
            | prompt_template
            | st.session_state.google_chat_model
            | StrOutputParser()
        )

        # Invoke the chain
        with st.spinner("Thinking..."):
            try:
                chain_input = {
                    "report_context": report_text,
                    "question": user_query,
                    "chat_history": st.session_state.chat_history # Provide the actual history list here
                }
                logger.info("Invoking RAG chain for chat response")
                response = rag_chain.invoke(chain_input)
                logger.info(f"Generated chat response length: {len(response)}")

                # Add AI response to history and display
                st.session_state.chat_history.append(AIMessage(content=response))
                with st.chat_message("assistant"):
                    st.write(response) # Use write for potential markdown

            except Exception as e:
                logger.exception(f"Error processing chat query: {e}")
                st.error(f"Sorry, I encountered an error: {e}")
                # Add error message to chat history as well
                error_msg_for_chat = f"Sorry, I encountered an error processing your request: {e}"
                st.session_state.chat_history.append(AIMessage(content=error_msg_for_chat))
                # Display error in chat interface too
                with st.chat_message("assistant"):
                    st.error(error_msg_for_chat)


# Placeholder function - Not used actively unless vision_model is initialized
def analyze_image(image):
    """Placeholder for image analysis using a vision model."""
    if st.session_state.vision_model:
        try:
            # Convert PIL image to bytes or required format for your vision model
            buffered = BytesIO()
            image.save(buffered, format="JPEG") # Or PNG, check model requirements
            img_bytes = buffered.getvalue()
            # Example call (replace with actual vision model API)
            # description = st.session_state.vision_model.invoke([HumanMessage(content=[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode()}"}}, {"type": "text", "text": "Describe this image."}])])
            # return description.content
            logger.info("analyze_image called, but using placeholder response.")
            return "Placeholder: Image analysis requires an initialized vision model."
        except Exception as e:
            logger.error(f"Error in placeholder analyze_image: {e}")
            return f"Error during placeholder image analysis: {e}"
    else:
        logger.warning("analyze_image called, but no vision model is initialized.")
        return "Vision model not available for analysis."


#####################################
# UI Components
#####################################

def setup_sidebar() -> Optional[str]:
    """Set up the sidebar. Returns the selected app mode."""
    try:
        image = PIL.Image.open(LOGO_PATH)
        st.sidebar.image(image, use_container_width=True)
    except FileNotFoundError:
        logger.warning(f"Logo file not found at: {LOGO_PATH}")
        # st.sidebar.warning("Logo file not found.")
    except Exception as e:
        logger.error(f"Could not load logo '{LOGO_PATH}': {e}")
        # st.sidebar.error("Error loading logo.")

    st.sidebar.title("Navigation")
    # Use keys for widgets that might re-render
    app_mode = st.sidebar.radio(
        "Choose a section:",
        ("Submittal Analysis", "Chat with Report"),
        key="app_mode_radio"
    )

    st.sidebar.markdown("---")
    # Add any other sidebar elements here (e.g., status indicators)
    if not st.session_state.gemini_model:
        st.sidebar.error("⚠️ Model Initialization Failed")


    return app_mode

def submittal_analysis_page(worker: Optional[Any]) -> None:
    """Render the submittal analysis page interface."""
    st.title("Submittal Review Agent")
    st.markdown("Upload a submittal datasheet (PDF) to generate a review report.")

    uploaded_file = st.file_uploader(
        "Choose a datasheet file",
        type=["pdf"],
        key="datasheet_uploader" # Add a key
    )

    # Disable button if worker (graph) isn't compiled or no file uploaded
    button_disabled = (worker is None or uploaded_file is None)
    analyze_pressed = st.button("Analyze Submittal", key="analyze_button", disabled=button_disabled)

    if analyze_pressed and uploaded_file is not None:
        temp_file_path = None # Initialize path variable
        try:
            # Create a temporary file safely
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_file_path = temp_file.name # Get the path
            logger.info(f"Uploaded file saved temporarily to: {temp_file_path}")

            if worker is None: # Should be caught by disabled state, but double check
                logger.error("Analyze button pressed, but worker is None.")
                st.error("Analysis engine is not ready. Please refresh or check logs.")
                return # Exit early

            st.info(f"Starting analysis for: {uploaded_file.name}")
            initial_state = {
                "file_name": temp_file_path,
                # Assuming specs collection is always default unless chosen differently
                "specs_collection": DEFAULT_SPECS_COLLECTION,
                "error_message": None # Start clean
            }

            with st.spinner("⚙️ Running analysis workflow... Please wait."):
                # Clear previous results before running
                st.session_state.generated_report = None
                st.session_state.chat_history = []
                st.session_state.chat_images = {} # Clear images too

                logger.info("Invoking analysis worker...")
                final_state = worker.invoke(initial_state)
                logger.info(f"Analysis worker finished. Final state keys: {final_state.keys()}")

            # --- Process Final State ---
            final_report_text = final_state.get("final_report", "Report could not be generated.")
            analysis_error = final_state.get("error_message") # Check for errors from the graph run

            if analysis_error:
                st.error(f"Analysis completed with errors: {analysis_error}")
                logger.error(f"Analysis workflow completed with error: {analysis_error}")
                # Display the potentially partial/error report
                st.subheader("Analysis Result (with errors)")
            else:
                st.success("✅ Analysis Complete!")
                st.subheader("Generated Report")
                # Store report for chat page ONLY if successful
                st.session_state.generated_report = final_report_text

            # Display the report content regardless of error (might contain error message)
            # Handle potential markdown in report
            if isinstance(final_report_text, str):
                # Clean potential markdown code fences if model adds them unnecessarily
                cleaned_report = final_report_text.strip()
                if cleaned_report.startswith("```markdown"):
                    cleaned_report = cleaned_report[len("```markdown"):].strip()
                if cleaned_report.endswith("```"):
                    cleaned_report = cleaned_report[:-len("```")].strip()
                st.markdown(cleaned_report) # Display cleaned report

                # --- Download Button (only if successful potentially) ---
                if not analysis_error and cleaned_report: # Only offer download for successful, non-empty reports
                    try:
                        docx_buffer = BytesIO()
                        # Assuming create_docx_from_markdown works correctly
                        create_docx_from_markdown(cleaned_report, docx_buffer)
                        docx_buffer.seek(0)

                        st.download_button(
                            label="Download Report as DOCX",
                            data=docx_buffer,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_report.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key="download_docx"
                        )
                    except Exception as docx_error:
                         logger.error(f"Error creating DOCX report: {docx_error}")
                         st.warning(f"Could not generate DOCX download: {docx_error}")
            else:
                 st.warning("Generated report content is not in the expected text format.")


        except Exception as e:
            # Catch errors during the invoke process itself or file handling
            logger.exception(f"Unexpected error during analysis execution: {e}")
            st.error(f"An unexpected error occurred: {e}")

        finally:
            # --- Robust Temporary File Cleanup ---
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.info(f"Cleaned up temporary file: {temp_file_path}")
                except OSError as cleanup_error:
                    logger.error(f"Error removing temporary file {temp_file_path}: {cleanup_error}")


    elif analyze_pressed and uploaded_file is None:
        st.warning("Please upload a file first.")

    # Display message if worker failed to initialize
    if worker is None and not st.session_state.gemini_model: # Check if model init was the root cause
        st.error("Analysis engine could not be initialized. Please ensure the `GOOGLE_API_KEY` secret is set correctly in Streamlit Cloud and the model name is valid.")
    elif worker is None:
         st.error("Analysis workflow failed to compile. Check logs for details.")


def chat_with_report_page() -> None:
    """Render the chat interface."""
    st.title("Chat with Generated Report")

    if st.session_state.generated_report is None:
        st.warning("📈 Please run the 'Submittal Analysis' first to generate a report.")
        logger.warning("Chat page accessed without a generated report.")
        return # Don't proceed

    if not st.session_state.google_chat_model:
        st.error("💬 Chat model is not available. Cannot start chat.")
        logger.error("Chat page accessed but chat model is unavailable.")
        return # Don't proceed

    st.markdown("Ask questions about the report generated in the 'Submittal Analysis' section.")
    logger.info("Setting up chat interface...")
    setup_chat_interface(st.session_state.generated_report)
