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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("submittal_review.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("submittal_review")

# Load environment variables
load_dotenv()
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Constants
VECTORSTORE_PATH = "vectorestores/chromadb"
DEFAULT_BOQ_COLLECTION = "boq"
DEFAULT_SPECS_COLLECTION = "specifications_MEP"
LOGO_PATH = "logo.png"

#####################################
# Model & Tool Initialization
#####################################

def initialize_gemini_model():
    """Initialize and return the Google Gemini model."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = ChatGoogleGenerativeAI(temperature=0.7, model=GEMINI_MODEL_NAME)
        logger.info(f"Successfully initialized Gemini model: {GEMINI_MODEL_NAME}")
        return model
    except ImportError:
        logger.error("Failed to import langchain-google-genai. Please install it.")
        st.error("Please install langchain-google-genai: pip install langchain-google-genai")
        return None
    except Exception as e:
        logger.error(f"Error initializing Gemini model: {e}")
        st.error(f"Error initializing Google GenAI Model: {e}. Make sure GOOGLE_API_KEY is valid.")
        return None


def initialize_session_state():
    """Initialize all necessary session state variables."""
    # Report and chat history
    if "generated_report" not in st.session_state:
        st.session_state.generated_report = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Models and tools
    if "google_chat_model" not in st.session_state:
        st.session_state.google_chat_model = initialize_gemini_model()
    
    if "gemini_model" not in st.session_state:
        st.session_state.gemini_model = initialize_gemini_model()
    
    
    # Bind tools to model if available
    if st.session_state.gemini_model:
        tools = [ChooseBOQ]
        st.session_state.model_with_tools = st.session_state.gemini_model.bind_tools(tools, tool_choice="ChooseBOQ")
        st.session_state.model_with_structured_output = st.session_state.gemini_model.with_structured_output(Title)
        logger.info("Successfully bound tools to Gemini model")
    else:
        logger.error("Gemini Model not initialized. Cannot bind tools.")
        st.error("Gemini Model not initialized. Cannot bind tools.")
        st.session_state.model_with_tools = None
        st.session_state.model_with_structured_output = None

#####################################
# LangGraph Node Functions
#####################################

def datasheet_extractor(state: State) -> Dict[str, Any]:
    """
    Extract text content from the uploaded datasheet.
    
    Args:
        state: Current state containing file_name
        
    Returns:
        Dictionary with extracted text or error message
    """
    logger.info(f"Extracting text from datasheet: {state['file_name']}")
    st.info("Extracting text from datasheet...")
    
    file_name = state["file_name"]
    submittal_text = datasheet_content(file_name, st.session_state.gemini_model)
    
    if "Error:" in submittal_text:
        error_msg = submittal_text.replace('Error: ', '')
        logger.error(f"Datasheet extraction failed: {error_msg}")
        st.error(f"Datasheet extraction failed: {error_msg}")
        return {"submittal_text": "", "error_message": submittal_text}
    
    logger.info("Text extraction successful")
    st.success("Text extracted.")
    return {"submittal_text": submittal_text, "error_message": None}


def decide_boq(state: State) -> Dict[str, Any]:
    """
    Determine the most relevant BOQ discipline based on submittal text.
    
    Args:
        state: Current state containing submittal_text
        
    Returns:
        Dictionary with selected BOQ and specs collection names
    """
    # Skip if previous error occurred
    if state.get("error_message"):
        logger.warning("Skipping BOQ decision due to previous error")
        return {}
    
    # Check if model is available
    if not st.session_state.model_with_tools:
        logger.error("BOQ model not available")
        return {
            "boq_collection": "Model Error", 
            "specs_collection": "Model Error", 
            "error_message": "BOQ model not available"
        }

    st.info("Deciding relevant BOQ discipline...")
    submittal_text = state["submittal_text"]
    boq_collection_name = DEFAULT_BOQ_COLLECTION
    specs_collection_name = DEFAULT_SPECS_COLLECTION
    
    try:
        if not submittal_text:
            logger.warning("Submittal text is empty. Using default BOQ")
            st.warning("Submittal text is empty. Using default BOQ")
        else:
            # Use LLM to determine the appropriate BOQ
            prompt = f"Choose the single most relevant BOQ discipline (boq, boq_Electrical, boq_HVAC, boq_Plumbing) for the following datasheet text: {submittal_text}"
            logger.info(f"Invoking LLM for BOQ decision with text length: {len(submittal_text)}")
            
            tool_call = st.session_state.model_with_tools.invoke(prompt).tool_calls[0]
            boq_collection_name = tool_call['args']['boq_name']
            logger.info(f"Selected BOQ: {boq_collection_name}")
            
        return {
            "boq_collection": boq_collection_name, 
            "specs_collection": specs_collection_name, 
            "error_message": None
        }
    except Exception as e:
        logger.error(f"Error deciding BOQ: {e}. Using default: {boq_collection_name}")
        st.error(f"Error deciding BOQ: {e}. Using default: {boq_collection_name}")
        
        # Fallback to default if LLM fails
        return {
            "boq_collection": boq_collection_name, 
            "specs_collection": specs_collection_name, 
            "error_message": f"BOQ decision failed: {e}"
        }


def retriever(state: State) -> Dict[str, Any]:
    """
    Retrieve relevant documents from vector stores based on submittal text.
    
    Args:
        state: Current state containing collection names and submittal text
        
    Returns:
        Dictionary with retrieved documents
    """
    # Skip if previous error occurred
    if state.get("error_message"):
        logger.warning("Skipping document retrieval due to previous error")
        return {}
    
    st.info("Retrieving documents from vector store...")
    boq_collection_name = state.get("boq_collection", DEFAULT_BOQ_COLLECTION)
    specs_collection_name = state.get("specs_collection", DEFAULT_SPECS_COLLECTION)
    submittal_text = state.get("submittal_text", "")
    
    logger.info(f"Retrieving documents from: BOQ={boq_collection_name}, Specs={specs_collection_name}")
    
    # Check if vector store path exists
    if not os.path.exists(VECTORSTORE_PATH):
        logger.error(f"Vector store path not found: {VECTORSTORE_PATH}")
        st.error(f"Vector store path not found: {VECTORSTORE_PATH}")
        return {
            "retrieved_docs": [[], []],  # Empty results
            "error_message": "Vector store path invalid"
        }
    
    try:
        # Perform document retrieval
        retrieved_data = retrieve_from_vectorstore(
            VECTORSTORE_PATH, 
            boq_collection_name, 
            specs_collection_name, 
            submittal_text
        )
        
        logger.info(f"Retrieved {len(retrieved_data[0])} BOQ docs and {len(retrieved_data[1])} spec docs")
        st.success("Document retrieval complete.")
        
        return {"retrieved_docs": retrieved_data}
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        st.error(f"Error retrieving documents: {e}")
        
        return {
            "retrieved_docs": [[], []],  # Empty results 
            "error_message": f"Retrieval failed: {e}"
        }


def report_generator(state: State) -> Dict[str, Any]:
    """
    Generate the final report based on retrieved documents and submittal text.
    
    Args:
        state: Current state containing retrieved documents and submittal text
        
    Returns:
        Dictionary with the final generated report
    """
    # Don't generate report if critical errors occurred
    if state.get("error_message"):
        error_msg = state.get("error_message", "Unknown error")
        logger.error(f"Report generation skipped due to previous error: {error_msg}")
        st.error(f"Report generation skipped due to previous error: {error_msg}")
        
        return {"final_report": f"Report generation failed due to error: {error_msg}"}

    st.info("Generating the final report...")
    retrieved_docs = state.get("retrieved_docs", [[], []])
    boq_retrieved_docs = retrieved_docs[0] if retrieved_docs and len(retrieved_docs) > 0 else []
    specs_retrieved_docs = retrieved_docs[1] if retrieved_docs and len(retrieved_docs) > 1 else []
    submittal_text = state.get("submittal_text", "N/A")
    
    logger.info(f"Generating report with {len(boq_retrieved_docs)} BOQ docs, {len(specs_retrieved_docs)} spec docs")
    
    try:
        # Check for valid inputs before calling generate
        if submittal_text == "N/A" or submittal_text == "":
            logger.warning("Cannot generate report without submittal text")
            st.warning("Cannot generate report without submittal text.")
            final_report = "Report generation failed: Missing submittal text."
        else:
            final_report = generate_report(
                boq_retrieved_docs, 
                specs_retrieved_docs, 
                submittal_text,
                st.session_state.gemini_model
            )
            logger.info("Report generation successful")
            st.success("Final report generated.")
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        st.error(f"Error generating report: {e}")
        final_report = f"Error during report generation: {e}"
        
        return {
            "final_report": final_report, 
            "error_message": f"Report generation failed: {e}"
        }

    return {"final_report": final_report}

#####################################
# LangGraph Definition
#####################################

def build_langgraph() -> Optional[Any]:
    """
    Build and compile the LangGraph for submittal analysis.
    
    Returns:
        Compiled graph worker or None if compilation fails
    """
    if not (st.session_state.gemini_model):
        logger.error("Cannot build graph - models or tools unavailable")
        return None
    
    logger.info("Building submittal analysis graph")
    try:
        # Create the state graph
        graph_builder = StateGraph(State)
        
        # Add nodes
        graph_builder.add_node("Extractor", datasheet_extractor)
        graph_builder.add_node("ChooseBOQ", decide_boq)
        graph_builder.add_node("Retriever", retriever)
        graph_builder.add_node("ReportGenerator", report_generator)
        
        # Connect the graph
        graph_builder.add_edge(START, "Extractor")
        graph_builder.add_edge("Extractor", "ChooseBOQ")
        graph_builder.add_edge("ChooseBOQ", "Retriever")
        graph_builder.add_edge("Retriever", "ReportGenerator")
        graph_builder.add_edge("ReportGenerator", END)
        
        # Compile the graph
        worker = graph_builder.compile()
        logger.info("Analysis graph compiled successfully")
        return worker
    except Exception as e:
        logger.error(f"Error compiling graph: {e}")
        st.sidebar.error(f"Error compiling graph: {e}")
        return None

#####################################
# Chat Functions
#####################################

def setup_chat_interface(report_text: str) -> None:
    """
    Set up and manage the chat interface for interacting with the report.
    
    Args:
        report_text: The generated report text to chat about
    """
    # Display existing chat messages
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)

    # Chat input field
    if user_query := st.chat_input("Ask a question about the report..."):
        # Add user message to history and display
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.chat_message("user").write(user_query)
        
        logger.info(f"New chat query: {user_query[:50]}{'...' if len(user_query) > 50 else ''}")

        # Prepare the prompt template for the chat
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer the user's questions considering the provided report context "
                      "(answer also the questions are not related to the report). Do not make up information."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Given the following report:\n\n---\nREPORT START\n---\n{report_context}\n---\nREPORT END\n---\n\n"
                     "Answer this question: {question}")
        ])

        # Create the RAG chain that includes history and context
        rag_chain = (
            RunnablePassthrough.assign(
                report_context=lambda x: x["report_context"],
                question=lambda x: x["question"],
                chat_history=lambda x: x["chat_history"]
            )
            | prompt_template
            | st.session_state.google_chat_model
            | StrOutputParser()
        )

        # Invoke the chain with the report context and query
        with st.spinner("Thinking..."):
            try:
                # Prepare input dictionary for the chain
                chain_input = {
                    "report_context": report_text,
                    "question": user_query,
                    "chat_history": st.session_state.chat_history
                }
                
                logger.info("Invoking RAG chain for chat response")
                response = rag_chain.invoke(chain_input)
                logger.info(f"Generated chat response of length: {len(response)}")

                # Add AI response to history and display
                st.session_state.chat_history.append(AIMessage(content=response))
                st.chat_message("assistant").write(response)

            except Exception as e:
                logger.error(f"Error processing chat query: {e}")
                st.error(f"Error processing chat query: {e}")
                
                # Add an error message to chat history
                error_msg = f"Sorry, I encountered an error: {e}"
                st.session_state.chat_history.append(AIMessage(content=error_msg))
                st.chat_message("assistant").write(error_msg)

#####################################
# UI Components
#####################################

def setup_sidebar() -> None:
    """Set up the sidebar with navigation and information."""
    # Load and display logo
    try:
        image = PIL.Image.open(LOGO_PATH)
        st.sidebar.image(image, use_column_width=True)
    except Exception as e:
        logger.warning(f"Could not load logo: {e}")
    
    # Navigation section
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a section:", ("Submittal Analysis", "Chat with Report"))
    
    # Footer
    st.sidebar.markdown("---")
    
    return app_mode

def submittal_analysis_page(worker: Optional[Any]) -> None:
    """
    Render the submittal analysis page interface.
    
    Args:
        worker: The compiled LangGraph worker
    """
    st.title("Submittal Review Agent")
    st.markdown("Enter the file path of the submittal datasheet to generate a review report.")
    
    uploaded_file = st.file_uploader("Choose a datasheet file", type=["pdf"])
    
    # Analysis button
    button_disabled = (worker is None or uploaded_file is None)
    if st.button("Analyze Submittal", key="analyze_button", disabled=button_disabled):
        if uploaded_file is not None:
            # Use tempfile to handle temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1], dir="temp") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                file_path = temp_file.name
            
            logger.info(f"Starting analysis for file: {file_path}")
        
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                st.error(f"Error: File not found at the provided path: {file_path}")
                return
            
            elif worker is None:
                logger.error("Analysis graph is not compiled")
                st.error("Analysis graph is not compiled. Cannot run analysis.")
                return
            
            else:
                st.write(f"Starting analysis for: {file_path}")
                # Prepare the initial state
                initial_state = {
                    "file_name": file_path,
                    "specs_collection": "specifications", 
                    "error_message": None
                }
                
                try:
                    with st.spinner("Running analysis graph... Please wait."):
                        # Clear previous report and chat history
                        st.session_state.generated_report = None
                        st.session_state.chat_history = []
                        
                        # Run the graph
                        logger.info("Invoking analysis worker")
                        final_state = worker.invoke(initial_state)
                        
                        # Check for errors
                        if final_state.get("error_message"):
                            error_msg = final_state.get("error_message")
                            logger.error(f"Analysis failed: {error_msg}")
                            st.error(f"Analysis failed: {error_msg}")
                            final_report_text = final_state.get("final_report", "Report generation failed due to error.")
                        else:
                            final_report_text = final_state.get("final_report", "Report could not be generated.")
                            # Handle different report formats
                            if hasattr(final_report_text, 'content'):
                                final_report_text = final_report_text.content.strip('markdown').strip()
                            
                            logger.info("Analysis completed successfully")
                            st.subheader("Analysis Complete")
                            # Store report for chat page
                            st.session_state.generated_report = final_report_text
                    
                    # Display the report
                    st.markdown("---")
                    st.subheader("Generated Report")
                    st.markdown(final_report_text)
                    
                    # Create downloadable report
                    docx_buffer = BytesIO()
                    create_docx_from_markdown(final_report_text, docx_buffer)
                    docx_buffer.seek(0)
                    
                    st.sidebar.download_button(
                        label="Download Report as DOCX",
                        data=docx_buffer,
                        file_name="report.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )

                    # Clean up temp file
                    
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            logger.info(f"Removed temporary file: {file_path}")
                            
                        except Exception as cleanup_error:
                            
                            logger.error(f"Error cleaning up Temp folder: {cleanup_error}")
                    
                except Exception as e:
                    logger.error(f"Unexpected error during graph execution: {e}", exc_info=True)
                    st.error(f"An unexpected error occurred during graph execution: {e}")
                    st.exception(e)
        else:
            st.warning("Please upload a file.")
    elif worker is None:
        st.error("Analysis engine could not be initialized. Please check API keys and configurations.")

def chat_with_report_page() -> None:
    """Render the chat interface for interacting with the generated report."""
    st.title("Chat with Generated Report")
    st.markdown("Ask questions about the report generated in the 'Submittal Analysis' section.")
    
    if st.session_state.generated_report is None:
        logger.warning("Attempted to chat without a generated report")
        st.warning("Please run the 'Submittal Analysis' first to generate a report.")
    elif st.session_state.google_chat_model is None:
        logger.error("Google GenAI Chat Model is not available")
        st.error("Google GenAI Chat Model is not available. Cannot start chat.")
    else:
        logger.info("Setting up chat interface")
        setup_chat_interface(st.session_state.generated_report)