import os
import shutil
import tempfile
import logging
from io import BytesIO
from typing import Dict, List, Optional, Any, Tuple

# External library imports
import pandas as pd
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
from classes import *
from helpers import datasheet_content, retrieve_from_vectorstore, generate_report, create_docx_from_markdown
import helpers

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
GEMINI_MODEL_NAME = st.secrets.get("GEMINI_MODEL_NAME")
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")

# Constants
VECTORSTORE_PATH = "vectorestores/chromadb"
DEFAULT_BOQ_COLLECTION = "boq_FireFighting"
DEFAULT_SPECS_COLLECTION = "specifications"
LOGO_PATH = "logo.png"

#####################################
# Model & Tool Initialization
#####################################

def initialize_gemini_model():
    """Initialize and return the Google Gemini model."""
    try:
        from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
        model = ChatGoogleGenerativeAI(temperature=0.7, model=GEMINI_MODEL_NAME,google_api_key=GOOGLE_API_KEY)
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
    logger.info(f"Extracting text from datasheet")
    st.info("Extracting text from datasheet...")
    
    file_paths = state["file_paths"]
    submittal_text = datasheet_content(file_paths,logger)
    
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
    # Initialize default values
    boq_collection_name = DEFAULT_BOQ_COLLECTION
    specs_collection_name = DEFAULT_SPECS_COLLECTION
    
    # Skip if previous error occurred
    if state.get("error_message"):
        logger.warning("Skipping BOQ decision due to previous error")
        return {
            "boq_collection": boq_collection_name,
            "specs_collection": specs_collection_name,
            "error_message": state.get("error_message")
        }
    
    # Check if model is available
    if not st.session_state.get("model_with_tools"):
        error_msg = "BOQ model not available"
        logger.error(error_msg)
        return {
            "boq_collection": "Model Error",
            "specs_collection": "Model Error",
            "error_message": error_msg
        }
    
    st.info("Deciding relevant BOQ discipline...")
    submittal_text = state.get("submittal_text", "")
    
    try:
        if not submittal_text:
            logger.warning("Submittal text is empty. Using default BOQ")
            st.warning("Submittal text is empty. Using default BOQ")
        else:
            # Use LLM to determine the appropriate BOQ
            prompt = f"Choose the single most relevant BOQ discipline for the following datasheet text: {submittal_text}"
            logger.info(f"Invoking LLM for BOQ decision with text length: {len(submittal_text)}")
            
            response = st.session_state.model_with_tools.invoke(prompt)
            
            # Safely handle the response
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_call = response.tool_calls[0]
                if isinstance(tool_call, dict) and 'args' in tool_call and 'boq_name' in tool_call['args']:
                    boq_collection_name = tool_call['args']['boq_name']
                    logger.info(f"Selected BOQ: {boq_collection_name}")
                else:
                    logger.warning("Invalid tool call format. Using default BOQ")
            else:
                logger.warning("No tool calls in response. Using default BOQ")
        
    except Exception as e:
        error_msg = f"Error deciding BOQ: {str(e)}"
        logger.error(f"{error_msg}. Using default: {boq_collection_name}")
        st.error(f"{error_msg}. Using default: {boq_collection_name}")
        
        return {
            "boq_collection": boq_collection_name,
            "specs_collection": specs_collection_name,
            "error_message": f"BOQ decision failed: {str(e)}"
        }
    
    # Successfully determined BOQ (or using default)
    return {
        "boq_collection": boq_collection_name,
        "specs_collection": specs_collection_name,
        "error_message": None
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
                str(st.session_state.input_features),
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
    Set up and manage the chat interface for interacting with the report and uploaded images.
    
    Args:
        report_text: The generated report text to chat about
    """
    # Initialize image container in session state if not exists
    if "chat_images" not in st.session_state:
        st.session_state.chat_images = {}
    
    # Image upload section
    with st.expander("Upload an image to discuss", expanded=False):
        uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="chat_image_uploader")
        
        if uploaded_image is not None:
            # Display the uploaded image
            image = PIL.Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Process and store the image
            image_id = f"img_{len(st.session_state.chat_images) + 1}"
            
            # Save image to session state
            st.session_state.chat_images[image_id] = {
                "image": image,
                "filename": uploaded_image.name
            }
            
            st.success(f"Image uploaded successfully! You can now ask questions about this image.")
            
            # Add image reference button
            if st.button("Reference this image in chat"):
                # Create a special message referencing the image
                image_reference = f"[Referencing uploaded image: {uploaded_image.name}]"
                st.session_state.chat_history.append(HumanMessage(content=image_reference))
                st.session_state.chat_images[image_id]["referenced"] = True
    
    # Display existing chat messages with images
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
    
    # Chat input field
    if user_query := st.chat_input("Ask a question about the report or uploaded images..."):
        # Add user message to history and display
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.chat_message("user").write(user_query)
        
        logger.info(f"New chat query: {user_query[:50]}{'...' if len(user_query) > 50 else ''}")

        # Check if there are any images in the context
        has_images = len(st.session_state.chat_images) > 0
        
        # Prepare system message with awareness of images
        if has_images:
            system_message = ("You are a helpful assistant. Answer the user's questions considering both the provided "
                             "report context and any images that have been uploaded. For image-related questions, "
                             "focus on providing helpful analysis of the visible content. When discussing images, "
                             "be specific about what you can see.")
        else:
            system_message = ("You are a helpful assistant. Answer the user's questions considering the provided report context "
                            "(answer also questions that are not related to the report). Do not make up information.")

        # Prepare the prompt template for the chat
        prompt_messages = [
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
        ]
        
        # Add image context if available
        image_context = ""
        if has_images:
            image_context = "\n\nThe user has uploaded the following images:\n"
            for img_id, img_data in st.session_state.chat_images.items():
                image_context += f"- {img_id}: {img_data['filename']}\n"
        
        # Construct the final prompt with report and image context
        human_prompt = ("Given the following report:\n\n---\nREPORT START\n---\n{report_context}\n---\nREPORT END\n---\n"
                       f"{image_context}\n\nAnswer this question: {{question}}")
        
        prompt_messages.append(("human", human_prompt))
        prompt_template = ChatPromptTemplate.from_messages(prompt_messages)

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
                
                # For image analysis queries, we need to process the image
                if has_images and any(img_term in user_query.lower() for img_term in ["image", "picture", "photo", "upload"]):
                    # Extract image descriptions if needed
                    # This would be where you integrate vision capabilities
                    logger.info("Query appears to be about uploaded images")
                    
                    # Process the images if a vision model is available
                    if hasattr(st.session_state, "vision_model") and st.session_state.vision_model:
                        # Process each image that hasn't been analyzed yet
                        for img_id, img_data in st.session_state.chat_images.items():
                            if not img_data.get("description"):
                                try:
                                    # This is where you'd call the vision model
                                    # For example: description = vision_model.analyze(img_data["image"])
                                    # But for now we'll just set a placeholder
                                    img_data["description"] = "Image analysis would appear here."
                                    logger.info(f"Analyzed image {img_id}")
                                except Exception as img_error:
                                    logger.error(f"Error analyzing image {img_id}: {img_error}")
                
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

def analyze_image(image):
    """
    Analyze an image using a vision model.
    
    Args:
        image: PIL Image to analyze
        
    Returns:
        String description of the image content
    """
    try:
        # If you have Gemini Pro Vision or similar model:
        # Convert PIL image to bytes
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        
        # You would need to add vision model initialization in your main code
        # For example:
        # from langchain_google_genai import GoogleGenerativeAIVisionModel
        # vision_model = GoogleGenerativeAIVisionModel(model_name="gemini-pro-vision")
        # return vision_model.generate_content(img_bytes)
        
        # For now, return a placeholder
        return "This is a placeholder for image analysis. Please initialize a vision model."
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return f"Error analyzing image: {e}"

def initialize_session_state():
    """Initialize all necessary session state variables."""
    # Report and chat history
    if "generated_report" not in st.session_state:
        st.session_state.generated_report = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_images" not in st.session_state:
        st.session_state.chat_images = {}
    
    # Models and tools
    if "google_chat_model" not in st.session_state:
        st.session_state.google_chat_model = initialize_gemini_model()
    
    if "gemini_model" not in st.session_state:
        st.session_state.gemini_model = initialize_gemini_model()
    
    # Initialize vision model if needed
    if "vision_model" not in st.session_state:
        try:
            # You would need to import and initialize a vision model here
            # For example:
            # from langchain_google_genai import GoogleGenerativeAIVisionModel
            # st.session_state.vision_model = GoogleGenerativeAIVisionModel(model_name="gemini-pro-vision")
            # logger.info("Successfully initialized vision model")
            st.session_state.vision_model = None  # Replace with actual model initialization
        except Exception as e:
            logger.error(f"Error initializing vision model: {e}")
            st.session_state.vision_model = None
    
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
# UI Components
#####################################

def setup_sidebar() -> None:
    """Set up the sidebar with navigation and information."""
    # Load and display logo
    try:
        image = PIL.Image.open(LOGO_PATH)
        st.sidebar.image(image, use_container_width=True)
    except Exception as e:
        logger.warning(f"Could not load logo: {e}")
    
    # Navigation section
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a section:", ("Submittal Analysis", "Chat with Report","Adding new data"))

    
    # Footer
    st.sidebar.markdown("---")
    
    return app_mode

def submittal_analysis_page(worker: Optional[Any]) -> None:
    """
    Renders the Streamlit interface for submittal analysis using a LangGraph worker.

    Args:
        worker (Optional[Any]): The compiled LangGraph worker used to process the analysis.
    """
    st.badge("Analysis Agent", icon=":material/analytics:")
    st.title("Submittal Review Agent")
    st.markdown("Upload submittal datasheet files and enter comparison features to generate a review report.")

    uploaded_files = st.file_uploader(
        "Choose datasheet files",
        accept_multiple_files=True,
        type=["pdf", "csv", "json", "txt", "docx", "xlsx", "png", "jpg", "jpeg"]
    )
    input_features = st.text_input("Comparison Features")

    button_disabled = (not worker or not uploaded_files)

    if st.button("Analyze Submittal", key="analyze_button", disabled=button_disabled):
        try:
            # Step 1: Extract input features
            #st.session_state.input_features = ""
            if input_features.strip() != "":
                model = initialize_gemini_model().with_structured_output(ComparisonFeatures)
                st.session_state.input_features = model.invoke(str(input_features)).features
                logger.info(f"Parsed input features: {st.session_state.input_features}")
            else:
                st.session_state.input_features = ""
                return

            # Step 2: Save uploaded files to temp directory
            file_paths = []
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1], dir="temp") as temp_file:
                    temp_file.write(file.getbuffer())
                    file_paths.append(temp_file.name)
                    logger.info(f"Saved uploaded file: {temp_file.name}")
            st.session_state.file_paths = file_paths
            # Step 3: Validate and prepare initial state
            if not file_paths:
                st.error("No valid files to analyze.")
                return

            initial_state = {
                "file_paths": file_paths,
                "specs_collection": "specifications",
                "error_message": None
            }

            # Step 4: Run LangGraph analysis
            with st.spinner("Running analysis graph... Please wait."):
                st.session_state.generated_report = None
                st.session_state.chat_history = []

                logger.info("Invoking analysis worker...")
                final_state = worker.invoke(initial_state)

                error_msg = final_state.get("error_message")
                final_report_text = final_state.get("final_report", "No report generated.")

                if error_msg:
                    logger.error(f"Analysis failed: {error_msg}")
                    st.error(f"Analysis failed: {error_msg}")
                else:
                    # Clean markdown if needed
                    if hasattr(final_report_text, 'content'):
                        final_report_text = final_report_text.content.strip("```markdown").strip("```").strip()

                    logger.info("Analysis completed successfully")
                    st.subheader("Analysis Complete")
                    st.session_state.generated_report = final_report_text

                    # Display final report
                    st.markdown("---")
                    st.subheader("Generated Report")
                    st.markdown(final_report_text)

                    # Offer downloadable .docx version
                    docx_buffer = BytesIO()
                    create_docx_from_markdown(final_report_text, docx_buffer)
                    docx_buffer.seek(0)

                    st.sidebar.download_button(
                        label="Download Report as DOCX",
                        data=docx_buffer,
                        file_name="report.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )

        except Exception as e:
            logger.error(f"Unexpected error during analysis: {e}", exc_info=True)
            st.error("An unexpected error occurred during analysis.")
            st.exception(e)

        finally:
            # Step 5: Clean up temporary files
            for file_path in file_paths:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.info(f"Removed temporary file: {file_path}")
                    except Exception as cleanup_error:
                        logger.error(f"Error cleaning up file {file_path}: {cleanup_error}")
    elif not worker:
        st.error("Analysis engine is not initialized. Please check configuration or API keys.")

def data_storing_page(archiver: Optional[Any]) -> None:
    """
    Render the data storing page interface.

    Args:
        archiver: The compiled LangGraph worker responsible for archiving/storing data.
    """
    st.badge("Archive",color="blue",icon=":material/archive:")
    st.title("Data Archiving Service")
    st.markdown("Upload a file to store it in the designated archive or database.")

    # Allow uploading various common data file types, adjust as needed
    uploaded_file = st.file_uploader("Choose a file to archive", type=["pdf", "csv", "json", "txt", "docx","xlsx"])

    # Archive button
    button_disabled = (archiver is None or uploaded_file is None)
    if st.button("Archive File", key="archive_button", disabled=button_disabled):
        if uploaded_file is not None:
            # Use tempfile to handle temporary file
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1], dir="temp") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                file_path = temp_file.name

            logger.info(f"Starting archiving process for file: {file_path}")
            

            if not os.path.exists(file_path):
                logger.error(f"Temporary file not found after creation: {file_path}")
                st.error(f"Error: Could not process the uploaded file.")
                # Clean up attempt even on error
                if os.path.exists(file_path):
                     try:
                         os.remove(file_path)
                     except Exception as cleanup_error:
                         logger.warning(f"Error cleaning up file {file_path} after processing error: {cleanup_error}")
                return

            elif archiver is None:
                logger.error("Archiving worker is not compiled")
                st.error("Archiving service is not available. Cannot store file.")
                # Clean up temp file if archiver is missing post-upload
                if os.path.exists(file_path):
                     try:
                         os.remove(file_path)
                     except Exception as cleanup_error:
                         logger.warning(f"Error cleaning up file {file_path} after archiver check: {cleanup_error}")
                return

            else:
                st.write(f"Starting archival for: {uploaded_file.name}")
                # Prepare the initial state for the archiver
                # Adjust this state based on what the archiver worker expects
                if uploaded_file.name.split(".")[-1] == "xlsx":
                    initial_state = {
                        "file_path": file_path,
                        "file_name": uploaded_file.name,
                        "metadata": { # Optional: Add any relevant metadata
                            "upload_time": st.session_state.get("upload_time", "unknown"), # Example metadata
                            "user": st.session_state.get("user", "anonymous") # Example metadata
                        },
                        "collection_name": "test_archive", # Example: Specify where to store it
                        "error_message": None
                    }
                elif uploaded_file.name.split(".")[-1] == "pdf":
                    initial_state ={
                        "file_path": file_path,
                        "file_name": uploaded_file.name,
                        "metadata": {
                            "upload_time": st.session_state.get("upload_time", "unknown"),
                            "user": st.session_state.get("user", "anonymous")
                        },
                        "collection_name": "test_archive",
                        "error_message": None
                    }

                try:
                    with st.spinner("Storing file... Please wait."):
                        # You might not need to clear session state like in the analysis page
                        # st.session_state.generated_report = None # Unlikely needed
                        # st.session_state.chat_history = [] # Unlikely needed

                        # Run the archiver graph
                        logger.info("Invoking archiving worker")
                        final_state = archiver.invoke(initial_state)

                        # Check for errors reported by the archiver
                        if final_state.get("error_message"):
                            error_msg = final_state.get("error_message")
                            logger.error(f"Archiving failed: {error_msg}")
                            st.error(f"Archiving failed: {error_msg}")
                        # Check for a success message or status
                        elif final_state.get("status_message"):
                            success_msg = final_state.get("status_message")
                            logger.info(f"Archiving completed successfully: {success_msg}")
                            st.success(f"Archiving successful: {success_msg}")
                            # Optionally store confirmation details in session state if needed elsewhere
                            st.session_state.last_archive_status = success_msg
                            st.session_state.last_archived_file = uploaded_file.name
                        else:
                            # Fallback if no specific error or success message is found
                            logger.warning("Archiving process finished, but no specific status message was returned.")
                            st.info("Archiving process finished.")

                    # Display confirmation (already done via st.success/st.error)
                    # No report generation or download button needed typically for storage

                except Exception as e:
                    logger.error(f"Unexpected error during archiving execution: {e}", exc_info=True)
                    st.error(f"An unexpected error occurred during the archiving process: {e}")
                    st.exception(e) # Shows detailed traceback in the app during development

                finally:
                    # Clean up the temporary file in all cases after processing attempt
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            logger.info(f"Removed temporary file: {file_path}")
                        except Exception as cleanup_error:
                            logger.error(f"Error cleaning up temporary file {file_path}: {cleanup_error}")

        else:
            st.warning("Please upload a file to archive.")

    elif archiver is None:
        # Error message shown if the archiver wasn't available on page load
        st.error("Archiving service could not be initialized. Please check configurations.")
        

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


def _excel_data_processor(state:StoreState) -> Dict[str, Any]:
    excel_data_df = helpers.read_boq(state['file_path'], state['sheet_name'])
    processed_data = helpers.process_data(excel_data_df)
    grouped_df = helpers.group_boq_items(processed_data)
    extracted_data = helpers.extract_boq_data(grouped_df)

    return {
        "data": extracted_data
    }

def _pdf_data_processor(state:StoreState) -> Dict[str, Any]:
    extracted_data = helpers.extract_data_from_doc(state['file_path'])
    return {
        "data": extracted_data
    }
    
def _save_excel_data_to_collection(state:StoreState) -> None:

    helpers.save_boq_to_collection(state['data'],"vectorestores/mzdb",state["collection_name"])

def _save_pdf_data_to_collection(state:StoreState) -> None:

    helpers.save_pdf_to_collection(state['data'],"vectorestores/mzdb",state["collection_name"])

def _which_file_type(state:StoreState):
    if state["file_name"].split(".")[-1] == "xlsx":
        return {"next_node": "ExcelDataProcessor"} # Return a dictionary with the next node name
    elif state["file_name"].split(".")[-1] == "pdf":
        return {"next_node": "PdfDataProcessor"} # Return a dictionary with the next node name
    else:
        return END
        

def add_to_store_graph():
    
    graph = StateGraph(StoreState)

    #Nodes
    graph.add_node("DecideFileType", _which_file_type)
    graph.add_node("ExcelDataProcessor", _excel_data_processor)
    graph.add_node("SavePDFToCollection", _save_pdf_data_to_collection)
    graph.add_node("SaveExcelToCollection", _save_excel_data_to_collection)
    graph.add_node("PdfDataProcessor", _pdf_data_processor)
    


    #Edges
    graph.add_edge(START, "DecideFileType")
    graph.add_edge("ExcelDataProcessor", "SaveExcelToCollection")
    graph.add_edge("PdfDataProcessor", "SavePDFToCollection")
    graph.add_edge("SaveExcelToCollection",END)
    graph.add_edge("SavePDFToCollection",END)
    graph.add_conditional_edges("DecideFileType", lambda state: state["next_node"], # Update to access the correct key
                                {
                                    "ExcelDataProcessor": "ExcelDataProcessor",
                                    "PdfDataProcessor": "PdfDataProcessor"
                                })  


    return graph.compile()
