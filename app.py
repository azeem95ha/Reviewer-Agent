# streamlit_app.py (or your main script)

# --- Start of SQLite fix for ChromaDB ---
# Check if running in a Streamlit hosted environment (like Community Cloud)
# or a Linux environment where system sqlite3 might be too old
import platform
if platform.system() == 'Linux':
    try:
        # This trick swaps the system's sqlite3 library with the one packaged
        # by pysqlite3-binary during runtime. Must happen before any module
        # imports sqlite3.
        __import__('pysqlite3')
        import sys
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        print("Successfully replaced sqlite3 with pysqlite3-binary.") # Optional: for logging/debugging
    except ImportError:
        print("pysqlite3-binary not found, using system sqlite3.") # Optional: for logging/debugging
    except KeyError:
         # Handles edge case where 'pysqlite3' might already be 'sqlite3'
        pass
# --- End of SQLite fix --
from reviewer_agent import (
    data_storing_page,
    initialize_session_state,
    logger,
    build_langgraph,
    setup_sidebar,
    submittal_analysis_page,
    chat_with_report_page,
    add_to_store_graph
)


#####################################
# Main Application
#####################################

def main():
    """Main application entry point."""
    logger.info("Starting Submittal Review Application")
    
    # Initialize session state and models
    initialize_session_state()
    
    # Build the graph
    worker = build_langgraph()
    archiver = add_to_store_graph()
    
    # Setup the sidebar and get the selected mode
    app_mode = setup_sidebar()
    
    # Render the selected page
    if app_mode == "Submittal Analysis":
        logger.info("Loading Submittal Analysis page")
        submittal_analysis_page(worker)
    elif app_mode == "Chat with Report":
        logger.info("Loading Chat with Report page")
        chat_with_report_page()
    elif app_mode == "Adding new data":
        logger.info("Loading Adding new data page")
        data_storing_page(archiver)
        

if __name__ == "__main__":
    main()
