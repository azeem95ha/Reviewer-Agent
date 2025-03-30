from reviewer_agent import (
    initialize_session_state,
    logger,
    build_langgraph,
    setup_sidebar,
    submittal_analysis_page,
    chat_with_report_page
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
    
    # Setup the sidebar and get the selected mode
    app_mode = setup_sidebar()
    
    # Render the selected page
    if app_mode == "Submittal Analysis":
        logger.info("Loading Submittal Analysis page")
        submittal_analysis_page(worker)
    elif app_mode == "Chat with Report":
        logger.info("Loading Chat with Report page")
        chat_with_report_page()

if __name__ == "__main__":
    main()
