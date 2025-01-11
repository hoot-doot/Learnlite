import streamlit as st

from streamlit_option_menu import option_menu


import ocr, account, history
# st.set_page_config(
#         page_title="Learnlite",
# )


class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):

        self.apps.append({
            "title": title,
            "function": func
        })

    def run():
        # app = st.sidebar(
        with st.sidebar:        
            app = option_menu(
                menu_title='LearnLite ',
                options=['Account','OCR & Text Summarization','History'],
                icons=['person-circle','chat-text-fill','chat-fill'],
                menu_icon='chat-text-fill',
                default_index=0,
                styles={
                    }
                
                )

        if app == "Account":
            account.app()    
        
        if app == "OCR & Text Summarization":
            ocr.app()

        if  app=="History":
            history.app()  
          
             
    run()            
         
