import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
import importlib
import os

st.set_page_config(
                    page_title="Template Project",
                    page_icon=Image.open("icon_site.png"),
                    layout="wide",
                    )

def get_pages():
    pages = []
    icons = []
    modules = []
    
    BLACKLIST_FILES = ['__init__', 'test']  # aggiungi qui i file da escludere    
    # page_order = []
    page_order = ['home', 'history', 'datavisualisation', 'map', 'contacts']

    files = [f[:-3] for f in os.listdir('pag') if f.endswith('.py') and f[:-3] not in BLACKLIST_FILES]
    files.sort(key=lambda x: page_order.index(x) if x in page_order else len(page_order))
    
    # Mapping icon 
    icon_mapping = {
                    'home': 'bi-house',
                    'history': 'bi-hourglass-split',
                    'datavisualisation': 'bi-card-image',
                    'map': 'bi-map',
                    'contacts': 'bi-envelope'
                    }
    
    for file in files:
        page_name = file.capitalize()
        pages.append(page_name)        
        icons.append(icon_mapping.get(file, 'bi-file'))        
        module = importlib.import_module(f'pag.{file}')
        modules.append(module)
    
    return pages, icons, modules

pages, icons, modules = get_pages()

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, function):
        self.apps.append({
                        "title": title,
                        "function": function
                        })

    def main():
        with st.sidebar:
            app = option_menu(
                                menu_title="Menu",
                                options=pages,
                                icons=icons,
                                menu_icon="bi-list",
                                default_index=0,
                                styles={
                                        "container": {"padding": "5!important", "background-color": "black"},
                                        "icon": {"color": "white", "font-size": "23px"},
                                        "nav-link": {"color": "white", "font-size": "20px", "text-align": "left", "margin": "0px"},
                                        "nav-link-selected": {"color": "black", "background-color": "#9ac280"}
                                        }
                                        )        
        selected_index = pages.index(app)
        modules[selected_index].main()

if __name__ == "__main__":
    MultiApp.main()