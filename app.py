import os
import tempfile
import streamlit as st
from PyPDF2 import PdfReader
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import TextLoader, UnstructuredExcelLoader
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
import io
import pandas as pd


def process_uploaded_outline_file(uploaded_file):
    # Determine the file extension
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    if file_extension == ".txt":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
            filename = os.path.splitext(os.path.basename(uploaded_file.name))[0]
        loader = TextLoader(tmp_file_path)
        docs = loader.load()   

    elif file_extension == ".pdf":
        # Create a temporary text file to store the extracted text
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
            # Read the PDF content
            pdf_reader = PdfReader(uploaded_file)
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text()
            
            # Write the extracted text to the temporary text file
            tmp_file.write(text_content.encode('utf-8'))
            tmp_file_path = tmp_file.name
            filename = os.path.splitext(os.path.basename(uploaded_file.name))[0]
        loader = TextLoader(tmp_file_path)
        docs = loader.load()
    
    elif file_extension == ".xls" or file_extension == ".xlsx":
        # Read the uploaded file into a pandas DataFrame
        xlsx_data = uploaded_file.read()
        xlsx_file = io.BytesIO(xlsx_data)
        df = pd.read_excel(xlsx_file)
        
        # Save the DataFrame to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
            df.to_excel(tmp_file, index=False)
            tmp_file_path = tmp_file.name
            filename = os.path.splitext(os.path.basename(uploaded_file.name))[0]
        loader = UnstructuredExcelLoader(tmp_file_path)
        docs = loader.load()

    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    return docs, filename

# Sidebar for API Key and options
with st.sidebar:
    # st.header("Claude API Key")
    api_key = st.text_input("**Enter Claude API key:**", type="password")

    # Option selector
    option = st.selectbox("**Choose Option**", ["Generate Outline", "Generate Blog"])  # Assuming options are labeled 'Option 1' and 'Option 2'


if option == "Generate Outline":
    st.title("Blog Outline Generator")
    st.write('\n \n \n')
    # # File uploader for the blog prompt and blog outline
    uploaded_prompt_file = st.file_uploader("**Upload outline prompt:**", type="txt")
    uploaded_outline_file = st.file_uploader("**Upload outline notes:**", type=["txt", "pdf", "xlsx", "xls"])

    # Submit button
    if st.button("Generate Outline"):

        if uploaded_prompt_file is not None and uploaded_outline_file is not None:

            # Reading the uploaded files
            prompt_template = uploaded_prompt_file.getvalue().decode("utf-8")
            prompt_template += "\n + Here is the content to reference in order to generate the outline : + \n\n {content}"

            docs, outline_filename = process_uploaded_outline_file(uploaded_outline_file)


            with st.spinner("Processing.. Est. Time ~ 1 minute"):
                blog_prompt = PromptTemplate.from_template(prompt_template)

                # Initialize the LLM and chains with the uploaded files and API key
                llm = ChatAnthropic(temperature=0, anthropic_api_key=api_key, model_name="claude-3-opus-20240229", max_tokens=4096)
                llm_chain = LLMChain(llm=llm, prompt=blog_prompt)

                # Configure the StuffDocumentsChain
                stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="content")

                # Running the chain and getting the output
                result = stuff_chain.run(docs)

            # Displaying the output in a scrollable text box
            st.text_area("**Generated Outline:**", value=result, height=500, disabled=True)

            # Download button for the generated text
            result_bytes = io.BytesIO(result.encode("utf-8"))
            st.download_button(label="Download Generated Outline",
                               data=result_bytes,
                               file_name=f"outline_{outline_filename}.txt",
                               mime="text/plain")
        else:
            st.warning("Please upload both files to generate the blog.")

else:
    st.title("Blog Content Generator")
    st.write('\n \n \n')
    # File uploader for the blog prompt and blog outline
    uploaded_prompt_file = st.file_uploader("**Upload blog prompt:**", type="txt")
    uploaded_outline_file = st.file_uploader("**Upload blog outline:**", type=["txt", "pdf", "xlsx", "xls"])

    # Submit button
    if st.button("Generate Blog"):

        if uploaded_prompt_file is not None and uploaded_outline_file is not None:

            # Reading the uploaded files
            prompt_template = uploaded_prompt_file.getvalue().decode("utf-8")
            prompt_template += """\n + 
                    Here is the outline. Based on the outline and the above instructions, write the content based on above instructions. 
                    Make all the sections as detailed as possible. AIm for around 2500 words in total.: 
                    + \n\n {outline}"""

            docs, outline_filename = process_uploaded_outline_file(uploaded_outline_file)

            with st.spinner("Processing.. Est. Time ~ 1.5 minutes"):
                blog_prompt = PromptTemplate.from_template(prompt_template)

                # Initialize the LLM and chains with the uploaded files and API key
                llm = ChatAnthropic(temperature=0, anthropic_api_key=api_key, model_name="claude-3-opus-20240229", max_tokens=4096)
                llm_chain = LLMChain(llm=llm, prompt=blog_prompt)

                # Configure the StuffDocumentsChain
                stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="outline")

                # Running the chain and getting the output
                result = stuff_chain.run(docs)

            # Displaying the output in a scrollable text box
            st.text_area("**Generated Blog:**", value=result, height=500, disabled=True)
            # st.text_area("**Generated Blog:**", value=st.markdown(f"{result}"), height=500, disabled=True)
            # with st.expander('Display Generated Blog', expanded=True):
            #     st.markdown(result)


            # Download button for the generated text
            result_bytes = io.BytesIO(result.encode("utf-8"))
            st.download_button(label="Download Generated Blog",
                            data=result_bytes,
                            file_name=f"blog_{outline_filename}.txt",
                            mime="text/plain")
        else:
            st.warning("Please upload both files to generate the blog.")


