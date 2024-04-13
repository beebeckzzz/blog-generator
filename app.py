import os
import tempfile
import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import TextLoader
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
import io



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
    uploaded_outline_file = st.file_uploader("**Upload outline notes:**", type="txt")

    # Submit button
    if st.button("Generate Blog"):

        if uploaded_prompt_file is not None and uploaded_outline_file is not None:

            # Reading the uploaded files
            prompt_template = uploaded_prompt_file.getvalue().decode("utf-8")
            prompt_template += "\n + Here is the content to reference in order to generate the blog outline : + \n\n {content}"

            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_outline:
                tmp_outline.write(uploaded_outline_file.getvalue())
                tmp_outline_path = tmp_outline.name
                outline_filename = os.path.splitext(os.path.basename(uploaded_outline_file.name))[0]

            with st.spinner("Processing.. Est. Time ~ 1 minute"):
                blog_prompt = PromptTemplate.from_template(prompt_template)

                # Initialize the LLM and chains with the uploaded files and API key
                llm = ChatAnthropic(temperature=0, anthropic_api_key=api_key, model_name="claude-3-opus-20240229", max_tokens=4096)
                llm_chain = LLMChain(llm=llm, prompt=blog_prompt)

                # Configure the StuffDocumentsChain
                stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="content")

                loader = TextLoader(tmp_outline_path)
                docs = loader.load()
                # Running the chain and getting the output
                result = stuff_chain.run(docs)

            # Displaying the output in a scrollable text box
            st.text_area("**Generated Outline:**", value=result, height=500, disabled=True)

            # Download button for the generated text
            result_bytes = io.BytesIO(result.encode("utf-8"))
            st.download_button(label="Download Generated Blog",
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
    uploaded_outline_file = st.file_uploader("**Upload blog outline:**", type="txt")

    # Submit button
    if st.button("Generate Blog"):

        if uploaded_prompt_file is not None and uploaded_outline_file is not None:

            # Reading the uploaded files
            prompt_template = uploaded_prompt_file.getvalue().decode("utf-8")
            prompt_template += "\n + Be as detailed as possible for every heading. Here is the outline for the blog: + \n\n {outline}"

            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_outline:
                tmp_outline.write(uploaded_outline_file.getvalue())
                tmp_outline_path = tmp_outline.name
                outline_filename = os.path.splitext(os.path.basename(uploaded_outline_file.name))[0]

            with st.spinner("Processing.. Est. Time ~ 1.5 minutes"):
                blog_prompt = PromptTemplate.from_template(prompt_template)

                # Initialize the LLM and chains with the uploaded files and API key
                llm = ChatAnthropic(temperature=0, anthropic_api_key=api_key, model_name="claude-3-opus-20240229", max_tokens=4096)
                llm_chain = LLMChain(llm=llm, prompt=blog_prompt)

                # Configure the StuffDocumentsChain
                stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="outline")

                loader = TextLoader(tmp_outline_path)
                docs = loader.load()
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


