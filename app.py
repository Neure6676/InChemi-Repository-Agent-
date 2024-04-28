import base64
import streamlit as st
from dotenv import load_dotenv
import requests

from core.parsing import read_file
from core.chunking import chunk_file
from core.embedding import embed_files
from core.qa import query_folder
from core.utils import get_llm

load_dotenv()

st.title('InChemi Repository Agent')
st.subheader(' - for chemistry knowledge retrieval ⚗️')
st.header(' ')

# def displayPDF(file):
#     # Opening file from file path
#     with open(file, "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')

#     # Embedding PDF in HTML
#     pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

#     # Displaying File
#     st.markdown(pdf_display, unsafe_allow_html=True)


def displayPDF(file):
    if file.startswith("http"):
        response = requests.get(file)
        pdf_bytes = response.content
    else:
        # Opening local file from file path
        with open(file, "rb") as f:
            pdf_bytes = f.read()

    # Convert PDF bytes to base64
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


uploaded_file = st.file_uploader(
    "Upload a pdf, docx, or txt file",
    type=["pdf", "docx", "txt"],
)

if not uploaded_file:
    st.stop()

file = read_file(uploaded_file)

chunked_file = chunk_file(file, chunk_size=300, chunk_overlap=0)


with st.sidebar:

    st.header("Doc Preview")

    # filepath = "docs/"+uploaded_file.name
    # with open(filepath, "wb") as temp_file:
    #         temp_file.write(uploaded_file.read())

    displayPDF("https://jcheminf.biomedcentral.com/counter/pdf/10.1186/1758-2946-5-7.pdf")

    st.divider()

    st.caption("<p style ='text-align:center'> made with by Team 3AM</p>",unsafe_allow_html=True )



with st.spinner("Indexing document... This may take a while⏳"):
    folder_index = embed_files(
        files=[chunked_file],
        embedding="openai",
        vector_store="faiss"
    )

with st.form(key="qa_form"):
    query = st.text_area("Ask a question about the document")
    submit = st.form_submit_button("Submit")

if submit:
    # Output Columns
    answer_col, sources_col = st.columns(2)

    llm = get_llm(model="gpt-4", temperature=0)
    result = query_folder(
        folder_index=folder_index,
        query=query,
        llm=llm,
        return_all=False
    )

    with answer_col:
        st.markdown("#### Answer")
        st.markdown(result.answer)

    with sources_col:
        st.markdown("#### Sources")
        for source in result.sources:
            st.markdown(source.page_content)
            st.markdown(source.metadata["source"])
            st.markdown("---")