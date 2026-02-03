from langchain_core.documents import Document

doc=Document(
    page_content="this contains the important topics covered in the overall universities",
    metadata={
               "source": "wikipedia",
               "pages":"1",
               "Author":"Friedrich Nietzsche",
               "created_date":"2025-01-01"
    }
)
print(doc)

import os
os.makedirs("excel")

# Text loader
from langchain_community.document_loaders import TextLoader
loader=TextLoader("C:/Users/ricky/PycharmProjects/DS/DS_6/data/text/Environment.txt")
doc=loader.load()
print(doc)

# Directory loader
from langchain_community.document_loaders import DirectoryLoader, TextLoader,PyMuPDFLoader

dir_loader=DirectoryLoader("C:/Users/ricky/PycharmProjects/DS/DS_6/data/text",
                           glob="*.txt",
                           loader_cls=TextLoader,
                           show_progress=True,
                            loader_kwargs={"encoding":"utf-8"})
docs=dir_loader.load()
print(docs)


dir_loader=DirectoryLoader("C:/Users/ricky/PycharmProjects/DS/DS_6/data/pdf",
                           glob="*.pdf",
                           loader_cls=PyMuPDFLoader,
                           show_progress=True
                            )
docs=dir_loader.load()
print(docs)