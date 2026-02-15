from langchain_community.document_loaders import AsyncChromiumLoader

def get_loader():
    loader = AsyncChromiumLoader(["https://www.gcu.ac.uk/study/courses/undergraduate-software-development-glasgow"])
    return loader
