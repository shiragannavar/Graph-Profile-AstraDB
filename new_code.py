import json

import jsonlines
from langchain.schema import Document
from langchain_community.chat_models import ChatOpenAI
from langchain_core.graph_vectorstores.links import METADATA_LINKS_KEY, Link
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.graph_vectorstores.cassandra import CassandraGraphVectorStore
import cassio
import openai
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA


from util.config import LOGGER, OPENAI_API_KEY, ASTRA_DB_ID, ASTRA_TOKEN

# Suppress all of the Langchain beta and other warnings
import warnings
warnings.filterwarnings("ignore", lineno=0)

# Set your OpenAI API key
import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)


# Initialize Cassandra connection (ensure AstraDB is properly configured)
cassio.init(database_id=ASTRA_DB_ID, token=ASTRA_TOKEN)

TABLE_NAME = "profile_graph"
store = CassandraGraphVectorStore(
    embedding=embeddings,
    node_table=TABLE_NAME
)

def extract_products(text: str) -> list:

    # prompt = (
    #     "From the following text, extract a list of products, technologies, or tools "
    #     "that a professional might have used or is an expert in. "
    #     "Provide the list as comma-separated values.\n\n"
    #     f"Text:\n{text}\n\nProducts:"
    # )
    # response = openai.chat.completions.create(
    #     model="gpt-4o",
    #     prompt=prompt
    # )

    messages = [
        {"role": "system", "content": "You are a expert software engineer. You know all the technology products or tool names"},
        {"role": "user",
         "content": f"From the following text, extract a list of products, technologies, or tools that a professional might have used or is an expert in. Provide the list as comma-separated values.\n\nText:\n{text}\n\nProducts:"}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    products_text = response.choices[0].message.content.strip()
    # Split the products by commas and strip whitespace
    products = [product.strip() for product in products_text.split(',') if product.strip()]
    return products

def parse_profile(profile_data: dict) -> list:
    documents = []
    links = set()

    # Create a node for the main profile
    main_profile_id = profile_data.get("profile_id")
    full_name = profile_data.get('full_name', 'Unknown')
    location = profile_data.get('location', 'Unknown')
    # main_profile_content = f"{full_name} - {location}"
    profile_data.pop('similar_names',None)
    profile_data.pop('also_viewed', None)
    cleaned_data_profile = {k: v for k, v in profile_data.items() if v not in (None, [], '', {})}
    main_profile_content = json.dumps(cleaned_data_profile)
    main_profile_doc = Document(
        id=main_profile_id,
        page_content=main_profile_content,
        metadata={
            "full_name": full_name,
            "location": location,
            METADATA_LINKS_KEY: [],  # Will be populated later
        },
    )
    documents.append(main_profile_doc)

    # Aggregate text for product extraction
    aggregate_text = ""

    # Process experiences
    experiences = profile_data.get("experience", [])
    for exp in experiences:
        company_name = exp.get('company_name', 'Unknown Company')
        position = exp.get('position', 'Position not specified')
        exp_id = f"{main_profile_id}_exp_{company_name.replace(' ', '_')}"
        exp_content = f"{position} at {company_name}"
        exp_doc = Document(
            id=exp_id,
            page_content=json.dumps(exp),
            metadata={
                "company_name": company_name,
                "position": position,
                "start_time": exp.get("start_time"),
                "end_time": exp.get("end_time"),
                METADATA_LINKS_KEY: [],
            },
        )
        documents.append(exp_doc)
        # Create a link between the main profile and the experience
        link = Link(
            kind="has_experience",
            tag=f"{main_profile_id}_to_{exp_id}",
            direction="out",
            source=main_profile_id,
            target=exp_id,
        )
        exp_doc.metadata[METADATA_LINKS_KEY].append(link)
        links.add(link)
        # Add experience content to aggregate text
        aggregate_text += f"{exp_content}. "

        # Process working for
        working_for = profile_data.get("working_for", [])
        for work in working_for:
            company_name = work.get('employer', 'Unknown Company')
            position = work.get('type', 'Position not specified')
            work_id = f"{main_profile_id}_exp_{company_name.replace(' ', '_')}"
            working_for_content = f"{position} at {company_name}"
            working_for_doc = Document(
                id=work_id,
                page_content=json.dumps(work),
                metadata={
                    "company_name": company_name,
                    "position": position,
                    METADATA_LINKS_KEY: [],
                },
            )
            documents.append(working_for_doc)
            # Create a link between the main profile and the experience
            link = Link(
                kind="working_for",
                tag=f"{main_profile_id}_to_{work_id}",
                direction="out",
                source=main_profile_id,
                target=work_id,
            )
            working_for_doc.metadata[METADATA_LINKS_KEY].append(link)
            links.add(link)
            # Add experience content to aggregate text
            aggregate_text += f"{working_for_content}. "

    # Optionally, include 'about_me' or other fields if available
    about_me = profile_data.get("about_me")
    if about_me:
        aggregate_text += f"{about_me}. "

    # Use OpenAI to extract products
    if aggregate_text:
        products = extract_products(aggregate_text)
    else:
        products = []

    # Process products
    for product in products:
        product_id = f"product_{product.lower().replace(' ', '_')}"
        product_doc = Document(
            id=product_id,
            page_content=product,
            metadata={
                "product_name": product,
                METADATA_LINKS_KEY: [],
            },
        )
        documents.append(product_doc)
        # Create a link between the main profile and the product
        link = Link(
            kind="expert_in",
            tag=f"{main_profile_id}_to_{product_id}",
            direction="bidir",
            source=main_profile_id,
            target=product_id,
        )
        product_doc.metadata[METADATA_LINKS_KEY].append(link)
        links.add(link)

    # Update the main profile's metadata with links
    main_profile_doc.metadata[METADATA_LINKS_KEY] = list(links)

    return documents



i = 0
with jsonlines.open('Vieu/fiserv-cleaned.jsonl') as f:
    for line in f.iter():
        i += 1
        if (i > 0):
            break
        # print(json.dumps(line))
        # documents = load_documents_from_json(line)
        documents_data = parse_profile(line)
        store.add_documents(documents_data)


# # Example: Query the graph for nodes related to the main profile
# from langchain.chains import GraphQAChain
# from langchain_openai import OpenAI
#
# llm = OpenAI(temperature=0)
# chain = GraphQAChain.from_llm(llm, graph=store, verbose=True)
# query = f"What products or technologies people are experts in?"
# answer = chain.run(query)
# print(answer)


def query_documents(query):

    # Initialize OpenAI LLM
    # llm = OpenAI(api_key=OPENAI_API_KEY, model="gpt-4-turbo")
    llm = ChatOpenAI(model="gpt-4-1106-preview")

    # Create a retriever from the vector store
    # Using Max Marginal Relevance traversal to leverage the graph structure
    retriever = store.as_retriever(
        search_type="mmr_traversal",
        search_kwargs={
            "k": 10,  # Number of documents to retrieve
            "lambda_mult": 0.5  # Balance between diversity and relevance
        }
    )
    from langchain.prompts import PromptTemplate

    # Define your prompt template
    prompt_template = """
    As an expert with handling profiles, provide detailed and accurate information based on the context provided.

    Question: {question}
    Context:
    {context}

    Answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["question", "context"]
    )

    # Create a RetrievalQA chain using the retriever and LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 'stuff' chain type concatenates documents for the LLM
        retriever=retriever,
        return_source_documents=True,  # Optionally return source documents
    chain_type_kwargs = {"prompt": PROMPT},
    )

    # Run the chain to get the answer
    result = qa_chain({"query": query})
    answer = result["result"]
    # Access the source documents if needed
    source_docs = result["source_documents"]

    # Execute a graph search on your stored documents
    # result_documents = list(
    #     store.traversal_search("Where does Erin McKone stay")
    # )
    # print(result_documents)

    return answer, source_docs

for i in range(25):
    query = input("Ask your question: ")
    answer, source_docs = query_documents(query)
    print("Answer:", answer)
print("\n\n\n\nSource Documents:")
for doc in source_docs:
    print(doc.page_content)
