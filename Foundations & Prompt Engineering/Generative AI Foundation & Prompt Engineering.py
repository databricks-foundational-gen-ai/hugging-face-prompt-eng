# Databricks notebook source
# MAGIC %md
# MAGIC # Generative AI Foundation
# MAGIC
# MAGIC ## GenAI & Maturity curve
# MAGIC
# MAGIC
# MAGIC
# MAGIC Deploying GenAI can be done in multiple ways:
# MAGIC
# MAGIC - **Prompt engineering on public APIs (e.g. LLama 2, openAI)**: answer from public information, retail (think ChatGPT)
# MAGIC - Retrieval Augmented Generation (RAG): specialize your model with additional content. *This is what we'll focus on in this demo*
# MAGIC - OSS model Fine tuning: when you have a large corpus of custom data and need specific model behavior (execute a task)
# MAGIC - Train your own LLM: for full control on the underlying data sources of the model (biomedical, Code, Finance...)
# MAGIC <img src="https://github.com/Hehehe421/Databricks-GenAI-Series/blob/main/Foundations%20%26%20Prompt%20Engineering/images/prompt.png?raw=true" width="600px" style="float:right"/>
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=%2F01-quickstart%2F00-RAG-chatbot-Introduction&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2F01-quickstart%2F00-RAG-chatbot-Introduction&version=1">

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Prompt Engineering Use Cases --  Applications with LLMs
# MAGIC
# MAGIC In this notebook, we'll take a whirlwind tour of some top common applications using Large Language Models (LLMs):
# MAGIC * Summarization
# MAGIC * Sentiment analysis
# MAGIC * Translation
# MAGIC * Zero-shot classification
# MAGIC * Few-shot learning
# MAGIC
# MAGIC We will see how existing, open-source (and proprietary) models can be used out-of-the-box for many applications.  For this, we will use [Hugging Face models](https://huggingface.co/models) and some simple prompt engineering.
# MAGIC
# MAGIC We will then look at Hugging Face APIs in more detail to understand how to configure LLM pipelines.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC 1. Use a variety of existing models for a variety of common applications.
# MAGIC 1. Understand basic prompt engineering.
# MAGIC 1. Understand zero-shot learning Vs few-shot learning

# COMMAND ----------

# MAGIC %run ./init/config $catalog_name=$catalog_name

# COMMAND ----------

from datasets import load_dataset
from transformers import pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hugging Face, Transformers Models, and Tokenizers
# MAGIC Let's navigate to [Hugging Face model hub](https://huggingface.co/models) and explore some models that are on the platform. You will see that (most) models come with descriptions of their task, the data that they were trained on, as well as their associated `Tokenizer`
# MAGIC
# MAGIC ### Transformers Models
# MAGIC In the Hugging Face library, a Transformers model refers to a pre-trained model that can be used for a wide range of NLP tasks. These models, like BERT, GPT, or T5, are built using the Transformers architecture and are trained on large datasets, enabling them to understand and generate human language effectively.
# MAGIC
# MAGIC ### Tokenizers
# MAGIC A tokenizer is a critical component that preprocesses text data for the model. Each model in the library usually comes with its associated tokenizer, ensuring compatibility and optimal performance.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Prompt Engineering User Cases -- Applications with LLMs
# MAGIC
# MAGIC The goal of this section is to get your feet wet with several LLM applications and to show how easy it can be to get started with LLMs.
# MAGIC
# MAGIC As you go through the examples, note the datasets, models, APIs, and options used.  These simple examples can be starting points when you need to build your own application.

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Summarization
# MAGIC
# MAGIC Input a block of text into the model, and it generates a concise summary, capturing the main points of the original text.
# MAGIC
# MAGIC Here, we will use a model which does *abstractive* summarization.
# MAGIC
# MAGIC **Background reading**: The [Hugging Face summarization task page](https://huggingface.co/docs/transformers/tasks/summarization) lists model architectures which support summarization. The [summarization course chapter](https://huggingface.co/course/chapter7/5) provides a detailed walkthrough.
# MAGIC

# COMMAND ----------

# MAGIC %md We next use the Hugging Face `pipeline` tool to load a pre-trained model.  In this LLM pipeline constructor, we specify:
# MAGIC * `task`: This first argument specifies the primary task.  See [Hugging Face tasks](https://huggingface.co/tasks) for more information.
# MAGIC * `model`: This is the name of the pre-trained model from the [Hugging Face Hub](https://huggingface.co/models).
# MAGIC * `min_length`, `max_length`: We want our generated summaries to be between these two token lengths.
# MAGIC * `truncation`: Some input articles may be too long for the LLM to process.  Most LLMs have fixed limits on the length of input sequences.  This option tells the pipeline to truncate the input if needed.

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers.utils import logging

# Load model from Hugging Face using the transformers library
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")

# create a transformers pipeline
# to create one, we need to define the following: The type of task, the model, the tokenizer, as well as a few other parameters that we'll discuss below
pipe = pipeline(
  task = "summarization", 
  model=model, 
  tokenizer=tokenizer, 
  max_new_tokens=1024, 
  device_map='auto', 
  truncation=True
)

logging.set_verbosity(40)

# COMMAND ----------

text_to_summarize= """
                    Barrington DeVaughn Hendricks (born October 22, 1989), known professionally as JPEGMafia (stylized in all caps), is an American rapper, singer, and record producer born in New York City and based in Baltimore, Maryland. His 2018 album Veteran, released through Deathbomb Arc, received widespread critical acclaim and was featured on many year-end lists. It was followed by 2019's All My Heroes Are Cornballs and 2021's LP!, released to further critical acclaim. 
                    """

pipe(text_to_summarize)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sentiment analysis
# MAGIC
# MAGIC Sentiment analysis is a text classification task of estimating whether a piece of text is positive, negative, or another "sentiment" label.  The precise set of sentiment labels can vary across applications.
# MAGIC
# MAGIC **Background reading**: See the Hugging Face [task page on text classification](https://huggingface.co/tasks/text-classification) or [Wikipedia on sentiment analysis](https://en.wikipedia.org/wiki/Sentiment_analysis).
# MAGIC
# MAGIC

# COMMAND ----------

sentiment_classifier = pipeline(
    task="text-classification",
    model="nickwong64/bert-base-uncased-poems-sentiment"
)

# COMMAND ----------

sentiment_classifier("I hate it when my phone battery dies.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Translation
# MAGIC
# MAGIC Translation models may be designed for specific pairs of languages, or they may support more than two languages.  We will see both below.
# MAGIC
# MAGIC **Background reading**: See the Hugging Face [task page on translation](https://huggingface.co/tasks/translation) or the [Wikipedia page on machine translation](https://en.wikipedia.org/wiki/Machine_translation).
# MAGIC
# MAGIC * **Models**:
# MAGIC    * [Helsinki-NLP/opus-mt-en-es](https://huggingface.co/Helsinki-NLP/opus-mt-en-es) is used for the first example of English ("en") to Spanish ("es") translation.  This model is based on [Marian NMT](https://marian-nmt.github.io/), a neural machine translation framework developed by Microsoft and other researchers.  See the [GitHub page](https://github.com/Helsinki-NLP/Opus-MT) for code and links to related resources.
# MAGIC    * [t5-small](https://huggingface.co/t5-small) model, which has 60 million parameters (242MB for PyTorch).  T5 is an encoder-decoder model created by Google which supports several tasks such as summarization, translation, Q&A, and text classification.  For more details, see the [Google blog post](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html), [code on GitHub](https://github.com/google-research/text-to-text-transfer-transformer), or the [research paper](https://arxiv.org/pdf/1910.10683.pdf).  For our purposes, it supports translation for English, French, Romanian, and German.

# COMMAND ----------

en_to_es_translation_pipeline = pipeline(
    task="translation",
    model="Helsinki-NLP/opus-mt-en-es"
)

# COMMAND ----------

en_to_es_translation_pipeline(
    "Existing, open-source (and proprietary) models can be used out-of-the-box for many applications."
)

# COMMAND ----------

# MAGIC %md
# MAGIC Other models are designed to handle multiple languages. Below, we show this with t5-small. Note that, since it supports multiple languages (and tasks), we give it an explicit instruction to translate from one language to another.

# COMMAND ----------

t5_small_pipeline = pipeline(
    task="text2text-generation",
    model="t5-small",
    max_length=50
)

# COMMAND ----------

t5_small_pipeline(
    "translate English to French: Existing, open-source (and proprietary) models can be used out-of-the-box for many applications."
)

# COMMAND ----------

# MAGIC %md
# MAGIC Note that between these two models that the prompt had to be engineered differently based on the model.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Prompt Engineering
# MAGIC
# MAGIC Prompt engineering is a critical aspect of interacting effectively with Large Language Models (LLMs). It involves crafting prompts that can be appended to user inputs that guide the model to generate the most relevant and accurate outputs. This skill is valuable for several reasons:
# MAGIC
# MAGIC 1. **Precision and Relevance**: Well-engineered prompts help the model understand the context and specificity of the query, leading to more precise and relevant responses.
# MAGIC 2. **Efficiency**: Effective prompts can reduce the number tokens generated by the model, while still maintaining accuracy/correctness. This saves time and computational resources.
# MAGIC 3. **Creative and Complex Tasks**: For tasks that require creativity or complex problem-solving, carefully designed prompts can significantly improve the quality of the model's output.

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC <img src="https://daxg39y63pxwu.cloudfront.net/images/blog/langchain/LangChain.webp" alt="LangChain" width="700"/>
# MAGIC
# MAGIC
# MAGIC ## LangChain 
# MAGIC LangChain is a framework designed to simplify the creation of applications using large It enables applications that:
# MAGIC     - Are context-aware: connect a language model to sources of context (prompt instructions, few shot examples, content to ground its response in, etc.)
# MAGIC     - Reason: rely on a language model to reason (about how to answer based on provided context, what actions to take, etc.) language models.

# COMMAND ----------

# MAGIC %md
# MAGIC <table border=0 cellpadding=0 cellspacing=0>
# MAGIC <tr>
# MAGIC   <td>
# MAGIC     <h1>DBRX Announcment</h1><br/>
# MAGIC     Get started with DBRX!
# MAGIC     <ul>
# MAGIC       <li>DBRX is a transformer-based decoder-only large language model (LLM) that was trained using next-token prediction. It uses a fine-grained mixture-of-experts (MoE) architecture with 132B total parameters of which 36B parameters are active on any input. It was pre-trained on 12T tokens of text and code data. </u></li>
# MAGIC       <li>Foundation Model APIs (<a href="https://docs.databricks.com/en/security/secrets/example-secret-workflow.html">AWS</a>)
# MAGIC       <li>AI Playground chat interface</li>
# MAGIC     </ul>
# MAGIC   </td>
# MAGIC <td>
# MAGIC   <img src="https://www.databricks.com/sites/default/files/2024-03/introducing-dbrx.gif" width="1000"/>
# MAGIC </td>
# MAGIC </tr>
# MAGIC </table>

# COMMAND ----------

# MAGIC %pip install langchain==0.1.5 mlflow[databricks] sqlalchemy
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chat_models import ChatDatabricks

#call llama2 70B, hosted by Databricks
llama_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 400)

#call DBRX model
dbrx_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 400)

# COMMAND ----------

user_question = "How can I speed up my Spark join operation?"
llama_model.predict(user_question)

# COMMAND ----------

user_question = "How can I speed up my Spark join operation?"
dbrx_model.predict(user_question)

# COMMAND ----------

# MAGIC %pip install --upgrade sqlalchemy

# COMMAND ----------

from langchain import PromptTemplate
from langchain.chains.llm import LLMChain

#now, let's create a prompt template to make our incoming queries databricks-specific
intro_template = """
You are a Databricks support engineer tasked with answering questions about Spark. Include Databricks-relevant information in your response and be as prescriptive as possible. Cite Databricks documentation for your answers. If you don't know the answer, just say you don't know, please do not make up the answer.
User Question:" {question}"
"""

# COMMAND ----------

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=intro_template,
)

dbrx_chain = LLMChain(
    llm=dbrx_model,
    prompt=prompt_template,
    output_key="Support Response",
    verbose=False
)

dbrx_chain_response = dbrx_chain.run({"question":user_question})
print(dbrx_chain_response)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Zero-shot classification
# MAGIC
# MAGIC Zero-shot classification (or zero-shot learning) is the task of classifying a piece of text into one of a few given categories or labels, without having explicitly trained the model to predict those categories beforehand.  The idea appeared in literature before modern LLMs, but recent advances in LLMs have made zero-shot learning much more flexible and powerful.
# MAGIC
# MAGIC **Background reading**: See the Hugging Face [task page on zero-shot classification](https://huggingface.co/tasks/zero-shot-classification) or [Wikipedia on zero-shot learning](https://en.wikipedia.org/wiki/Zero-shot_learning).
# MAGIC
# MAGIC
# MAGIC **Scenario**: An e-commerce platform seeks to automate the analysis of diverse customer feedback. The goal is to categorize feedback into distinct themes such as `Product Quality`, `Customer Service`, `Pricing`, `Technical Support`, and `Delivery` to better understand customer sentiments and identify areas for improvement. This process is currently manual, time-consuming, and prone to inconsistency.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Example with DBRX
# Test Databricks Foundation LLM model
from langchain_community.chat_models import ChatDatabricks
#chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200)
dbrx_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 200)

# COMMAND ----------

#now, let's create a prompt template to make our incoming queries databricks-specific
intro_template = """
You are a classifier that was tasked to catogirze the customer feedback into one of these categories:["Product Quality", "Customer Service", "Pricing", "Technical Support", "Delivery","Unkown"]. Make sure only output the category that is most likely to be correct. No need to add any explaination.
feedback:" {question}"
"""

# COMMAND ----------

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=intro_template,
)

dbrx_chain = LLMChain(
    llm=dbrx_model,
    prompt=prompt_template #,
    #output_key="Support Response",
    #verbose=False
)

# COMMAND ----------

# Example usage
feedback_examples = [
    "I'm really disappointed with the late delivery of my order. It was supposed to arrive last week!",
    "Your support team did a fantastic job helping me resolve an issue with my account.",
    "I found the pricing to be quite competitive compared to other brands."
]

for feedback in feedback_examples:
    print(dbrx_chain.predict(question = feedback))
    print("-----")

# COMMAND ----------

# DBTITLE 1,Add your openai API
import openai
# Replace with your actual OpenAI API key
openai.api_key = 'xxxxxxx'

# COMMAND ----------

# DBTITLE 1,Example with OpenAI
# This example uses OpenAI's zero-shot learning model
import openai

def categorize_feedback(feedback: str):
    """
    Categorizes customer feedback into predefined themes using zero-shot learning.
    Parameters:
    - feedback (str): The customer feedback text to be categorized.
    """
    categories = ["Product Quality", 
                  "Customer Service", 
                  "Pricing", 
                  "Technical Support", 
                  "Delivery"]
    prompt = f"Feedback: \"{feedback}\"\n\nCategories: {', '.join(categories)}\n\nThis feedback is most likely about:"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": " "}
        ],
        max_tokens=60,
        stop=["\n"]
    )

    # Extract and print the predicted category from the response
    predicted_category = response.choices[0].message.content.strip()
    print(f"Predicted Category: {predicted_category}")

# Example usage
feedback_examples = [
    "I'm really disappointed with the late delivery of my order. It was supposed to arrive last week!",
    "Your support team did a fantastic job helping me resolve an issue with my account.",
    "I found the pricing to be quite competitive compared to other brands."
]

for feedback in feedback_examples:
    categorize_feedback(feedback)
    print("-----")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Few-shot learning
# MAGIC
# MAGIC In few-shot learning tasks, you give the model an instruction, a few query-response examples of how to follow that instruction, and then a new query.  The model must generate the response for that new query.  This technique has pros and cons: it is very powerful and allows models to be reused for many more applications, but it can be finicky and require significant prompt engineering to get good and reliable results.
# MAGIC
# MAGIC **Background reading**: See the [Wikipedia page on few-shot learning](https://en.wikipedia.org/wiki/Few-shot_learning_&#40;natural_language_processing&#41;) or [this Hugging Face blog about few-shot learning](https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api).
# MAGIC
# MAGIC
# MAGIC **Scenario**: An medical insurance company seeks to automate extract the corresponding diagnosis code (such as ICD9/10) from diverse medical descriptions in order to build robust structured database for down streaming reporting purpose. This process is currently manually m time-consuming, and require large amount of experts in specific areas. 

# COMMAND ----------

# DBTITLE 1,Example with DBRX
#Create the few shot learning examples
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
examples = [
  {
    "input":"A patient presents with a fever, cough, and shortness of breath. The patient has been diagnosed with pneumonia.",
    "output":"""J18.9""",
  },
  {
    "input":"The individual complains of severe, throbbing headache, nausea, and is extremely sensitive to light. A diagnosis of migraine without aura is made.",
    "output":"""G43.909""",
  },
  {
    "input":"Examination reveals elevated blood pressure readings on three separate occasions, leading to a diagnosis of primary hypertension.",
    "output":"""I10""",
  },
]

# This is a prompt template used to format each individual example.
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Extracts diagnosis codes from a medical descriptions just like the following examples. Please only output a single and cleaned diagnosis code that most likely as the result."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

# COMMAND ----------

print(final_prompt)

# COMMAND ----------

from langchain.chains.llm import LLMChain
# Initialize the ChatDatabricks model with endpoint and token settings
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens=200)

# Initialize the chat model with the prompt template
chain = LLMChain(
    llm=chat_model,
    prompt=final_prompt
)

# COMMAND ----------

#example 1
chain.predict(input="A child is brought in with a red, itchy rash and small blisters on the skin, diagnosed as chickenpox")

# COMMAND ----------

#example 2
chain.predict(input= "The patient reports chronic pain in the lower back area, worsening with movement.")

# COMMAND ----------

# DBTITLE 1,Example with OpenAI API
def extract_diagnosis_code(review: str):
    """
    Extracts diagnosis codes from a medical descriptions using few-shot learning via the OpenAI API.
    Parameters:
    - review (str): The medical descriptions text from which to extract diagnosis codes.
    """
    # Clearly formatted few-shot learning examples
    examples = """
    Examples:
    Text: "A patient presents with a fever, cough, and shortness of breath. The patient has been diagnosed with pneumonia."
    Feature: J18.9

    Text: "The individual complains of severe, throbbing headache, nausea, and is extremely sensitive to light. A diagnosis of migraine without aura is made."
    Feature: G43.909

    Text: "Examination reveals elevated blood pressure readings on three separate occasions, leading to a diagnosis of primary hypertension."
    Feature: I10
    """

    prompt = f"{examples}\nText: \"{review}\"\nFeature:"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": " "}
        ],
        max_tokens=60,
        stop=["\n"]
    )

    # Extract and print the predicted product features from the response
    predicted_features = response['choices'][0]['message']['content'].strip()
    print(f"Extracted Features: {predicted_features}")

# Example usage
reviews = [
    "A child is brought in with a red, itchy rash and small blisters on the skin, diagnosed as chickenpox.",
    "The patient reports chronic pain in the lower back area, worsening with movement. Diagnosis: lumbar spine osteoarthritis.",
]

for review in reviews:
    extract_diagnosis_code(review)
    print("-----")

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://miro.medium.com/v2/resize:fit:1400/1*OVqzvRSNWloHMYCF1EZtqg.png" alt="mlflow" width="700"/>
# MAGIC
# MAGIC # Logging and Registering with MLflow
# MAGIC Now that we have our model, we want to log the model and its artifacts, so we can version it, deploy it, and also share it with other users.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a Model Signature
# MAGIC For LLMs, we need to generate a [model signature](https://mlflow.org/docs/latest/models.html#model-signature-and-input-example).
# MAGIC Model signatures show the expected input and output types for a model. Which makes quality assurance for downstream serving easier

# COMMAND ----------

from mlflow.models import infer_signature

input_str="A child is brought in with a red, itchy rash and small blisters on the skin, diagnosed as chickenpox"
prediction = chain.run(input_str)
input_columns = [
    {"type": "string", "name": input_key} for input_key in chain.input_keys
]
signature = infer_signature(input_columns, prediction)


# COMMAND ----------

# MAGIC %pip install --upgrade mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import cloudpickle
import langchain 

# Create a new mlflow experiment or get the existing one if already exists.
current_user = spark.sql("SELECT current_user() as username").collect()[0].username
experiment_name = f"/Users/{current_user}/genai-intro-workshop"
mlflow.set_experiment(experiment_name)

# set the name of our model
model_name = "dbrx_genai_chain"

# get experiment id to pass to the run
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
with mlflow.start_run(experiment_id=experiment_id):
    mlflow.langchain.log_model(
        chain,
        model_name,
        signature=signature,
        input_example=input_str,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
            "pydantic==2.5.2 --no-binary pydantic",
            "cloudpickle==" + cloudpickle.__version__,
        ]
    )
    # Log the dictionary as a table
    # mlflow.log_table(data=table_dict, artifact_file="qabot_eval_results.json")

# COMMAND ----------

import mlflow

#grab our most recent run (which logged the model) using our experiment ID
runs = mlflow.search_runs([experiment_id])
last_run_id = runs.sort_values('start_time', ascending=False).iloc[0].run_id

#grab the model URI that's generated from the run
model_uri = f"runs:/{last_run_id}/{model_name}"

#log the model to catalog.schema.model. The schema name referenced below is generated for you in the init script
catalog = dbutils.widgets.get("catalog_name")
schema = 'genai_workshop_sixuan'

#set our registry location to Unity Catalog
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model(
    model_uri=model_uri,
    name=f"{catalog}.{schema}.{model_name}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC Registered model [here](https://e2-demo-field-eng.cloud.databricks.com/explore/data/models/main/genai_workshop_sixuan/dbrx_genai_chain?o=1444828305810485)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # What's the next?
# MAGIC
# MAGIC ## Generative AI workshop on Retrieval Augmented Generation (RAG)
# MAGIC
# MAGIC [Register here](https://events.databricks.com/EV-AEB-Hands-on-Workshop-GenAIExploringRAG)
# MAGIC
# MAGIC Deploying GenAI can be done in multiple ways:
# MAGIC
# MAGIC - Prompt engineering on public APIs (e.g. LLama 2, openAI): answer from public information, retail (think ChatGPT)
# MAGIC - **Retrieval Augmented Generation (RAG)**: specialize your model with additional content. *This is what we'll focus on in this demo*
# MAGIC - OSS model Fine tuning: when you have a large corpus of custom data and need specific model behavior (execute a task)
# MAGIC - Train your own LLM: for full control on the underlying data sources of the model (biomedical, Code, Finance...)
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-maturity.png?raw=true" width="600px" style="float:right"/>
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=%2F01-quickstart%2F00-RAG-chatbot-Introduction&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2F01-quickstart%2F00-RAG-chatbot-Introduction&version=1">
# MAGIC
# MAGIC
