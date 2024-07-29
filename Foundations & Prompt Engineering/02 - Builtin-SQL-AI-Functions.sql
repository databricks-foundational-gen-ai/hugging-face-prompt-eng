-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Process and analyse text with Built In Databricks SQL AI functions
-- MAGIC
-- MAGIC Databricks SQL provides [built-in GenAI capabilities](https://docs.databricks.com/en/large-language-models/ai-functions.html), letting you perform adhoc operation, leveraging state of the art LLM, optimized for these tasks.
-- MAGIC
-- MAGIC These functions are the following:
-- MAGIC
-- MAGIC - `ai_analyze_sentiment`
-- MAGIC - `ai_classify`
-- MAGIC - `ai_extract`
-- MAGIC - `ai_fix_grammar`
-- MAGIC - `ai_gen`
-- MAGIC - `ai_mask`
-- MAGIC - `ai_similarity`
-- MAGIC - `ai_summarize`
-- MAGIC - `ai_translate`
-- MAGIC
-- MAGIC Using these functions is pretty straightforward. Under the hood, they call specialized LLMs with a custom prompt, providing fast answers.
-- MAGIC
-- MAGIC You can use them on any column in any SQL text.
-- MAGIC
-- MAGIC Let's give it a try.
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Using a SQL Warehouse to run this notebook
-- MAGIC
-- MAGIC This demo runs using a SQL Warehouse! 
-- MAGIC
-- MAGIC Make sure you select one using the dropdown on the top right of your notebook (don't select a classic compute/cluster)
-- MAGIC
-- MAGIC

-- COMMAND ----------

-- DBTITLE 1,Generate dataset
-- verify that we're running on a SQL Warehouse
SELECT assert_true(current_version().dbsql_version is not null, 'YOU MUST USE A SQL WAREHOUSE, not a cluster');

SELECT ai_gen('Generate a concise, cheerful email title for a summer bike sale with 20% discount');

-- COMMAND ----------

-- DBTITLE 1,Fix grammar
SELECT ai_fix_grammar('This sentence have some mistake');

-- COMMAND ----------

-- DBTITLE 1,Automatically classify text into categories
SELECT ai_classify("My password is leaked.", ARRAY("urgent", "not urgent"));

-- COMMAND ----------

-- DBTITLE 1,Translate into other language
SELECT ai_translate("This function is so amazing!", "fr")

-- COMMAND ----------

-- DBTITLE 1,Compute similarity between sentences
SELECT ai_similarity('Databricks', 'Apache Spark'),  ai_similarity('Apache Spark', 'The Apache Spark Engine');

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ## Going further: creating your own AI function with your Model Serving Endpoint or other Foundation Models
-- MAGIC
-- MAGIC #### Generating a more complete sample dataset with prompt engineering
-- MAGIC
-- MAGIC Now that we know how to send a basic query to Open AI using SQL functions, let's ask the model a more detailed question.
-- MAGIC
-- MAGIC We'll directly ask to model to generate multiple rows and directly return as a json. 
-- MAGIC
-- MAGIC Here's a prompt example to generate JSON:
-- MAGIC ```
-- MAGIC Generate a sample dataset for me of 2 rows that contains the following columns: "date" (random dates in 2022), 
-- MAGIC "review_id" (random id), "product_name" (use popular grocery product brands), and "review". Reviews should mimic useful product reviews 
-- MAGIC left on an e-commerce marketplace website. 
-- MAGIC
-- MAGIC The reviews should vary in length (shortest: one sentence, longest: 2 paragraphs), sentiment, and complexity. A very complex review 
-- MAGIC would talk about multiple topics (entities) about the product with varying sentiment per topic. Provide a mix of positive, negative, 
-- MAGIC and neutral reviews
-- MAGIC
-- MAGIC Return JSON ONLY. No other text outside the JSON. JSON format:
-- MAGIC [{"review_date":<date>, "review_id":<review_id>, "product_name":<product_name>, "review":<review>}]
-- MAGIC ```

-- COMMAND ----------

-- DBTITLE 1,Generate the fake data
-- MAGIC %run ./init/config_generation

-- COMMAND ----------

SELECT * FROM fake_reviews

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Automated product review and classification with SQL functions
-- MAGIC
-- MAGIC
-- MAGIC In this demo, we will explore the SQL AI function `ai_query` to create a pipeline extracting product review information.
-- MAGIC
-- MAGIC <img src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/main/images/product/sql-ai-functions/sql-ai-query-function-flow.png" width="1000">
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ## Simplifying AI function access for SQL users (*CHANGE THIS IMAGE)
-- MAGIC
-- MAGIC As reminder, `ai_query` signature is the following:
-- MAGIC
-- MAGIC ```
-- MAGIC SELECT ai_query(<Endpoint Name>, <prompt>)
-- MAGIC ```
-- MAGIC
-- MAGIC In order to simplify the user-experience for our analysts, we will build prescriptive SQL functions that ask natural language questions of our data and return the responses as structured data.

-- COMMAND ----------

SELECT * FROM fake_reviews INNER JOIN fake_customers using (customer_id)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Review analysis with prompt engineering 
-- MAGIC &nbsp;
-- MAGIC The keys to getting useful results back from a LLM model are:
-- MAGIC - Asking it a well-formed question
-- MAGIC - Being specific about the type of answer that you are expecting
-- MAGIC
-- MAGIC In order to get results in a form that we can easily store in a table, we'll ask the model to return the result in a string that reflects `JSON` representation, and be very specific of the schema that we expect

-- COMMAND ----------

CREATE OR REPLACE FUNCTION ANNOTATE_REVIEW(review STRING)
    RETURNS STRUCT<product_name: STRING, entity_sentiment: STRING, followup: STRING, followup_reason: STRING>
    RETURN FROM_JSON(
      ASK_LLM_MODEL(CONCAT(
        'A customer left a review. We follow up with anyone who appears unhappy.
         extract the following information:
          - classify sentiment as ["POSITIVE","NEUTRAL","NEGATIVE"]
          - returns whether customer requires a follow-up: Y or N
          - if followup is required, explain what is the main reason

        Return JSON ONLY. No other text outside the JSON. JSON format:
        {
            "product_name": <entity name>,
            "entity_sentiment": <entity sentiment>,
            "followup": <Y or N for follow up>,
            "followup_reason": <reason for followup>
        }
        
        Review:', review)),
      "STRUCT<product_name: STRING, entity_sentiment: STRING, followup: STRING, followup_reason: STRING>")

-- ALTER FUNCTION ANNOTATE_REVIEW OWNER TO `your_principal`; -- for the demo only, make sure other users can access your function

-- COMMAND ----------

CREATE OR REPLACE TABLE reviews_annotated as 
    SELECT * EXCEPT (review_annotated), review_annotated.* FROM (
      SELECT *, ANNOTATE_REVIEW(review) AS review_annotated
        FROM fake_reviews LIMIT 10)
    INNER JOIN fake_customers using (customer_id)

-- COMMAND ----------

SELECT * FROM reviews_annotated

-- COMMAND ----------

CREATE OR REPLACE FUNCTION GENERATE_RESPONSE(firstname STRING, lastname STRING, article_this_year INT, product STRING, reason STRING)
  RETURNS STRING
  RETURN ASK_LLM_MODEL(
    CONCAT("Our customer named ", firstname, " ", lastname, " who ordered ", article_this_year, " articles this year was unhappy about ", product, 
    "specifically due to ", reason, ". Provide an empathetic message I can send to my customer 
    including the offer to have a call with the relevant product manager to leave feedback. I want to win back their 
    favour and I do not want the customer to churn")
  );
-- ALTER FUNCTION GENERATE_RESPONSE OWNER TO `account users`; -- for the demo only, make sure other users can access your function

-- COMMAND ----------

SELECT GENERATE_RESPONSE("Summer", "He", 235, "Country Choice Snacking Cookies", "Quality issue") AS customer_response
