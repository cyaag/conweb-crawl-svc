import requests
from bs4 import BeautifulSoup
import json
import logging
import boto3
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from flask_cors import CORS
from flask import Flask, jsonify, Response, request
from collections import defaultdict
# from langchain_openai import OpenAIEmbeddings
import io
import gzip

# Set the cache directory to /tmp
os.environ['TRANSFORMERS_CACHE'] = '/tmp'
# Initialize Pinecone
api_key = str(os.environ.get("PINEACCESSKEY", default=None))
aws_access_key_id = str(os.environ.get("AWSACCESSKEY", default=None))
aws_secret_access_key = str(os.environ.get("AWSSECRETKEY", default=None))
aws_region = "us-east-1"
pc = Pinecone(api_key=api_key)
index_name = 'webindex'

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
index = pc.Index(index_name)

bedrock_runtime = boto3.client(
    service_name = "bedrock-runtime",
    region_name = "us-east-1"
)

# Instantiate a bedrock instance
bedrock = boto3.client(
    service_name = "bedrock",
    region_name = "us-east-1"
)

# Start a session with AWS using boto3
# session = boto3.Session(profile_name='default')

session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

# Initialize the bedrock-runtime client to interact with AWS AI services
bedrock = session.client(service_name='bedrock-runtime')

# modelId = 'amazon.titan-text-express-v1'
modelId = 'amazon.titan-embed-text-v1'
accept = 'application/json'
contentType = 'application/json'

bedrock_embeddings = BedrockEmbeddings(client=bedrock_runtime)

def get_text_from_html(content, url):
    try:
        soup = BeautifulSoup(content, 'html.parser')
        result_text = {
            "title": soup.title.string if soup.title else "",
            "h1": soup.h1.string if soup.h1 else "",
            "h2": soup.h2.string if soup.h2 else "",
            "h3": soup.h3.string if soup.h3 else "",
            "p": " ".join(p.get_text() for p in soup.find_all('p')),
            "links": [
                {"text": a.get_text(), "href": a.get('href')}
                for a in soup.find_all('a')
            ],
            "tabs": [
                {
                    "tab-link": [
                        {"text": link.get_text(), "href": link.get('href')}
                        for link in tab_div.find_all(class_='tab-link')
                    ]
                }
                for tab_div in soup.find_all('div', class_='tabs')
            ]
        }
        text_content = " ".join(v for v in result_text.values() if isinstance(v, str))
        return text_content

    except Exception as e:
        logging.error('Failed to process HTML: %s', e)
        raise

def lambda_handler(event, context):
    for record in event['Records']:
        body = json.loads(record['body'])
        sns_message = json.loads(body.get('Message', '{}'))
        website_url = sns_message.get('url')
        print(website_url)
        if not website_url:
            logging.error('No URL found in the request')
            return {"statusCode": 400, "body": json.dumps({"error": "URL is required"})}

        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; MSIE 6.0; Windows NT 5.1)'}
            response = requests.get(website_url, headers=headers)
            response.raise_for_status()
            content = response.content

            if response.headers.get('Content-Encoding') == 'gzip' or content[:2] == b'\x1f\x8b':
                buf = io.BytesIO(content)
                try:
                    with gzip.GzipFile(fileobj=buf) as f:
                        content = f.read().decode('utf-8')
                except OSError as e:
                    logging.error("Error decompressing gzipped content: %s", e)
                    content = response.text
            else:
                content = content.decode('utf-8')

            text_data = get_text_from_html(content, website_url)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(text_data)

            for chunk in chunks:
                embedding = bedrock_embeddings.embed_documents([chunk])[0]
                index.upsert(vectors=[{
                    'id': f"{website_url}",
                    'values': embedding,
                    'metadata': {'text': content}
                }])

            logging.info("Text data processed and indexed")
            return {"statusCode": 200, "body": json.dumps({'message': 'Text data processed and indexed'})}

        except requests.RequestException as e:
            logging.error(f"Failed to retrieve {website_url}: {e}")
            return {"statusCode": 500, "body": json.dumps({'error': f"Failed to retrieve {website_url}: {e}"})}

        except Exception as e:
            logging.error(f"Failed to process text data: {e}")
            return {"statusCode": 500, "body": json.dumps({'error': f"Failed to process text data: {e}"})}
