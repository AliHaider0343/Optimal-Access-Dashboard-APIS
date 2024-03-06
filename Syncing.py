import sqlite3
import xml.etree.ElementTree as ET
from datetime import datetime
import requests
import pandas as pd
import re
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import numpy as np
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.
os.environ['OPENAI_API_KEY'] =  os.getenv('OPENAI_API_KEY')

source_column = "KuratedContent_sourceUrl"
metadata_columns = ['Channel_about', 'Channel_keywords', 'Collection_about', 'Collection_keywords', 'File_about',
                    'File_keywords', 'KuratedContent_author', 'KuratedContent_datePublished',
                    'KuratedContent_dateModified', 'KuratedContent_keywords', 'KuratedContent_publisher',
                    'KuratedContent_sourceUrl', 'KuratedContent_WordpressPopupUrl', 'user_id', 'knowledge_Store_id']
main_content_columns = ['KuratedContent_Description_and_headline']

text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)

chroma_db = Chroma(persist_directory=f"./Optimal-Access-Vector-Store", embedding_function=OpenAIEmbeddings())

def download_and_print_xml(url):
    def fix_br_tags(xml_string):
        fixed_xml_string = re.sub(r'<br([^>]*)>', r'<br\1/>', xml_string)
        fixed_xml_string = fixed_xml_string.replace('&amp;', '').replace('&apos;', '').replace('&nbsp;', '').replace(
            '&', 'and')
        return fixed_xml_string
    try:
        response = requests.get(url)
        if response.status_code == 200:
            xml_content = response.content.decode('utf-8')  # Decode using UTF-8
            fixed_xml_content = fix_br_tags(xml_content)
            return fixed_xml_content
        else:
            print(f"Error: Unable to fetch XML. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")
def extract_text_from_element(element):
    try:
        def get_text_from_element(element):
            text = element.text or ''
            for child in element:
                text += get_text_from_element(child)
            return text

        # Extract text content from the provided element
        text_content = get_text_from_element(element)
        return text_content.strip()
    except Exception as e:
        print(f"Error extracting text: {e}")
    return None
def parse_xml(xml_text, wordPress_website_link):
    root = ET.fromstring(xml_text)
    data = {
        'about': root.find('.//span[@itemprop="about"]').text,
        'comment': root.find('.//span[@itemprop="comment"]').text,
        'encoding': root.find('.//span[@itemprop="encoding"]').text,
        'publisher': root.find('.//span[@itemprop="publisher"]').text,
        'author': root.find('.//span[@itemprop="author"]').text,
        'keywords': root.find('.//span[@itemprop="keywords"]').text,
        'Channels': []
    }

    for channel in root.findall('.//group'):
        channel_data = {
            'about': channel.find('.//span[@itemprop="about"]').text,
            'comment': channel.find('.//span[@itemprop="comment"]').text,
            'encoding': channel.find('.//span[@itemprop="encoding"]').text,
            'keywords': channel.find('.//span[@itemprop="keywords"]').text,
            'Collections': []}
        for collection in channel.findall('.//page'):
            collection_data = {
                'about': collection.find('.//span[@itemprop="about"]').text,
                'comment': collection.find('.//span[@itemprop="comment"]').text,
                'encoding': collection.find('.//span[@itemprop="encoding"]').text,
                'keywords': collection.find('.//span[@itemprop="keywords"]').text,
                'KuratedContent': []}
            for artical in collection.findall('.//link'):
                articals_data = {
                    'ID': artical.find('.//span[@itemprop="ID"]').text,
                    'sourceUrl': artical.find('.//meta[@itemprop="mainEntityOfPage"]').get('itemid'),
                    'WordpressPopupUrl': str(
                        wordPress_website_link + clean_wordpress_Link(channel_data['about'], collection_data['about'], (
                            artical.find('.//h2[@itemprop="headline"]').text).strip())),
                    'headline': artical.find('.//h2[@itemprop="headline"]').text,
                    'author': artical.find('.//h3[@itemprop="author"]/span[@itemprop="name"]').text,
                    'description': extract_text_from_element(artical.find('.//span[@itemprop="description"]')),
                    'publisher': artical.find('.//div[@itemprop="publisher"]/meta[@itemprop="name"]').get('content'),
                    'datePublished': artical.find('.//meta[@itemprop="datePublished"]').get('content'),
                    'dateModified': artical.find('.//meta[@itemprop="dateModified"]').get('content'),
                    'keywords': artical.find('.//span[@itemprop="keywords"]').text
                }
                collection_data['KuratedContent'].append(articals_data)
            channel_data['Collections'].append(collection_data)
        data['Channels'].append(channel_data)
    return data
def clean_text(text):
    # Use regular expression to remove unwanted characters
    cleaned_text = text.replace("'", "")
    cleaned_text = cleaned_text.replace("/", "")
    cleaned_text = re.sub(r'[^a-zA-Z0-9\'’-“”]', '-', cleaned_text)
    # Replace empty spaces with hyphens
    cleaned_text = cleaned_text.replace(' ', '-')
    # Remove extra hyphens and convert to lowercase
    cleaned_text = re.sub(r'-+', '-', cleaned_text).lower()
    return cleaned_text.strip('-')
def clean_wordpress_Link(channel_name, collection_name, articals_headline):
    channel_name = channel_name.replace(' ', '-')
    collection_name = collection_name.replace(' ', '-').lower()
    articals_headline = clean_text(articals_headline.replace(' ', '-').lower())
    return '/' + channel_name + '/' + collection_name + '/' + articals_headline

def Append_Single_file_Articals(data):
    Single_data_Collection = []
    for channel_data in data['Channels']:
        for collection_data in channel_data['Collections']:
            for articals_data in collection_data['KuratedContent']:
                row = {
                    'KuratedContent_article_id': articals_data['ID'],
                    'KuratedContent_sourceUrl': articals_data['sourceUrl'],
                    'KuratedContent_WordpressPopupUrl': articals_data['WordpressPopupUrl'],
                    'KuratedContent_headline': articals_data['headline'],
                    'KuratedContent_author': articals_data['author'],
                    'KuratedContent_description': articals_data['description'],
                    'KuratedContent_publisher': articals_data['publisher'],
                    'KuratedContent_datePublished': articals_data['datePublished'],
                    'KuratedContent_dateModified': articals_data['dateModified'],
                    'KuratedContent_keywords': articals_data['keywords'],
                    'Collection_about': collection_data['about'],
                    'Collection_comment': collection_data['comment'],
                    'Collection_encoding': collection_data['encoding'],
                    'Collection_keywords': collection_data['keywords'],
                    'Channel_about': channel_data['about'],
                    'Channel_comment': channel_data['comment'],
                    'Channel_encoding': channel_data['encoding'],
                    'Channel_keywords': channel_data['keywords'],
                    'File_about': data['about'],
                    'File_comment': data['comment'],
                    'File_encoding': data['encoding'],
                    'File_publisher': data['publisher'],
                    'File_author': data['author'],
                    'File_keywords': data['keywords']
                }
                Single_data_Collection.append(row)
    return Single_data_Collection

def Collect_Process_Data(parsed_data_against_urls, user_id, chatbot_id):
    Articals_Collection = []
    for data in parsed_data_against_urls:
        Articals_Collection.extend(Append_Single_file_Articals(data))
    Dataset_csv = pd.DataFrame(Articals_Collection)
    Dataset_csv['user_id'] = user_id
    Dataset_csv['knowledge_Store_id'] = chatbot_id
    Dataset_csv['KuratedContent_Description_and_headline'] = Dataset_csv['KuratedContent_headline'] + ':' + Dataset_csv[
        'KuratedContent_description']
    columns_to_drop = ['KuratedContent_headline', 'KuratedContent_description', 'KuratedContent_article_id',
                       'Collection_comment', 'Collection_encoding', 'Channel_comment', 'Channel_encoding',
                       'File_comment', 'File_encoding', 'File_author', 'File_publisher']
    Dataset_csv = Dataset_csv.drop(columns_to_drop, axis=1)
    Dataset_csv = Dataset_csv.drop_duplicates()
    Dataset_csv.replace({np.nan: "Not Specified"}, inplace=True)
    Dataset_csv.reset_index()
    return Dataset_csv

def filter_information(stored_latest_article_date, processed_data):
    # Convert the 'KuratedContent_dateModified' column to int64
    processed_data['KuratedContent_dateModified'] = processed_data['KuratedContent_dateModified'].astype('int64')
    # Filter based on the condition
    filtered_data = processed_data[processed_data['KuratedContent_dateModified'] > int(stored_latest_article_date)]
    return filtered_data

def Generate_documents_from_dataframe(processed_data):
    documents = []
    for index, row in processed_data.iterrows():
        data = row[main_content_columns[0]]
        metadata = {column: row[column] for column in metadata_columns}
        document = Document(page_content=data, metadata=metadata)
        documents.append(document)
    return documents

def update_documents_to_delete_pervious_versions_of_updated_Data(chroma_db, filtered_source_urls_list):
    documents = chroma_db.get()
    count = 0
    for source in filtered_source_urls_list:
        for document_id, metadata in zip(documents['ids'], documents['metadatas']):
            if str(metadata['KuratedContent_sourceUrl']) == str(source):
                chroma_db.delete([document_id])
                count += 1
    return count>0

def update_store(knowledgeStoreid,user_id,urls,wordpressUrls):
    Urls_for_chatbot = [urls]
    urls_corresponding_wordPress_website_links = [wordpressUrls]
    try:
        parsed_data_against_urls = []
        for url, wordpress_link in zip(Urls_for_chatbot, urls_corresponding_wordPress_website_links):
            parsed_data_against_urls.append(parse_xml(download_and_print_xml(url), wordpress_link))
        processed_data = Collect_Process_Data(parsed_data_against_urls, user_id, knowledgeStoreid)
        documents = chroma_db.get()
        filtered_documents={'ids':[],
                            'embeddings':[],
                            'metadatas': [],
                            'document': [],
                            'uris': [],
                            'data': []
                            }
        for ids, embeddings, metadatas, document, uris, data in zip(documents['ids'],
                                                                    [None] * len(documents['ids']),
                                                                    documents['metadatas'],
                                                                    documents['documents'],
                                                                    [None] * len(documents['ids']),
                                                                    [None] * len(documents['ids'])):

            if str(metadatas['knowledge_Store_id'])==str(knowledgeStoreid) and str(metadatas['user_id'])==str(user_id):
                filtered_documents['ids'].append(ids)
                filtered_documents['embeddings'].append(embeddings)
                filtered_documents['metadatas'].append(metadatas)
                filtered_documents['document'].append(document)
                filtered_documents['uris'].append(uris)
                filtered_documents['data'].append(data)

        unique_dates = set()
        for metadata in filtered_documents['metadatas']:
            unique_dates.add(int(metadata['KuratedContent_dateModified']))
        stored_latest_artical_date = max(list(unique_dates))
        filtered_data = filter_information(stored_latest_artical_date, processed_data)
        filtered_documents = Generate_documents_from_dataframe(filtered_data)
        update_documents_to_delete_pervious_versions_of_updated_Data(chroma_db, list(filtered_data['KuratedContent_sourceUrl']))
        texts = text_splitter.split_documents(filtered_documents)
        print(len(texts))
        if len(texts) > 0:
            chroma_db.add_documents(texts)
    except Exception as e:
        print('Exception',e)

def get_tasks_to_perform(conn):
    cur = conn.cursor()
    cur.execute("SELECT * ,strftime(strftime('%s', last_performed) + (syncing_period * 3600)), strftime('%s', 'now') FROM Sycing_Jobs WHERE strftime('%s', 'now') >= strftime(strftime('%s', last_performed) + (syncing_period * 3600))")
    tasks = cur.fetchall()
    return tasks

def update_status(conn,jobId):
    cur = conn.cursor()
    current_time =datetime.now()
    # Update last_performed for the given job_id
    cur.execute("UPDATE Sycing_Jobs SET last_performed = ? WHERE job_id = ?", (current_time, jobId))
    # Commit changes and close connection
    conn.commit()


database = "Database.db"
conn = sqlite3.connect(database)

if __name__ == "__main__":
    jobs=get_tasks_to_perform(conn)
    print(jobs)
    for job in jobs:
        update_store(job[1],job[2],job[3],job[4])
        update_status(conn,job[0])
