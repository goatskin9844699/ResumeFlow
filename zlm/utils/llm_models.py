'''
-----------------------------------------------------------------------
File: LLM.py
Creation Time: Nov 1st 2023 1:40 am
Author: Saurabh Zinjad
Developer Email: zinjadsaurabh1997@gmail.com
Copyright (c) 2023 Saurabh Zinjad. All rights reserved | GitHub: Ztrimus
-----------------------------------------------------------------------
'''
import json
import textwrap
import pandas as pd
import streamlit as st
import os
from openai import OpenAI
from langchain_community.llms.ollama import Ollama
from langchain_ollama import OllamaEmbeddings
import google.generativeai as genai
from google.generativeai.types.generation_types import GenerationConfig
import requests

from zlm.utils.utils import parse_json_markdown
from zlm.variables import GEMINI_EMBEDDING_MODEL, GPT_EMBEDDING_MODEL, OLLAMA_EMBEDDING_MODEL, LLM_MAPPING

def get_api_key(provider, api_key=None):
    """Get API key from environment variable or provided value"""
    env_var = LLM_MAPPING[provider]["api_env"]
    return api_key or os.getenv(env_var)

class ChatGPT:
    def __init__(self, api_key=None, model="gpt-4", system_prompt=""):
        if system_prompt.strip():
            self.system_prompt = {"role": "system", "content": system_prompt}
        self.client = OpenAI(api_key=get_api_key("GPT", api_key))
        self.model = model
    
    def get_response(self, prompt, expecting_longer_output=False, need_json_output=False):
        user_prompt = {"role": "user", "content": prompt}

        try:
            # TODO: Decide value(temperature, top_p, max_tokens, stop) to get apt response
            completion = self.client.chat.completions.create(
                model=self.model,
                messages = [self.system_prompt, user_prompt],
                temperature=0,
                max_tokens = 4000 if expecting_longer_output else None,
                response_format = { "type": "json_object" } if need_json_output else None
            )

            response = completion.choices[0].message
            content = response.content.strip()
            
            if need_json_output:
                return parse_json_markdown(content)
            else:
                return content
        
        except Exception as e:
            print(e)
            st.error(f"Error in OpenAI API, {e}")
            st.markdown("<h3 style='text-align: center;'>Please try again! Check the log in the dropdown for more details.</h3>", unsafe_allow_html=True)
    
    def get_embedding(self, text, model=GPT_EMBEDDING_MODEL, task_type="retrieval_document"):
        try:
            text = text.replace("\n", " ")
            return self.client.embeddings.create(input = [text], model=model).data[0].embedding
        except Exception as e:
            print(e)

class Gemini:
    # TODO: Test and Improve support for Gemini API
    def __init__(self, api_key=None, model="gemini-1.5-flash", system_prompt=""):
        genai.configure(api_key=get_api_key("Gemini", api_key))
        self.system_prompt = system_prompt
        self.model = model
    
    def get_response(self, prompt, expecting_longer_output=False, need_json_output=False):
        try:
            model = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=self.system_prompt
                )
            
            content = model.generate_content(
                contents=prompt,
                generation_config=GenerationConfig(
                    temperature=0.7,
                    max_output_tokens = 4000 if expecting_longer_output else None,
                    response_mime_type = "application/json" if need_json_output else None
                    )
                )

            if need_json_output:
                result = parse_json_markdown(content.text)
            else:
                result = content.text
            
            if result is None:
                st.write("LLM Response")
                st.markdown(f"```json\n{content.text}\n```")

            return result
        
        except Exception as e:
            print(e)
            st.error(f"Error in Gemini API, {e}")
            st.markdown("<h3 style='text-align: center;'>Please try again! Check the log in the dropdown for more details.</h3>", unsafe_allow_html=True)
            return None
    
    def get_embedding(self, content, model=GEMINI_EMBEDDING_MODEL, task_type="retrieval_document"):
        try:
            def embed_fn(data):
                result = genai.embed_content(
                    model=model,
                    content=data,
                    task_type=task_type,
                    title="Embedding of json text" if task_type in ["retrieval_document", "document"] else None)
                
                return result['embedding']
            
            df = pd.DataFrame(content)
            df.columns = ['chunk']
            df['embedding'] = df.apply(lambda row: embed_fn(row['chunk']), axis=1)
            
            return df
        
        except Exception as e:
            print(e)

class OllamaModel:
    def __init__(self, model, system_prompt):
        self.model = model
        self.system_prompt = system_prompt
    
    def get_response(self, prompt, expecting_longer_output=False, need_json_output=False):
        try:
            llm = Ollama(
                model=self.model, 
                system=self.system_prompt,
                temperature=0.8, 
                top_p=0.999, 
                top_k=250,
                num_predict=4000 if expecting_longer_output else None,
                # format='json' if need_json_output else None,
                )
            content = llm.invoke(prompt)

            if need_json_output:
                result = parse_json_markdown(content)
            else:
                result = content
            
            if result is None:
                st.write("LLM Response")
                st.markdown(f"```json\n{content.text}\n```")

            return result
        
        except Exception as e:
            print(e)
            st.error(f"Error in Ollama model - {self.model}, {e}")
            st.markdown("<h3 style='text-align: center;'>Please try again! Check the log in the dropdown for more details.</h3>", unsafe_allow_html=True)
            return None
    
    def get_embedding(self, content, model=OLLAMA_EMBEDDING_MODEL, task_type="retrieval_document"):
        try:
            def embed_fn(data):
                embedding = OllamaEmbeddings(model=model)
                result = embedding.embed_query(data)
                return result
            
            df = pd.DataFrame(content)
            df.columns = ['chunk']
            df['embedding'] = df.apply(lambda row: embed_fn(row['chunk']), axis=1)
            
            return df
        
        except Exception as e:
            print(e)

class OpenRouter:
    def __init__(self, api_key=None, model="openai/gpt-4", system_prompt=""):
        self.api_key = get_api_key("OpenRouter", api_key)
        self.model = model
        self.system_prompt = system_prompt
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/Ztrimus/job-llm",
            "Content-Type": "application/json"
        }
        self._available_models = None
        self._last_model_fetch = 0
        self._model_cache_duration = 3600  # Cache for 1 hour
    
    def get_available_models(self):
        """Fetch available models from OpenRouter API with caching."""
        import time
        
        current_time = time.time()
        
        # Return cached models if they exist and are not expired
        if self._available_models and (current_time - self._last_model_fetch) < self._model_cache_duration:
            return self._available_models
            
        try:
            # Models endpoint doesn't require authentication
            response = requests.get(
                f"{self.base_url}/models"
            )
            
            if response.status_code != 200:
                raise Exception(f"OpenRouter API error: {response.text}")
            
            models_data = response.json()
            # Extract model IDs from the response
            self._available_models = [model['id'] for model in models_data.get('data', [])]
            self._last_model_fetch = current_time
            return self._available_models
            
        except Exception as e:
            print(f"Error fetching OpenRouter models: {e}")
            # Return default models if API call fails
            return [
                "anthropic/claude-3-opus-20240229",
                "anthropic/claude-3-sonnet-20240229",
                "meta-llama/codellama-70b-instruct",
                "google/gemini-pro",
                "google/gemini-2.0-flash-lite-001",
                "openai/gpt-4-turbo-preview",
                "openai/gpt-3.5-turbo"
            ]
    
    def get_response(self, prompt, expecting_longer_output=False, need_json_output=False):
        user_prompt = {"role": "user", "content": prompt}

        try:
            messages = [self.system_prompt, user_prompt] if hasattr(self, 'system_prompt') else [user_prompt]
            
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 4000 if expecting_longer_output else None,
                "response_format": { "type": "json_object" } if need_json_output else None
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=data
            )
            
            if response.status_code != 200:
                raise Exception(f"OpenRouter API error: {response.text}")
            
            response_data = response.json()
            content = response_data['choices'][0]['message']['content'].strip()
            
            if need_json_output:
                return parse_json_markdown(content)
            else:
                return content
        
        except Exception as e:
            print(e)
            st.error(f"Error in OpenRouter API, {e}")
            st.markdown("<h3 style='text-align: center;'>Please try again! Check the log in the dropdown for more details.</h3>", unsafe_allow_html=True)
    
    def get_embedding(self, text, model=GPT_EMBEDDING_MODEL, task_type="retrieval_document"):
        try:
            text = text.replace("\n", " ")
            data = {
                "model": model,
                "input": [text]
            }
            
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json=data
            )
            
            if response.status_code != 200:
                raise Exception(f"OpenRouter API error: {response.text}")
            
            return response.json()['data'][0]['embedding']
        except Exception as e:
            print(e)