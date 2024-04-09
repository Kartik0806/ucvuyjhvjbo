import streamlit as st
import numpy as np 
import pandas as pd 
import os
from PIL import Image
import requests
from sentence_transformers import SentenceTransformer, util
from IPython.display import display
from IPython.display import Image as IPImage
from io import StringIO


st.header('Search Engine')

model = SentenceTransformer('clip')
text_embedder = SentenceTransformer('miniLM')
flickr = pd.read_csv("caption.csv")
flickr_text_embeddings = pd.read_csv("text_embeddings.csv", header=None).to_numpy()
flickr_image_embeddings = pd.read_csv("image_embeddings.csv",header=None).to_numpy()
flickr_text_embeddings = flickr_text_embeddings.astype(np.float32)
flickr_image_embeddings = flickr_image_embeddings.astype(np.float32)

img_names = flickr['image'].to_numpy()
cap = flickr['caption'].to_numpy()
img_folder = "Images"
toggle = [0,0]
def search_by_text(query, k=5):
# with st.empty():
    query_emb = text_embedder.encode([query], convert_to_tensor=True, show_progress_bar=False)
    hits = util.semantic_search(query_emb, flickr_text_embeddings, top_k=k)[0]
    img_paths = []
    captions = []
    for hit in hits:
        img_path = os.path.join(img_folder, img_names[hit['corpus_id']])
        img_paths.append(img_path)
        captions.append(cap[hit['corpus_id']])
        # display(cap[hit['corpus_id']])
    st.image(img_paths, caption=captions)
        

def search_by_image(query, k=5):
    # with st.empty():
    query_emb = model.encode(query, convert_to_tensor=True, show_progress_bar=False)
    hits = util.semantic_search(query_emb, flickr_image_embeddings, top_k=k)[0]
    img_paths = []
    captions = []
    for hit in hits:
        img_path = os.path.join(img_folder, img_names[hit['corpus_id']])
        img_paths.append(img_path)
        captions.append(cap[hit['corpus_id']])
        # display(cap[hit['corpus_id']])
    st.image(img_paths, caption=captions)
def main():

    uploaded_file = st.file_uploader("Choose a file",type=['jpg','jpeg','jpg','png'])
    title = st.text_input('Doogle Search', None)
    if uploaded_file is not None:
        image=Image.open(uploaded_file)
        search_by_image(image)
    if title is not None:
        search_by_text(title)

if __name__=="__main__":
    main()