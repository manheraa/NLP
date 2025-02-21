import streamlit as st
import spacy
import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def paragraph_to_sentences(paragraph):
    """Splits a paragraph into individual sentences."""
    doc = nlp(paragraph)
    return [sent.text.strip() for sent in doc.sents]

def is_task_sentence(sentence):
    """Determines whether a sentence describes a task."""
    doc = nlp(sentence)
    return any(token.pos_ == "VERB" and token.dep_ in {"ROOT", "xcomp", "acl", "advcl"} for token in doc)

def clean_text(sentences):
    """Cleans sentences by removing punctuation while preserving capitalization."""
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
        doc = nlp(sentence)  # Keep case and structure
        filtered_tokens = [token.text for token in doc]
        cleaned_sentences.append(" ".join(filtered_tokens))
    return cleaned_sentences



def categorize_tasks(tasks):
    """Categorizes tasks using LDA, ensuring non-empty input."""
    if not tasks:
        return {}
    
    categories = ['Task-Oriented', 'Communication', 'Decision-Making', 'Personal']
    vectorizer = CountVectorizer(stop_words="english")
    
    try:
        task_matrix = vectorizer.fit_transform(tasks)
    except ValueError:
        return {}
    
    lda = LatentDirichletAllocation(n_components=len(categories), random_state=42)
    lda.fit(task_matrix)
    task_topics = lda.transform(task_matrix)
    
    structured_tasks = {category: [] for category in categories}
    for i, task in enumerate(tasks):
        category = categories[np.argmax(task_topics[i])]
        structured_tasks[category].append(task)
    
    return structured_tasks

def extract_task_details_spacy(structured_tasks):
    """Extracts assignees and deadlines from tasks using spaCy."""
    extracted_info = []
    for category, tasks in structured_tasks.items():
        for task in tasks:
            doc = nlp(task)  # Run spaCy on original (uncleaned) text
            
            assignees = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]  # Preserve case
            deadline = next((ent.text for ent in doc.ents if ent.label_ in ["TIME", "DATE"]), None)
            actions = [token.text for token in doc if token.pos_ == "VERB" and token.dep_ != "aux"]

            extracted_info.append({
                "task": task,
                "category": category,
                "assigned_to": assignees if assignees else None,
                "deadline": deadline,
                "action": actions if actions else None
            })
    return extracted_info



# Streamlit UI
st.title("📌 Task Extractor")
st.write("Enter a paragraph, and I will extract task details for you!")

paragraph = st.text_area("Enter paragraph:")
if st.button("Extract Tasks"):
    if paragraph.strip():
        sentences = paragraph_to_sentences(paragraph)
        tasks = [s for s in sentences if is_task_sentence(s)]
        cleaned_tasks = tasks
        structured_tasks = categorize_tasks(cleaned_tasks)
        task_details = extract_task_details_spacy(structured_tasks)
        
        if not task_details:
            st.warning("No valid tasks found. Try providing a clearer paragraph with specific tasks.")
        else:
            for task in task_details:
                st.subheader(f"📌 {task['category']}")
                st.write(f"**Task:** {task['task']}")
                if task['assigned_to']:
                    st.write(f"**Assigned to:** {', '.join(task['assigned_to'])}")
                if task['deadline']:
                    st.write(f"**Deadline:** {task['deadline']}")
                if task['action']:
                    st.write(f"**Actions:** {', '.join(task['action'])}")
    else:
        st.warning("Please enter a paragraph to extract tasks.")