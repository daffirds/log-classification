from pandas import pd
from dotenv import load_dotenv
from groq import Groq
import re
import os
import joblib
from sentence_transformers import SentenceTransformer

load_dotenv()

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

model_embedding = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model
bert_model = joblib.load("models/bert_model.joblib")

def bert_classify(log_message):
    embeddings = model_embedding.encode([log_message])
    probabilities = bert_model.predict_proba(embeddings)[0]
    if max(probabilities) < 0.5:
        return "Unknown"
    predicted_label = bert_model.predict(embeddings)[0]
    
    return predicted_label

def llm_classify(log_message):
    prompt = f'''Classify the log message into one of these categories: 
    (1) Workflow Error, (2) Deprecation Warning.
    If you can't figure out a category, return "Unknown". 
    Log message: {log_message}'''

    chat_completion = groq.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-oss-20b"
    )

    content = chat_completion.choices[0].message.content
    match = re.search(r'.*', content, flags=re.DOTALL)
    category = "Unknown"
    if match:
        category = match.group(1)
    return category

def classify(logs):
    labels = []
    for source, log_msg in logs:
        label = classify_log(source, log_msg)
        labels.append(label)
    return labels


def classify_log(source, log_msg):
    if source == "LegacyCRM":
        label = llm_classify(log_msg)
    else:
        label = bert_classify(log_msg)
    return label

def classify_csv(input_file):
    import pandas as pd
    df = pd.read_csv(input_file)
    if "source" not in df.columns or "log_message" not in df.columns:
        raise ValueError("CSV must contain 'source' and 'log_message' columns.")
    
    # Perform classification
    df["target_label"] = classify(list(zip(df["source"], df["log_message"])))

    # Save the modified file
    output_file = "output.csv"
    df.to_csv(output_file, index=False)

    return output_file

if __name__ == '__main__':
    logs = [
        ("ModernCRM", "IP 192.168.133.114 blocked due to potential attack"),
        ("BillingSystem", "User 12345 logged in."),
        ("AnalyticsEngine", "File data_6957.csv uploaded successfully by user User265."),
        ("AnalyticsEngine", "Backup completed successfully."),
        ("ModernHR", "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1 RCODE  200 len: 1583 time: 0.1878400"),
        ("ModernHR", "Admin access escalation detected for user 9429"),
        ("LegacyCRM", "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."),
        ("LegacyCRM", "Invoice generation process aborted for order ID 8910 due to invalid tax calculation module."),
        ("LegacyCRM", "The 'BulkEmailSender' feature is no longer supported. Use 'EmailCampaignManager' for improved functionality."),
        ("LegacyCRM", " The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025")
    ]
    logs = pd.DataFrame(logs, columns=["source", "log_message"])
    logs.to_csv("test/input_test.csv", index=False)
    classify_csv("test/input_test.csv")