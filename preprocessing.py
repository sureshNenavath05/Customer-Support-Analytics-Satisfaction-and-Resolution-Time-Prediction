import re
import pandas as pd
import numpy as np
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize NLP tools
stop = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

def age_group(age):
    if age < 25:
        return "Youth"
    elif age < 60:
        return "Adult"
    else:
        return "Senior"

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop]
    return ' '.join(words)

def process_input_data(
    df,
    task,
    model_cols,
    desc_vectorizer,
    res_vectorizer
):
    df = df.copy()
    df_output = df.copy()

    drop_cols = ['Customer Name', 'Customer Email', 'Ticket ID', 'Ticket Status']
    if task == "resolution":
        drop_cols.append('Time to Resolution')

    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

    # Convert dates
    for col in ['Date of Purchase', 'First Response Time', 'Time to Resolution']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Datetime-based features
    df['purchase_month'] = df['Date of Purchase'].dt.month
    df['purchase_year'] = df['Date of Purchase'].dt.year
    df['Is_Weekend_Purchase'] = df['Date of Purchase'].dt.weekday >= 5

    df['first_response_days'] = (df['First Response Time'] - df['Date of Purchase']).dt.total_seconds() / (60 * 60 * 24)
    df['Response_Delay_Hours'] = (df['First Response Time'] - df['Date of Purchase']).dt.total_seconds() / 3600

    if task == "satisfaction" and 'Time to Resolution' in df.columns:
        df['resolution_days'] = (df['Time to Resolution'] - df['Date of Purchase']).dt.total_seconds() / (60 * 60 * 24)
        df['Resolution_Time_Hours'] = (df['Time to Resolution'] - df['First Response Time']).dt.total_seconds() / 3600
        df['Total_Time_Hours'] = (df['Time to Resolution'] - df['Date of Purchase']).dt.total_seconds() / 3600
        df['Resolution_vs_Response'] = df['Resolution_Time_Hours'] / (df['Response_Delay_Hours'] + 1e-3)
        df['Total_vs_Response'] = df['Total_Time_Hours'] / (df['Response_Delay_Hours'] + 1e-3)
        df['Total_vs_Resolution'] = df['Total_Time_Hours'] / (df['Resolution_Time_Hours'] + 1e-3)

    # Age & Text
    df['customer_age_group'] = df['Customer Age'].apply(age_group)
    df['clean_description'] = df['Ticket Description'].apply(preprocess_text)
    df['clean_subject'] = df['Ticket Subject'].apply(preprocess_text)
    df['ticket_length'] = df['clean_description'].apply(lambda x: len(x.split()))
    df['is_critical'] = df['Ticket Priority'].str.lower().str.strip().apply(lambda x: 1 if x == 'critical' else 0)

    df['sentiment_score'] = df['Ticket Description'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    
    keywords = [
    'angry', 'disappointed', 'frustrated', 'concerned',
    'problem', 'issue', 'error', 'bug', 'fail', 'help', 'support', 'crash', 'freeze', 'not working', 'stuck',
    'urgent', 'as soon as possible', 'immediately', 'affecting my work', 'productivity', 'critical', 'emergency',
    'broken', 'missing', 'incompatible', 'overheating', 'restart', 'reset', 'setup', 'installation', 'update', 'slow',
    'refund', 'cancel', 'delay', 'late', 'waiting', 'no response', 'wrong item', 'faulty',
    'login', 'payment', 'disconnect', 'security', 'data', 'software update', 'troubleshooting', 'manual', 'unresolved',
    'already contacted', 'still not working', 'persisting', 'didn’t help', 'error message',
]
    for kw in keywords:
        df[f'has_{kw.replace(" ", "_")}'] = df['clean_description'].str.contains(kw, case=False, regex=False).astype(int)

    priority_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
    df['Priority_Num'] = df['Ticket Priority'].map(priority_map).fillna(0)
    df['priority_critical'] = df['Priority_Num'] * df['is_critical']
    df['age_ticket_length'] = df['Customer Age'] * df['ticket_length']
    df['sentiment_priority'] = df['sentiment_score'] * df['Priority_Num']
    df['sentiment_age'] = df['sentiment_score'] * df['Customer Age']
    df['sentiment_length'] = df['sentiment_score'] * df['ticket_length']

    # VADER Sentiment
    df['desc_sentiment'] = df['Ticket Description'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    df['res_sentiment'] = df['Resolution'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    df['Net_Sentiment'] = df['res_sentiment'] - df['desc_sentiment']

    df['Description_Length'] = df['Ticket Description'].fillna('').apply(lambda x: len(str(x).split()))
    df['Resolution_Length'] = df['Resolution'].fillna('').apply(lambda x: len(str(x).split()))
    df['desc_vs_res_len_ratio'] = (df['Description_Length'] + 1) / (df['Resolution_Length'] + 1)
    df['resolution_quality'] = df['res_sentiment'] * df['Resolution_Length']
    df['Empty_Resolution'] = df['Resolution'].apply(lambda x: 1 if pd.isna(x) or len(str(x).strip()) == 0 else 0)

    df['Urgency_Mismatch'] = df['Response_Delay_Hours'] / (df['Priority_Num'] + 1e-3)

    df['clean_Description'] = df['Ticket Description'].fillna('').apply(clean_text)
    df['clean_Resolution'] = df['Resolution'].fillna('').apply(clean_text)

    # TF-IDF Vectorization (use pre-loaded vectorizers)
    desc_matrix = desc_vectorizer.transform(df['clean_Description'])
    desc_df = pd.DataFrame(desc_matrix.toarray(), columns=[f'desc_tfidf_{i}' for i in range(desc_matrix.shape[1])])

    res_matrix = res_vectorizer.transform(df['clean_Resolution'])
    res_df = pd.DataFrame(res_matrix.toarray(), columns=[f'res_tfidf_{i}' for i in range(res_matrix.shape[1])])

    df_model = pd.concat([df.reset_index(drop=True), desc_df, res_df], axis=1)

    # Drop unnecessary columns
    drop_cols = [
        'Ticket Description', 'Resolution',
        'Date of Purchase', 'First Response Time', 'Time to Resolution',
        'Customer Satisfaction Rating', 'clean_Description', 'clean_Resolution',
        'clean_description', 'Ticket Priority'
    ]
    df_model.drop(columns=[c for c in drop_cols if c in df_model.columns], inplace=True)

    # One-hot encode remaining categorical
    cat_cols = df_model.select_dtypes(include='object').columns.tolist()
    df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

    # Align with model columns
    for col in model_cols:
        if col not in df_model.columns:
            df_model[col] = 0
    df_model = df_model[model_cols]
    df_model.fillna(0, inplace=True)

    return df_model, df_output
