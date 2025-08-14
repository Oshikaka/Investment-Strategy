from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd

# NLP model: ProsusAI/finbert
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create a sentiment analysis pipeline
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, top_k=None)

# Input sentences/news
# !!!!!!!! ⚠️ Make sure to put the sentences into "" to avoid syntax errors !!!!!!!!!!
# In the format: "sentence1", "sentence2", ...
# Remember to add a comma after each sentence except the last one
sentences = [
    "there is a shortage of capital, and we need extra financing",
    "growth is strong and we have plenty of liquidity",
    "there are doubts about our finances",
    "Our company is broke, and we need to raise funds",
]

# Get sentiment analysis results
# nlp() returns a list of dictionaries, each with 'label' and 'score'
results = nlp(sentences)


# Convert results to a DataFrame
# The labels used by ProsusAI/finbert model are: positive, negative, neutral (lowercase)
# The DataFrame will have columns: Sentence, Positive, Negative, Neutral
# Note: The model outputs labels in lowercase, so we will capitalize them for consistency
data = []
for sentence, res in zip(sentences, results):
    row = {"Sentence": sentence}
    for item in res:  # item is dict: {'label': 'positive', 'score': 0.99}
        row[item['label'].capitalize()] = item['score']
    data.append(row)

df = pd.DataFrame(data)
print(df)
# Example output:
#                                             Sentence  Negative   Neutral  Positive
# 0  there is a shortage of capital, and we need ex...  0.865358  0.105948  0.028694
# 1   growth is strong and we have plenty of liquidity  0.010695  0.086726  0.902579
# 2                there are doubts about our finances  0.426424  0.456974  0.116602
# 3   Our company is broke, and we need to raise funds  0.915018  0.063115  0.021867

# Save the DataFrame to a CSV file
# It can open in Excel or any spreadsheet software
file_name = "financial_news_sentiment.csv" # You can change the file name as needed
df.to_csv(file_name, index=False)
