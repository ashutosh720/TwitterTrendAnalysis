# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load the model and tokenizer
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load a smaller spaCy model for efficiency
nlp = spacy.load("en_core_web_sm")

# Extend stop words
STOP_WORDS = set(STOPWORDS).union(set(['rt', 'via', 'amp', 'http', 'https', 'co']))
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def batch_summarize(texts, batch_size=8):
    summaries = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
        summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0,
                                     num_beams=4, early_stopping=True)
        batch_summaries = [tokenizer.decode(g, skip_special_tokens=True) for g in summary_ids]
        summaries.extend(batch_summaries)
    return summaries

def generate_summaries(df):
    tweet_batches = [df['processed_text'][i:i + 100].str.cat(sep=' ') for i in range(0, len(df), 100)]
    batch_summaries = batch_summarize(tweet_batches)
    overall_summary = batch_summarize([' '.join(batch_summaries)])[0]
    return batch_summaries, overall_summary