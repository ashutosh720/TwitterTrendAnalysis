@functools.lru_cache(maxsize=10000)
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+|@\w+|#\w+|[^a-zA-Z\s]', '', text)
        return text.lower().strip()
    return ''

@functools.lru_cache(maxsize=10000)
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

@functools.lru_cache(maxsize=10000)
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(tokens)

def improved_preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and token.lemma_.lower() not in STOP_WORDS]
    return ' '.join(tokens)

def extract_topics(texts, num_topics=5):
    processed_texts = [improved_preprocess_text(text) for text in texts]
    dictionary = corpora.Dictionary([text.split() for text in processed_texts])
    corpus = [dictionary.doc2bow(text.split()) for text in processed_texts]

    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=2)

    topics = lda_model.print_topics(num_words=3)
    return [' '.join([word.split('"')[1] for word in topic[1].split('+')]) for topic in topics]

def improved_summarization(df):
    df['processed_text'] = df['tweetText'].apply(improved_preprocess_text)

    topics = extract_topics(df['processed_text'])

    overall_text = ' '.join(df['processed_text'])
    inputs = tokenizer(overall_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4,
                                 early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    sentences = summary.split('. ')
    structured_summary = []

    if len(sentences) >= 4:
        structured_summary = [
            "The tweets discuss various topics:",
            f"1. Main theme: {sentences[0]}",
            f"2. Secondary theme: {sentences[1]}",
            f"3. Notable mentions: {sentences[2]}",
            f"4. Overall sentiment: {sentences[3]}"
        ]
    else:
        structured_summary = ["The tweets mainly focus on:"] + [f"{i + 1}. {sent}" for i, sent in enumerate(sentences)]

    structured_summary.append("Key topics discussed:")
    structured_summary.extend([f"- {topic}" for topic in topics])

    return '\n'.join(structured_summary)

def process_chunk(chunk):
    chunk['clean_text'] = chunk['tweetText'].apply(clean_text)
    chunk['sentiment'] = chunk['clean_text'].apply(get_sentiment)
    chunk['processed_text'] = chunk['clean_text'].apply(preprocess_text)
    return chunk

def process_data(df):
    num_cores = 4  # Adjust based on your system
    chunks = np.array_split(df, num_cores)
    with Pool(num_cores) as pool:
        results = pool.map(process_chunk, chunks)
    df = pd.concat(results)

    df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce')
    df = df.dropna(subset=['createdAt'])
    df['date'] = df['createdAt'].dt.date
    df['hour'] = df['createdAt'].dt.hour

    summary = improved_summarization(df)

    return df, summary

def perform_eda(df):
    basic_stats = df.describe().to_dict()

    hashtags = [tag.strip() for tags in df['hashtags'].dropna() for tag in tags.split(',')]
    top_hashtags = Counter(hashtags).most_common(10)

    sentiment_counts = df['sentiment'].apply(
        lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral')).value_counts()

    most_active_users = df['handle'].value_counts().head(10)

    word_freq = Counter(" ".join(df['processed_text']).split())
    top_words = word_freq.most_common(20)

    return basic_stats, top_hashtags, sentiment_counts, most_active_users, top_words

def format_basic_stats(stats):
    formatted = ""
    for column, values in stats.items():
        formatted += f"<h4>{column}</h4>\n"
        formatted += f"  Count: {values['count']:.2f}<br>\n"
        formatted += f"  Mean: {values['mean']:.2f}<br>\n"
        formatted += f"  Standard Deviation: {values['std']:.2f}<br>\n"
        formatted += f"  Minimum: {values['min']:.2f}<br>\n"
        formatted += f"  25th Percentile: {values['25%']:.2f}<br>\n"
        formatted += f"  Median: {values['50%']:.2f}<br>\n"
        formatted += f"  75th Percentile: {values['75%']:.2f}<br>\n"
        formatted += f"  Maximum: {values['max']:.2f}<br>\n\n"
    return formatted

def format_top_hashtags(hashtags):
    return "\n".join([f"#{tag}: {count}" for tag, count in hashtags])