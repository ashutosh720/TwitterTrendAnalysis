def generate_visualizations(df, sentiment_counts, top_hashtags, most_active_users, top_words):
    images = []

    # Sentiment Distribution
    plt.figure(figsize=(16, 12))
    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, textprops={'fontsize': 14})
    plt.title('Sentiment Distribution', fontsize=24)
    plt.axis('equal')
    plt.tight_layout(pad=3.0)
    images.append(get_image())
    plt.close()

    # Top 10 Hashtags
    plt.figure(figsize=(20, 12))
    sns.barplot(x=[tag for tag, count in top_hashtags], y=[count for tag, count in top_hashtags])
    plt.title('Top 10 Hashtags', fontsize=24)
    plt.xlabel('Hashtags', fontsize=18)
    plt.ylabel('Count', fontsize=18)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout(pad=3.0)
    images.append(get_image())
    plt.close()

    # Most Active Users
    plt.figure(figsize=(20, 12))
    most_active_users.plot(kind='bar')
    plt.title('Most Active Users', fontsize=24)
    plt.xlabel('User Handle', fontsize=18)
    plt.ylabel('Number of Tweets', fontsize=18)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout(pad=3.0)
    images.append(get_image())
    plt.close()

    # Top 10 Most Frequent Words
    plt.figure(figsize=(20, 12))
    sns.barplot(x=[word for word, count in top_words[:10]], y=[count for word, count in top_words[:10]])
    plt.title('Top 10 Most Frequent Words', fontsize=24)
    plt.xlabel('Words', fontsize=18)
    plt.ylabel('Count', fontsize=18)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout(pad=3.0)
    images.append(get_image())
    plt.close()

    return images

def get_image():
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()