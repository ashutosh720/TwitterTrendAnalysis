@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            df, improved_summary = process_data(df)
            basic_stats, top_hashtags, sentiment_counts, most_active_users, top_words = perform_eda(df)
            visualizations = generate_visualizations(df, sentiment_counts, top_hashtags, most_active_users, top_words)
            batch_summaries, overall_summary = generate_summaries(df)

            response = {
                'basic_stats': format_basic_stats(basic_stats),
                'top_hashtags': format_top_hashtags(top_hashtags),
                'sentiment_distribution': sentiment_counts.to_dict(),
                'most_active_users': most_active_users.to_dict(),
                'top_words': dict(top_words),
                'visualizations': visualizations,
                'improved_summary': improved_summary,
                'batch_summaries': batch_summaries,
                'overall_summary': overall_summary
            }

            return jsonify(response)
        else:
            return jsonify({'error': 'Invalid file format'})
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(debug=True)