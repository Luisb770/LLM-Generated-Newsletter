import arxiv
import ollama
from rouge_score import rouge_scorer

# Caching mechanism
cache = {}

# Function to retrieve the latest statistics papers
def fetch_latest_papers():
    print("Fetching results from arXiv...")
    client = arxiv.Client()
    search = arxiv.Search(
        query="cat:stat.*",
        max_results=10,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = list(client.results(search))
    print(f"Fetched {len(papers)} results.")
    return papers

# Function to prompt Llama 3 for summarization
def summarize_abstract(abstract_text):
    print("Summarizing abstract...")
    try:
        response = ollama.chat(model="llama3", messages=[
            {
                'role': 'user',
                'content': f'Summarize the following abstract in one sentence: "{abstract_text}"'
            },
        ])
        if isinstance(response, dict) and 'message' in response and 'content' in response['message']:
            summary = response['message']['content']
            return summary.strip()
        else:
            return "Summary not available."
    except Exception as e:
        print(f"An error occurred while summarizing the text: {e}")
        return "Summary not available."

# Function to categorize text with Llama
def categorize_abstract_with_llama(summary):
    print("Categorizing summary...")
    categories = [
        "Bayesian Statistics", "Computational Statistics", "Biostatistics",
        "Statistics Methodology", "Unsupervised Learning", "Supervised Learning",
        "High-Dimensional Statistics", "Time Series Analysis", "Multivariate Analysis",
        "Experimental Design", "Nonparametric Statistics", "Econometrics",
        "Probability Theory", "Statistical Learning Theory", "Applied Statistics",
        "Environmental Statistics", "Financial Statistics", "Survey Statistics",
        "Spatial Statistics", "Stochastic Processes", "Data Mining", "Statistical Methodology",
        "Neural Networks", "Reinforcement Learning", "Ensemble Learning",
        "Inferential Statistics", "Descriptive Statistics", "Machine Learning",
        "Statistics Sampling", "Bioinformatics", "Statistical Decision Theory",
        "Spatial Statistics", "Casual Inference"
    ]
    try:
        response = ollama.chat(model="llama3", messages=[
            {
                'role': 'user',
                'content': f'Categorize the following summary into one of the given categories: "{summary}". Categories: {", ".join(categories)}.'
            },
        ])
        if isinstance(response, dict) and 'message' in response and 'content' in response['message']:
            category = response['message']['content'].strip()
            # Check if the response category is in the predefined list of categories
            if any(cat.lower() in category.lower() for cat in categories):
                return next(cat for cat in categories if cat.lower() in category.lower())
            else:
                print(f"Received uncategorized response: {category}")
                return "Uncategorized"
        else:
            return "Uncategorized"
    except Exception as e:
        print(f"An error occurred while categorizing the summary: {e}")
        return "Uncategorized"

# Function to get summary from cache
def get_summary_from_cache(abstract_text):
    if abstract_text in cache:
        return cache[abstract_text]
    summary = summarize_abstract(abstract_text)
    cache[abstract_text] = summary
    return summary

# Function to calculate ROUGE score
def calculate_rouge(reference, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return scores

# Function to create the newsletter
def create_newsletter(categorized):
    print("Creating the newsletter catered to statistics researchers and PhDs...")
    try:
        prompt = "Create a newsletter catered to statistics researchers and PhDs based on the following summaries and categories:\n"
        for category, summaries in categorized.items():
            if summaries:  # Only add non-empty categories
                prompt += f"{category}:\n"
                for summary, paper in summaries:
                    authors = ', '.join(author.name for author in paper.authors)
                    link = paper.entry_id
                    prompt += f"Title: {paper.title}\nAuthors: {authors}\nLink: {link}\nSummary: {summary}\n\n"
        prompt += "Ensure the newsletter is detailed and formatted for a professional audience.\n"

        # Add prompt for the bigger picture explanation
        prompt += "At the end of the newsletter, write a 4-sentence explanation on how the topics are related to each other and title it 'The Bigger Picture'.\n"

        # Add prompt for the joke
        prompt += "At the very end of the newsletter, create a joke based on the information in the newsletter and title it 'The Punchline'.\n"

        # Add the sign-off message
        prompt += "\n That is all for this issue.Stay tuned for our next issue and stay curious and keep crunching those numbers!\n\nBest regards,\nThe Probability Post Team"

        # Add the newsletter title and introduction
        prompt = "Title: The Probability Post\n\nHi Stat Fam,\n\nWelcome to the latest edition of The Probability Post, where we bring you cutting-edge research from the world of statistics. Let's dive into the exciting new papers that are shaping our field!\n\n" + prompt

        response = ollama.chat(model="llama3", messages=[
            {
                'role': 'user',
                'content': prompt
            },
        ])

        if isinstance(response, dict) and 'message' in response and 'content' in response['message']:
            newsletter = response['message']['content']
        else:
            newsletter = "Newsletter creation failed."

        return newsletter
    except Exception as e:
        print(f"An error occurred while creating the newsletter: {e}")
        return "Newsletter creation failed."

# Main function
def main():
    try:
        papers = fetch_latest_papers()
        summaries_and_categories = []
        for index, paper in enumerate(papers, start=1):
            print(f"Processing paper {index}/{len(papers)}: {paper.title}")
            summary = get_summary_from_cache(paper.summary)
            rouge_scores = calculate_rouge(paper.summary, summary)
            print(f"Title: {paper.title}")
            print(f"Authors: {', '.join(author.name for author in paper.authors)}")
            print(f"Summary: {summary}")
            print(f"Link: {paper.entry_id}")
            category = categorize_abstract_with_llama(summary)
            print(f"Category: {category}")
            print(f"ROUGE Scores: {rouge_scores}")
            summaries_and_categories.append((summary, category, paper))

        categorized = {category: [] for category in [
            "Bayesian Statistics", "Computational Statistics", "Biostatistics", 
            "Statistics Methodology", "Unsupervised Learning", "Supervised Learning", 
            "High-Dimensional Statistics", "Time Series Analysis", "Multivariate Analysis",
            "Experimental Design", "Nonparametric Statistics", "Econometrics",
            "Probability Theory", "Statistical Learning Theory", "Applied Statistics",
            "Environmental Statistics", "Financial Statistics", "Survey Statistics",
            "Spatial Statistics", "Stochastic Processes", "Data Mining", "Statistical Methodology",
            "Neural Networks", "Reinforcement Learning", "Ensemble Learning",
            "Inferential Statistics", "Descriptive Statistics", "Machine Learning",
            "Statistics Sampling", "Bioinformatics", "Statistical Decision Theory", "Casual Inference", "Uncategorized"
        ]}

        # Categorize each summary and paper
        for summary, category, paper in summaries_and_categories:
            if category not in categorized:
                categorized["Uncategorized"].append((summary, paper))
            else:
                categorized[category].append((summary, paper))
        
        print("All papers processed.")
        newsletter = create_newsletter(categorized)
        print(newsletter)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

