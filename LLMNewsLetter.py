import arxiv
import ollama
from rouge_score import rouge_scorer
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

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

# Function to prompt Llama 3 for summarization using N-shot learning
def summarize_abstract(abstract_text, prompt):
    print(f"Generating summary with prompt: {prompt}...")
    try:
        examples = [
            {
                "abstract": "We introduce a rigorous mathematical framework for Granger causality in extremes, designed to identify causal links from extreme events in time series...",
                "summary": "This paper introduces a mathematical framework for Granger causality in extremes, designed to identify causal relationships between extreme events in time series, offering advantages over traditional methods and demonstrating effectiveness in financial and extreme weather applications.",
                "category": "Extreme Value Theory"
            },
            {
                "abstract": "Due to the high dimensionality or multimodality that is common in modern astronomy, sampling Bayesian posteriors can be challenging...",
                "summary": "This paper describes a new, efficient C-language code called Nii-C that uses automatic parallel tempering and parallelization to improve sampling of complex probability distributions in astronomy and other fields, addressing challenges in high-dimensional or multimodal data analysis.",
                "category": "Computational Statistics"
            },
            {
                "abstract": "For a sequence of  n  random variables taking values 0 or 1, the hot hand statistic of streak length  k  counts what fraction of the streaks of length  k , that is,  k  consecutive variables taking the value 1, among the  n  variables are followed by another 1...",
                "summary": "The paper discusses a statistical measure called the 'hot hand statistic' that examines patterns in binary sequences, highlighting potential bias in estimating probabilities and proposing a new approach to calculate its expected value for single-event streaks.",
                "category": "Time Series Analysis"
            }
        ]
        example_prompts = "\n".join([
            f"Abstract: {ex['abstract']}\nSummary: {ex['summary']}\nCategory: {ex['category']}"
            for ex in examples
        ])

        response = ollama.chat(model="llama3", messages=[
            {
                'role': 'user',
                'content': f"{prompt}\n\nHere are some examples of summarizing and categorizing abstracts:\n{example_prompts}\n\nNow, summarize the following abstract in one sentence and provide the category: \"{abstract_text}\"."
            },
        ])
        if isinstance(response, dict) and 'message' in response and 'content' in response['message']:
            summary_category = response['message']['content'].strip()
            summary, category = summary_category.split("Category:")
            return summary.strip(), category.strip()
        return "Summary not available.", "Uncategorized"
    except Exception as e:
        print(f"An error occurred while summarizing the text: {e}")
        return "Summary not available.", "Uncategorized"

# Function to perform chain of thought prompting on each summary
def chain_of_thought_prompting(summary, index):
    print(f"Performing chain of thought prompting for summary {index}...")
    try:
        response = ollama.chat(model="llama3", messages=[
            {
                'role': 'user',
                'content': f'Explain the advantages and disadvantages of this summary: "{summary}"'
            }
        ])
        if isinstance(response, dict) and 'message' in response and 'content' in response['message']:
            explanations = response['message']['content'].strip()
            return f"{summary}\n\nChain of Thought Prompting for summary {index}:\n{explanations}"
        else:
            return f"{summary}\n\nChain of thought response not available."
    except Exception as e:
        print(f"An error occurred while performing chain of thought prompting: {e}")
        return f"{summary}\n\nChain of thought response not available."

# Function to select the best summary
def select_best_summary(summaries, explanations):
    print("Selecting the best summary...")
    try:
        response = ollama.chat(model="llama3", messages=[
            {
                'role': 'user',
                'content': f'Here are four summaries with their explanations:\n1. "{summaries[0]}"\nExplanation: {explanations[0]}\n2. "{summaries[1]}"\nExplanation: {explanations[1]}\n3. "{summaries[2]}"\nExplanation: {explanations[2]}\n4. "{summaries[3]}"\nExplanation: {explanations[3]}\nWhich summary would you use for a newsletter and why?'
            }
        ])
        if isinstance(response, dict) and 'message' in response and 'content' in response['message']:
            best_summary_response = response['message']['content'].strip()
            return best_summary_response
        else:
            return "Selection response not available."
    except Exception as e:
        print(f"An error occurred while selecting the best summary: {e}")
        return "Selection response not available."

# Function to categorize text with Llama, including chain of thought reasoning
def categorize_abstract_with_llama(abstract_text):
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
        "Casual Inference", "Uncategorized"
    ]
    try:
        response = ollama.chat(model="llama3", messages=[
            {
                'role': 'user',
                'content': f'Categorize the following abstract into one of the given categories and explain why: "{abstract_text}". Categories: {", ".join(categories)}.'
            }
        ])
        if isinstance(response, dict) and 'message' in response and 'content' in response['message']:
            category_response = response['message']['content'].strip()
            if "Category:" in category_response:
                category, explanation = category_response.split("Explanation:")
                category = category.replace("Category:", "").strip()
                explanation = explanation.strip()
                if category in categories:
                    return category, explanation
            print(f"Received uncategorized response: {category_response}")
            return "Uncategorized", "Categorization explanation not available."
        else:
            return "Uncategorized", "Categorization explanation not available."
    except Exception as e:
        print(f"An error occurred while categorizing the summary: {e}")
        return "Uncategorized", "Categorization explanation not available."

# Function to calculate ROUGE score
def calculate_rouge(reference, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return scores

# Function to create the newsletter
def create_newsletter(categorized):
    print("Creating the newsletter catered to statistics researchers and PhDs...")
    try:
        # Add prompt for jokes with N-shot prompting
        joke_examples = [
            "Why did the statistician bring a ladder to the bar? Because they heard the drinks were on the house!",
            "Why don’t statisticians play hide-and-seek? Because good luck hiding from someone who always finds the mean.",
            "How do statisticians keep warm in winter? By gathering more samples!"
        ]
        joke_prompt = "\n".join(joke_examples)
        
        joke_response = ollama.chat(model="llama3", messages=[
            {
                'role': 'user',
                'content': f"Create a joke based on the information in the newsletter using these examples for inspiration:\n{joke_prompt}"
            }
        ])
        
        joke_section = joke_response['message']['content'].strip() if 'message' in joke_response and 'content' in joke_response['message'] else "Joke not available."

        # Add prompt for The Spotlight section
        spotlight_prompt = "Out of all the papers listed, which one is the most important and interesting, and why? Write a paragraph explaining your choice."
        spotlight_response = ollama.chat(model="llama3", messages=[{'role': 'user', 'content': spotlight_prompt}])
        spotlight_paragraph = spotlight_response['message']['content'] if 'message' in spotlight_response and 'content' in spotlight_response['message'] else "Spotlight explanation not available."

        # Ensure the Spotlight paragraph is fully captured
        spotlight_paragraph = spotlight_paragraph.strip().replace("\n", " ")

        # Construct the newsletter content using HTML
        newsletter_content = """
        <html>
        <body>
            <h1>The Probability Post</h1>
            <p>Hi Stat Fam,</p>
            <p>Welcome to the latest edition of The Probability Post, where we bring you cutting-edge research from the world of statistics. Let's dive into the exciting new papers that are shaping our field!</p>
        """

        for category, summaries in categorized.items():
            if summaries:  # Only add non-empty categories
                # Generate discussion for each category
                category_summaries_text = "\n".join([f"{summary}" for summary, _ in summaries])
                discussion_prompt = f"Write a short paragraph discussing the following papers in the {category} category and how they relate to each other:\n{category_summaries_text}"
                discussion_response = ollama.chat(model="llama3", messages=[{'role': 'user', 'content': discussion_prompt}])
                category_discussion = discussion_response['message']['content'] if 'message' in discussion_response and 'content' in discussion_response['message'] else "Discussion not available."

                # Add category discussion to the newsletter
                newsletter_content += f"<h2>{category}</h2><p>{category_discussion}</p>"

                for summary, paper in summaries:
                    authors = ', '.join(author.name for author in paper.authors)
                    link = paper.entry_id
                    # Add categorization explanation as a separate query
                    categorization_explanation_prompt = f"Explain why the paper '{paper.title}' is categorized under '{category}'."
                    categorization_explanation_response = ollama.chat(model="llama3", messages=[{'role': 'user', 'content': categorization_explanation_prompt}])
                    categorization_explanation = categorization_explanation_response['message']['content'] if 'message' in categorization_explanation_response and 'content' in categorization_explanation_response['message'] else "Explanation not available."

                    newsletter_content += f"""
                        <h3>{paper.title}</h3>
                        <p><strong>Authors:</strong> {authors}</p>
                        <p><strong>Link:</strong> <a href="{link}">{link}</a></p>
                        <p><strong>Summary:</strong> {summary}</p>
                        <p><strong>Categorization Explanation:</strong> {categorization_explanation}</p>
                    """

        # Add Spotlight and Punchline sections before the sign-off message
        newsletter_content += f"""
            <h2>The Spotlight</h2>
            <p>{spotlight_paragraph}</p>
            <h2>The Punchline</h2>
            <p>{joke_section}</p>
            <p>That is all for this issue. Stay tuned for our next issue and stay curious and keep crunching those numbers!</p>
            <p>Best regards,<br>The Probability Post Team</p>
        </body>
        </html>
        """

        return newsletter_content
    except Exception as e:
        print(f"An error occurred while creating the newsletter: {e}")
        return "Newsletter creation failed."

# Function to send email via SMTP
def send_email(subject, body, to_email, from_email, smtp_server, smtp_port, smtp_user, smtp_password):
    print("Sending email...")
    try:
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'html'))

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(f"An error occurred while sending the email: {e}")

# Main function
def main():
    try:
        papers = fetch_latest_papers()
        summaries_and_categories = []
        prompts = [
            "Summarize the abstract focusing on its novelty and applications.",
            "Summarize the abstract emphasizing the methodology and key findings.",
            "Summarize the abstract highlighting its significance and potential impact.",
            "Summarize the abstract detailing the challenges addressed and solutions provided."
        ]
        for index, paper in enumerate(papers, start=1):
            print(f"Processing paper {index}/{len(papers)}: {paper.title}")
            print(f"Title: {paper.title}\nAuthors: {', '.join(author.name for author in paper.authors)}\nLink: {paper.entry_id}")
            summaries = []
            explanations = []
            for i in range(4):
                summary, _ = summarize_abstract(paper.summary, prompts[i])
                summaries.append(summary)
                explanation = chain_of_thought_prompting(summary, i+1)
                explanations.append(explanation)
                print(f"Summary {i+1} Prompt: {prompts[i]}")
                print(f"Summary {i+1} Generated: {summary}")
                print(f"Explanation {i+1}: {explanation}")

            best_summary_decision = select_best_summary(summaries, explanations)
            best_summary = summaries[0]  # Placeholder for selected summary based on decision
            for summary in summaries:
                if best_summary in summary:
                    best_summary = summary
                    break

            category, _ = categorize_abstract_with_llama(paper.summary)  # Only get the category, explanation generated separately
            print(f"Best Summary Decision: {best_summary_decision}")
            print(f"Best Summary: {best_summary}")
            print(f"Category: {category}")

            # Calculate and print ROUGE score for each summary
            rouge_scores = [calculate_rouge(paper.summary, summary) for summary in summaries]
            for i, score in enumerate(rouge_scores):
                print(f"ROUGE score for summary {i+1}: {score}")

            summaries_and_categories.append((best_summary, category, paper))

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

        # Track categorized papers to ensure each paper is only categorized once
        categorized_papers = set()

        # Categorize each summary and paper
        for summary, category, paper in summaries_and_categories:
            if paper.entry_id not in categorized_papers:
                if category not in categorized:
                    categorized["Uncategorized"].append((summary, paper))
                else:
                    categorized[category].append((summary, paper))
                categorized_papers.add(paper.entry_id)
        
        print("All papers processed.")
        newsletter = create_newsletter(categorized)
        print(newsletter)

        # Send the newsletter via email
        send_email(
            subject="The Probability Post",
            body=newsletter,
            to_email="",
            from_email="",
            smtp_server="",
            smtp_port=587,
            smtp_user="",
            smtp_password=""
        )

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
