

## Overview

This project involves creating a weekly newsletter titled "The Probability Post," which summarizes and categorizes the latest research papers in the field of statistics from arXiv. The newsletter aims to provide concise and informative summaries of research papers, categorized by their subfields, and is delivered via email. The project leverages the Llama 3 model for summarization and categorization, and includes humorous elements to engage the readers.

## Components

### 1. Fetching Latest Papers
The script retrieves the latest statistics papers from arXiv using the `arxiv` library. The search is limited to papers in the statistics category (`cat:stat.*`) and fetches up to 300 results, sorted by submission date.

### 2. Summarizing Abstracts
The project uses the Llama 3 model to summarize the abstracts of the fetched papers. The summarization is done using N-shot learning with example prompts. Four different prompts are used to generate diverse summaries, each focusing on different aspects of the abstract, such as novelty, methodology, significance, and challenges addressed.

### 3. Chain of Thought Prompting
For each generated summary, chain of thought prompting is applied to evaluate the advantages and disadvantages of the summary. This step helps in refining the summaries by highlighting their strengths and weaknesses.

### 4. Selecting the Best Summary
The script selects the best summary from the generated ones by querying the Llama 3 model. The model evaluates the summaries and their explanations, and determines which summary would be most suitable for the newsletter.

### 5. Categorizing Abstracts
Abstracts are categorized into predefined statistics subfields using the Llama 3 model. The model provides a category along with an explanation for the categorization.

### 6. Calculating ROUGE Scores
ROUGE scores are calculated for each summary to quantitatively evaluate their quality against the original abstracts. The `rouge_score` library is used for this purpose.

### 7. Creating the Newsletter
The newsletter is created in HTML format. It includes:
- A welcome message.
- Categorized sections with summaries and discussions.
- A spotlight section highlighting the most important paper.
- A humorous joke section.
The newsletter content is structured to engage and inform the readers effectively.

### 8. Sending the Newsletter
The newsletter is sent via email using the `smtplib` library. The email is composed with HTML content and sent to the recipients through an SMTP server.

## Running the Project

### Prerequisites
- Python 3.7 or later
- Required libraries: `arxiv`, `ollama`, `rouge_score`, `smtplib`, `email.mime.multipart`, `email.mime.text`

### Setup
1. Install the required libraries using pip:
   ```
   pip install arxiv ollama rouge_score
   ```
2. Configure the email settings with the appropriate SMTP server details and credentials.

### Execution
1. Run the main script:
   ```
   python main.py
   ```
2. The script will fetch the latest papers, generate summaries and categories, create the newsletter, and send it via email.

## Customization

### Modifying Prompts
You can customize the prompts used for summarization and chain of thought prompting by modifying the `prompts` and `example_prompts` variables in the script. This allows for tailoring the summaries to specific aspects of the papers.

### Adding Categories
To add or modify the categories for categorization, update the `categories` list in the `categorize_abstract_with_llama` function. Ensure that the new categories are included in the prompts and categorization logic.

### Email Settings
Configure the email settings such as `to_email`, `from_email`, `smtp_server`, `smtp_port`, `smtp_user`, and `smtp_password` in the `send_email` function. These settings are necessary for sending the newsletter via email.

## Error Handling
The script includes error handling mechanisms to manage issues during summarization, categorization, and email sending. Errors are logged to the console for debugging purposes.

## Conclusion
This project provides a comprehensive solution for generating and delivering a weekly statistics newsletter. By leveraging advanced language models and providing detailed summaries and categories, the newsletter aims to keep the statistics community informed and engaged with the latest research developments.
