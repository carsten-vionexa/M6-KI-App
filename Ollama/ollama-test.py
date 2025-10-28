import ollama
import PyPDF2       # Library for reading PDF files

#%% Summarization

prompt = (
    "Summarize the following text: "
    "Artificial intelligence (AI) has rapidly transformed various industries over the past decade. "
    "From healthcare and finance to transportation and entertainment, AI technologies are enabling automation, "
    "improving decision-making, and creating new opportunities for innovation. As organizations continue to adopt "
    "AI-driven solutions, it becomes increasingly important to consider the ethical implications and ensure responsible use. "
    "This includes addressing concerns such as data privacy, algorithmic bias, and the potential impact on employment. "
    "By fostering transparency and collaboration among stakeholders, society can maximize the benefits of AI while mitigating its risks."
)

response = ollama.generate(model="llama3.1", prompt=prompt)
print(response['response'])