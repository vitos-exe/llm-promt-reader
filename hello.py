from litellm import completion
import gradio as gr
from retriever import Retriever
import glob

MODEL_NAME = 'ollama/llama3.2'
API_BASE = 'http://localhost:11434'
SYSTEM_MESSAGE = {
    'role': 'system', 
    'content': 
    '''
    You are given some context and also a question. Answer the question using given context.
    '''
}

class RAGQuestionAnsweringBot:
    def __init__(self,retriver: Retriever) -> None:
        self.retriever = retriver

    def answer_question(self, question: str):
        context = self.retriever.get_docs(question)

        messages = [
            SYSTEM_MESSAGE,
            {'role': 'user', 'content': f'Context:\n{context}\nQuestion: {question}'}
        ]

        answer = completion(
            model=MODEL_NAME,
            api_base=API_BASE,
            messages=messages
        )

        return answer.json()['choices'][0]['message']['content']


if __name__ == "__main__":
    docs = []
    for path in glob.glob('data/*txt'):
        with open(path) as f:
            docs.append(f.read())

    retriever = Retriever(docs)
    bot = RAGQuestionAnsweringBot(retriever)
    query = 'What is real name of Doctor Migos?'

    answer = bot.answer_question(query)
    print(answer)

    # demo = gr.Interface(bot.answer_question, inputs='text', outputs='text')
    # demo.launch();

