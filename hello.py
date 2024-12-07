from litellm import completion
import gradio as gr
from retriever import Retriever


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
    def __init__(self, docs: list[str]) -> None:
        self.retriever = Retriever(docs)

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
    bot = RAGQuestionAnsweringBot([])
    query = 'How are you doing?'
    answer = bot.answer_question(query)
    print(answer)

    # demo = gr.Interface(fn=answer_question, inputs='text', outputs='text')
    # demo.launch();

