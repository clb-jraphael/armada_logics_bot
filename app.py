# app.py

import gradio as gr
from chatbot import answer_query

chat_history = []

def chat_interface(user_input):
    global chat_history
    if not user_input.strip():
        return "", chat_history
    
    answer, sources = answer_query(user_input)
    chat_history.append((user_input, answer))

    # Format sources for display
    if sources:
        source_texts = []
        for s in sources:
            line = f"📄 **{s['source']}**"
            if s["category"]:
                line += f" _(Category: {s['category']})_"
            if s["page"] is not None:
                line += f", Page {s['page']}"
            line += f"\n> {s['snippet']}"
            source_texts.append(line)
        answer += "\n\n---\n**Sources:**\n" + "\n\n".join(source_texts)

    return "", chat_history

with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Armada Logics Chatbot (Dagger One)")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask something...")
    clear = gr.Button("Clear")

    msg.submit(chat_interface, [msg], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(
        share=True,        #  Shows public gradio.live link
        inbrowser=True     #  Opens browser automatically
    )

