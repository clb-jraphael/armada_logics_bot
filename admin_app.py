# admin_app.py

import gradio as gr
from ingestion import prepare_document_chunks, qdrant

def add_knowledge(input_text):
    if not input_text.strip():
        return "⚠️ Please enter some content."
    chunks = prepare_document_chunks(input_text)
    if chunks:
        qdrant.add_documents(chunks)
        return f"✅ Added {len(chunks)} new chunk(s) to the knowledge base."
    else:
        return "📭 No new content to add. All chunks already present."

with gr.Blocks() as admin_ui:
    gr.Markdown("## 🔐 Admin Panel - Add to Knowledge Base")
    input_text = gr.Textbox(label="New Knowledge", lines=10, placeholder="Paste content here...")
    output_msg = gr.Markdown()
    add_btn = gr.Button("➕ Add to Knowledge Base")

    add_btn.click(fn=add_knowledge, inputs=input_text, outputs=output_msg)

if __name__ == "__main__":
    admin_ui.launch(
        share=True,        #  Enables public access link
        inbrowser=True     #  Auto-opens your default browser
    )

