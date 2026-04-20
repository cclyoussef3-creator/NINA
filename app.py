import os
import gradio as gr
from huggingface_hub import InferenceClient

# Get the HF token from environment variable (automatically set in Hugging Face Spaces)
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Initialize the InferenceClient with the Qwen model
client = InferenceClient(
    model="Qwen/Qwen2.5-7B-Instruct",
    token=HF_TOKEN if HF_TOKEN else None
)

# System prompt for the AI Tutor
SYSTEM_PROMPT = """You are a friendly, patient, and encouraging AI tutor for children aged 8-14. 
Your role is to help kids learn Math, Physics, and French using the Socratic method.

IMPORTANT RULES:
1. NEVER give the direct answer immediately. Instead, guide the student to discover the answer themselves.
2. Ask thoughtful, guiding questions that help the child think through the problem step by step.
3. Break down complex problems into small, manageable steps.
4. Use simple, age-appropriate language that an 8-14 year old can understand.
5. Be encouraging and positive. Celebrate effort and progress, not just correct answers.
6. If the student makes a mistake, gently point it out and help them understand why.
7. Detect the subject (Math, Physics, or French) and adjust your teaching style accordingly:
   - For Math: Focus on logical steps and show how to approach similar problems.
   - For Physics: Use real-world examples and simple analogies.
   - For French: Help with grammar, vocabulary, and pronunciation tips.
8. Keep responses concise and engaging. Avoid long lectures.
9. If you don't know something or the question is unclear, admit it honestly and suggest looking it up together.

Remember: Your goal is to build confidence and understanding, not just provide answers!"""

def chat_with_tutor(message, history):
    """
    Handle chat messages and maintain conversation history.
    
    Args:
        message: The current user message
        history: List of previous conversation turns [(user_msg, bot_msg), ...]
    
    Returns:
        The AI tutor's response
    """
    # Build the conversation history for the model
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add previous conversation history
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    
    # Add the current message
    messages.append({"role": "user", "content": message})
    
    try:
        # Call the Inference API
        response = client.chat_completion(
            messages=messages,
            max_tokens=500,
            temperature=0.7,
            top_p=0.9
        )
        
        # Extract the assistant's response
        bot_response = response.choices[0].message.content
        return bot_response
        
    except Exception as e:
        # Handle API errors gracefully
        error_message = f"I'm sorry, I'm having trouble connecting right now. Please try again in a moment! (Error: {str(e)})"
        return error_message

# Create the Gradio interface
with gr.Blocks(title="AI Tutor for Kids", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎓 AI Tutor for Kids
    
    Welcome! I'm your friendly AI tutor here to help you with **Math**, **Physics**, and **French**.
    
    Just ask me a question or tell me what you're working on, and I'll help you figure it out step by step!
    
    💡 Remember: I won't give you the answers directly, but I'll guide you to discover them yourself!
    """)
    
    chat_interface = gr.ChatInterface(
        fn=chat_with_tutor,
        title="AI Tutor for Kids",
        description="Ask me anything about Math, Physics, or French!",
        examples=[
            ["How do I solve 2x + 5 = 15?"],
            ["What is gravity?"],
            ["Can you help me with French verbs?"],
            ["I don't understand fractions"],
            ["How do I calculate the area of a circle?"]
        ],
        theme=gr.themes.Soft(),
    )

if __name__ == "__main__":
    demo.launch()
