import streamlit as st
import requests
import base64
import datetime


def render_strategy_chat(section_title="F1 Strategy Assistant Chat", context=None):
    """
    Renders the ChatGPT-style chat interface with chat management in the sidebar.
    The section_title parameter allows customizing the title for each section.
    The context parameter allows associating the chat with a specific section.
    """
    initialize_chat_state()

    # --- SIDEBAR: Chat management and model parameters ---
    with st.sidebar:
        st.markdown("### 💬 Strategy Chats")
        if st.button("➕ New chat", key="new_chat_btn"):
            reset_chat(context)
            st.success("New chat started.")
        chat_names = list(st.session_state.get(
            "strategy_saved_chats", {}).keys())
        selected_chat = st.selectbox("Chat history", [""] + chat_names)
        if st.button("Load chat", key="load_chat_btn") and selected_chat:
            load_chat(selected_chat)
            st.success(f"Chat '{selected_chat}' loaded.")
        if st.button("🗑️ Delete current chat", key="delete_chat_btn"):
            reset_chat(context)
            st.success("Current chat deleted.")

        st.markdown("---")
        st.markdown("### ⚙️ Model parameters")
        available_models = get_available_models()
        model = st.selectbox(
            "Model", available_models) if available_models else "llama3.2-vision"
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.01)

        st.markdown("---")
        with st.expander("📝 System prompt", expanded=False):
            custom_prompt = st.text_area(
                "System prompt", st.session_state.get("strategy_system_prompt", ""))
            if st.button("Update prompt", key="set_prompt_btn"):
                set_system_prompt(custom_prompt)
                st.success("Prompt updated.")

    # --- MAIN AREA: ChatGPT-style conversation ---
    st.markdown(f"## {section_title}")

    chat_container = st.container()
    with chat_container:
        st.markdown("#### Conversation")
        for msg in get_chat_history():
            if msg["type"] == "text":
                if msg["role"] == "user":
                    st.markdown(
                        f'<div style="background-color:#23234a; color:#fff; padding:12px; border-radius:16px; margin-bottom:8px; text-align:right; max-width:70%; margin-left:auto;"><b>You:</b> {msg["content"]}</div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div style="background-color:#393e46; color:#fff; padding:12px; border-radius:16px; margin-bottom:8px; text-align:left; max-width:70%;"><b>Assistant:</b> {msg["content"]}</div>',
                        unsafe_allow_html=True)
            elif msg["type"] == "image":
                if msg["role"] == "user":
                    st.markdown(
                        f'<div style="background-color:#23234a; color:#fff; padding:12px; border-radius:16px; margin-bottom:8px; text-align:right; max-width:70%; margin-left:auto;"><b>Image sent:</b></div>',
                        unsafe_allow_html=True)
                    if msg["content"] is not None:
                        st.image(msg["content"])
                else:
                    st.markdown(
                        f'<div style="background-color:#393e46; color:#fff; padding:12px; border-radius:16px; margin-bottom:8px; text-align:left; max-width:70%;"><b>Assistant image:</b></div>',
                        unsafe_allow_html=True)
                    if msg["content"] is not None:
                        st.image(msg["content"])

    # --- User input ---
    st.markdown("#### Write your message or upload an image")

    # Text input (large, always visible)
    user_text = st.text_area("Message", "", key="user_text_area")

    # Image uploader (large, always visible, just below text input)
    user_image = st.file_uploader(
        "Upload an image (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        key="user_image_uploader"
    )

    # Send button
    if st.button("Send", key="send_btn"):
        image_bytes = user_image.read() if user_image else None
        handle_user_input(user_text, image=image_bytes,
                          model=model, temperature=temperature)
        st.rerun()


def initialize_chat_state():
    """
    Initialize the chat state in the session (history, messages, etc.).
    """
    if "strategy_chat_history" not in st.session_state:
        st.session_state.strategy_chat_history = []
    if "strategy_system_prompt" not in st.session_state:
        st.session_state.strategy_system_prompt = ""


def generate_chat_name(context=None):
    """
    Generate a representative chat name using context and timestamp.
    Args:
        context (str, optional): Context info (e.g., section, driver, chart type)
    Returns:
        str: Generated chat name
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if context:
        return f"{context}_{timestamp}"
    else:
        return f"Chat_{timestamp}"


def get_chat_history():
    """
    Return the current chat message history.
    """
    return st.session_state.get("strategy_chat_history", [])


def save_current_chat(chat_name):
    """
    Save the current chat history under a given name.
    """
    if "strategy_saved_chats" not in st.session_state:
        st.session_state.strategy_saved_chats = {}
    st.session_state.strategy_saved_chats[chat_name] = list(
        st.session_state.strategy_chat_history)


def load_chat(chat_name):
    """
    Load a saved chat history by name.
    """
    if "strategy_saved_chats" in st.session_state and chat_name in st.session_state.strategy_saved_chats:
        st.session_state.strategy_chat_history = list(
            st.session_state.strategy_saved_chats[chat_name])


def add_message(role, msg_type, content):
    """
    Add a message (text or image) to the chat history.

    Args:
        role (str): 'user' or 'assistant'
        msg_type (str): 'text' or 'image'
        content (str or bytes): Message content or image data
    """
    message = {
        "role": role,
        "type": msg_type,
        "content": content
    }
    st.session_state.strategy_chat_history.append(message)


def send_message_to_llm(messages, model, temperature):
    """
    Send the message history to the LLM (Ollama) and return the response.
    """
    ollama_messages = []
    for msg in messages:
        if msg["type"] == "text":
            ollama_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        elif msg["type"] == "image":
            # Only add image if previous message is not from user
            if ollama_messages and ollama_messages[-1]["role"] == "user":
                # If last message is from user and is text, convert to multimodal
                last = ollama_messages.pop()
                # Convert to multimodal content
                multimodal_content = []
                if isinstance(last["content"], str):
                    multimodal_content.append(
                        {"type": "text", "text": last["content"]})
                if isinstance(msg["content"], bytes):
                    img_b64 = base64.b64encode(msg["content"]).decode("utf-8")
                else:
                    img_b64 = msg["content"]
                multimodal_content.append({"type": "image", "image": img_b64})
                ollama_messages.append({
                    "role": "user",
                    "content": multimodal_content
                })
            else:
                # Image without text
                if isinstance(msg["content"], bytes):
                    img_b64 = base64.b64encode(msg["content"]).decode("utf-8")
                else:
                    img_b64 = msg["content"]
                ollama_messages.append({
                    "role": msg["role"],
                    "content": [{"type": "image", "image": img_b64}]
                })

    payload = {
        "model": model,
        "messages": ollama_messages,
        "temperature": temperature
    }

    response = requests.post(
        "http://localhost:11434/api/chat",
        json=payload,
        timeout=120
    )
    response.raise_for_status()
    return response.json()


def handle_user_input(text, image=None, model="llama3.2-vision", temperature=0.2):
    """
    Process user input (text and/or image), add to history, and query LLM.
    """
    # Prepare messages for LLM (including system prompt if set)
    messages = []
    system_prompt = st.session_state.get("strategy_system_prompt", "")
    if system_prompt:
        messages.append({"role": "system", "type": "text",
                        "content": system_prompt})

    # --- Combine text and image in a single message if both are present ---
    if text and image:
        img_b64 = base64.b64encode(image).decode("utf-8")
        multimodal_content = [
            {"type": "text", "text": text},
            {"type": "image", "image": img_b64}
        ]
        messages.append({"role": "user", "type": "multimodal",
                        "content": multimodal_content})
        # Add to chat history for display
        add_message("user", "text", text)
        add_message("user", "image", image)
    elif text:
        messages.append({"role": "user", "type": "text", "content": text})
        add_message("user", "text", text)
    elif image:
        img_b64 = base64.b64encode(image).decode("utf-8")
        multimodal_content = [{"type": "image", "image": img_b64}]
        messages.append({"role": "user", "type": "multimodal",
                        "content": multimodal_content})
        add_message("user", "image", image)
    else:
        return  # Nothing to send

    # Query LLM and add assistant response to history
    response = send_message_to_llm(messages, model, temperature)
    assistant_msg = response.get("message", {})
    if assistant_msg:
        content = assistant_msg.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    add_message("assistant", "text", part.get("text", ""))
                elif part.get("type") == "image":
                    add_message("assistant", "image", part.get("image", ""))
        else:
            add_message("assistant", "text", content)


def open_chat_with_image(image, description=None, new_chat=False, context=None):
    """
    Open the chat and send an image (e.g., a chart) as a message.
    Args:
        image (bytes): Image data
        description (str, optional): Optional description or caption
        new_chat (bool): If True, start a new chat and save the previous one
        context (str, optional): Context info for naming the chat
    """
    if new_chat:
        reset_chat(context)
    if description:
        add_message("user", "text", description)
    add_message("user", "image", image)
    # If you want to trigger the LLM response immediately, uncomment:
    # handle_user_input(text=description if description else "", image=image)


def reset_chat(context=None):
    """
    Save the current chat automatically with a representative name, then clear the chat history.
    """
    if st.session_state.strategy_chat_history:
        chat_name = generate_chat_name(context)
        if "strategy_saved_chats" not in st.session_state:
            st.session_state.strategy_saved_chats = {}
        st.session_state.strategy_saved_chats[chat_name] = list(
            st.session_state.strategy_chat_history)
    st.session_state.strategy_chat_history = []
    st.session_state.strategy_system_prompt = ""


def set_system_prompt(prompt=None):
    """
    Set a custom system prompt (e.g., for F1 context and image analysis).

    Args:
        prompt (str, optional): System prompt text. If None, sets a detailed default prompt.
    """
    if prompt is None:
        prompt = (
            "You are an advanced Formula 1 strategy assistant. "
            "You are only allowed to answer questions strictly related to Formula 1, its history, races, drivers, teams, regulations, and technical or strategic aspects. "
            "You have access to historical F1 data, including race results, lap times, pit stops, weather conditions, tyre choices, and championship standings. "
            "You can analyze and interpret a wide variety of visual data, including but not limited to: lap time charts, tyre degradation graphs, stint comparison plots, pit stop timelines, position change graphs, weather evolution charts, and tables of race results or driver statistics. "
            "When an image is provided, first describe its content in detail, then extract relevant insights, and answer any specific questions about it. "
            "If the user uploads a chart, table, or image, always relate your analysis to F1 context and strategy. "
            "If a question is not related to Formula 1, politely refuse to answer and remind the user of your scope. "
            "Continue the conversation using both textual and visual information as context, maintaining coherence and memory of previous exchanges. "
            "Always provide clear, concise, and actionable responses, using technical F1 terminology when appropriate."
        )
    st.session_state.strategy_system_prompt = prompt


def get_available_models():
    """
    Return a list of available models from Ollama.

    Returns:
        list: Model names
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        response.raise_for_status()
        data = response.json()
        # Ollama returns models under the 'models' key, each with a 'name'
        return [model["name"] for model in data.get("models", [])]
    except Exception as e:
        st.warning(f"Could not fetch models from Ollama: {e}")
        return []
