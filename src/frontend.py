import uuid
import gradio as gr
import requests

# Конфігурація за замовчуванням
DEFAULT_API_URL = "http://localhost:8000"


def turn(history, user, bot):
    """Додає пару повідомлень у форматі messages (Gradio >= 5/6 не приймає старі списки [[user, bot]])."""
    return (history or []) + [{"role": "user", "content": user}, {"role": "assistant", "content": bot}]

def check_health(base_url):
    """Перевірка стану бекенду (/health)"""
    try:
        resp = requests.get(f"{base_url}/health", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            return f"✅ Сервер доступний | Статус: {data.get('status')} | Граф: {'Підключено' if data.get('graph_available') else 'Недоступний'} | Чанків у базі: {data.get('chunks', 0)}"
        return f"⚠️ Сервер відповів з кодом {resp.status_code}"
    except Exception as e:
        return f"❌ Помилка підключення до сервера: {e}"

def ask_rag(
    message, 
    history, 
    api_key, 
    use_graph, 
    current_user_id, 
    base_url
):
    """Виклик ендпоінту /ask та обробка відповіді"""
    if not api_key:
        return history, current_user_id, "⚠️ Будь ласка, введіть API-ключ у налаштуваннях.", "", ""
    
    headers = {"X-API-Key": api_key.strip()}
    payload = {
        "prompt": message,
        "use_graph": use_graph
    }
    
    if current_user_id:
        payload["user_id"] = current_user_id

    try:
        resp = requests.post(f"{base_url}/ask", json=payload, headers=headers, timeout=60)
        
        if resp.status_code == 401:
            return turn(history, message, "❌ Помилка 401: Невірний або відсутній API-ключ."), current_user_id, "Помилка автентифікації", "", ""
        elif resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After", "60")
            return turn(history, message, f"⏳ Помилка 429: Перевищено ліміт запитів. Спробуйте через {retry_after} сек."), current_user_id, "Ліміт перевищено", "", ""
        elif resp.status_code != 200:
            err_detail = resp.json().get("detail", resp.text)
            return turn(history, message, f"❌ Помилка {resp.status_code}: {err_detail}"), current_user_id, f"Помилка {resp.status_code}", "", ""

        data = resp.json()
        new_user_id = data.get("user_id")
        answer = data.get("answer")
        sources = data.get("sources", [])
        graph_info = data.get("graph", {})
        usage = data.get("usage", {})
        latency = data.get("latency_ms", 0)
        role = data.get("role", "unknown")

        # Форматування джерел для відображення
        sources_md = "### 📚 Використані джерела:\n"
        if sources:
            for s in sources:
                sources_md += f"- **ID:** `{s['chunk_id']}` | **Назва:** {s.get('title', 'N/A')} (Розділ: {s.get('section', 'N/A')}) | * Sensitivity: {s['sensitivity']} *\n"
        else:
            sources_md += "*Джерела не використовувалися або не знайдені.*\n"

        # Форматування інформації про граф
        graph_md = f"### 🕸️ Граф знань:\n"
        graph_md += f"- **Використано:** {'Так' if graph_info.get('used') else 'Ні'}\n"
        graph_md += f"- **Режим:** {graph_info.get('mode', 'N/A')}\n"
        if graph_info.get('entities'):
            graph_md += f"- **Сутності:** {', '.join(graph_info.get('entities'))}\n"
        if graph_info.get('facts'):
            graph_md += f"\n**Факти підграфа:**\n> {graph_info.get('facts')}\n"

        # Статистика
        stats_md = f"**Роль:** `{role}` | **Час генерації:** `{latency} мс` | **Токени (In/Out):** `{usage.get('tokens_in', 0)} / {usage.get('tokens_out', 0)}`"

        updated_history = turn(history, message, answer)
        return updated_history, new_user_id, sources_md, graph_md, stats_md

    except requests.exceptions.RequestException as e:
        return turn(history, message, f"❌ Мережева помилка під час запиту до API: {e}"), current_user_id, "Помилка зв'язку", "", ""

def fetch_chunk_detail(source_id, api_key, base_url):
    """Отримання повного тексту чанку за GET /sources/{id}"""
    if not api_key:
        return "⚠️ Введіть API-ключ для перегляду вмісту джерела."
    if not source_id:
        return "⚠️ Вкажіть ID джерела."

    headers = {"X-API-Key": api_key.strip()}
    try:
        resp = requests.get(f"{base_url}/sources/{source_id.strip()}", headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            out = f"### Деталі джерела: `{data.get('chunk_id')}`\n\n"
            out += f"**Документ:** {data.get('doc_id')} | **Назва:** {data.get('title')} | **Розділ:** {data.get('section')}\n"
            out += f"**Дата:** {data.get('date')} | **Тип:** {data.get('type')} | **Sensitivity:** `{data.get('sensitivity')}`\n\n"
            out += f"---\n**Повний текст фрагмента:**\n\n{data.get('text')}"
            return out
        elif resp.status_code == 403:
            return "❌ Помилка 403: У вашої ролі немає прав доступу до цього джерела (обмеження sensitivity)."
        elif resp.status_code == 404:
            return "❌ Помилка 404: Джерело з таким ID не знайдено."
        else:
            return f"❌ Помилка {resp.status_code}: {resp.text}"
    except Exception as e:
        return f"❌ Помилка запиту: {e}"

# Створення UI
with gr.Blocks(title="Domain Knowledge Hub: RAG Chat") as demo:
    gr.Markdown("# 🤖 Domain Knowledge Hub: RAG-чат")
    gr.Markdown("Інтерфейс взаємодії з корпоративним RAG-чатом (BM25 + Dense + Reranker + Neo4j Graph). Роль визначається автоматично за API-ключом на бекенді.")

    user_id_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=3):
            api_url_input = gr.Textbox(value=DEFAULT_API_URL, label="API Base URL", placeholder="http://localhost:8000")
        with gr.Column(scale=3):
            api_key_input = gr.Textbox(value="", type="password", label="X-API-Key (ключ доступу)", placeholder="Введіть API_KEY_PUBLIC або API_KEY_ANALYST")
        with gr.Column(scale=2):
            use_graph_checkbox = gr.Checkbox(value=True, label="Використовувати граф знань (use_graph)")
        with gr.Column(scale=2):
            health_btn = gr.Button("Перевірити статус сервера 🔌")

    health_status_md = gr.Markdown("Статус сервера: натисніть кнопку перевірки.")
    health_btn.click(fn=check_health, inputs=[api_url_input], outputs=[health_status_md])

    with gr.Row():
        with gr.Column(scale=7):
            try:
                chatbot = gr.Chatbot(label="Діалог з RAG", height=450, type="messages")
            except TypeError:                      # Gradio 6: параметра type немає, формат завжди messages
                chatbot = gr.Chatbot(label="Діалог з RAG", height=450)
            msg_input = gr.Textbox(label="Ваше питання", placeholder="Введіть запит до бази знань...", lines=2)
            
            with gr.Row():
                submit_btn = gr.Button("Надіслати 🚀", variant="primary")
                clear_btn = gr.Button("Очистити історію чату 🗑️")

            stats_output = gr.Markdown("**Статистика:** неактивно")

        with gr.Column(scale=5):
            with gr.Tabs():
                with gr.TabItem("📚 Джерела та Граф"):
                    sources_output = gr.Markdown("Тут з'являться джерела останньої відповіді.")
                    graph_output = gr.Markdown("Тут з'явиться інформація про використання графа.")
                
                with gr.TabItem("🔍 Перегляд чанку за ID"):
                    gr.Markdown("Введіть `chunk_id` з блока джерел, щоб отримати повний текст та перевірити права доступу.")
                    chunk_id_input = gr.Textbox(label="Chunk ID", placeholder="напр. doc_001_chunk_3")
                    fetch_chunk_btn = gr.Button("Завантажити текст чанку")
                    chunk_detail_output = gr.Markdown("Результат відобразиться тут.")
                    
                    fetch_chunk_btn.click(
                        fn=fetch_chunk_detail,
                        inputs=[chunk_id_input, api_key_input, api_url_input],
                        outputs=[chunk_detail_output]
                    )

    # Обробники подій чату
    submit_event = msg_input.submit(
        fn=ask_rag,
        inputs=[msg_input, chatbot, api_key_input, use_graph_checkbox, user_id_state, api_url_input],
        outputs=[chatbot, user_id_state, sources_output, graph_output, stats_output]
    )
    # Очищуємо поле введення після сабміту
    submit_event.then(lambda: "", outputs=[msg_input])

    submit_btn_event = submit_btn.click(
        fn=ask_rag,
        inputs=[msg_input, chatbot, api_key_input, use_graph_checkbox, user_id_state, api_url_input],
        outputs=[chatbot, user_id_state, sources_output, graph_output, stats_output]
    )
    submit_btn_event.then(lambda: "", outputs=[msg_input])

    clear_btn.click(lambda: ([], None, "**Статистика:** очищено", "Тут з'являться джерела останньої відповіді.", "Тут з'явиться інформація про використання графа."), 
                    outputs=[chatbot, user_id_state, stats_output, sources_output, graph_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)