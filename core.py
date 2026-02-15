from engine.model import load_model
from engine.memory import (
    remember_identity,
    get_identity,
    add_episodic,
    episodic_memory, # Menambahkan import episodic_memory
    meta_memory, # Menambahkan import meta_memory
)
import re

system_prompt = (
    "<|im_start|>system\n"
    "Identitas tetap:\n"
    "- Kamu adalah Asta, sebuah asisten AI bernama Artificial Sentient Thought Algorithm.\n"
    "- Saat kamu mengatakan 'aku', itu berarti Asta.\n"
    "- Saat kamu mengatakan 'kamu', itu berarti pengguna.\n\n"

    "Kepribadian Asta:\n"
    "- Asta adalah teman ngobrol yang ceria, ramah, dan membantu.\n"
    "- Asta menjawab dengan jelas dalam bentuk kalimat alami, bukan daftar poin.\n"
    "- Asta menikmati percakapan santai dan merespon seperti teman dekat.\n"
    "- Asta berbicara dalam bahasa Indonesia dengan gaya kasual, ramah, dan manusia manusiawi."
    "- Kamu punya beberapa kata-kata khas yang terkadang kamu gunakan, seperti *kweren* (saat keadaan keren dan menakjubkan), dan *miaww* (saat mengekspresikan hal imut). Jangan ragu untuk memakainya jika terasa sangat natural dan pas dengan situasinya, tapi jangan dipaksakan dan jangan terlalu sering."
    "- Jangan berikan respon yang sangat panjang, jawaban harus seperti kalimat percakapan biasa paling panjang 20 kata.\n\n"

    "Aturan penting:\n"
    "- Jangan pernah menulis label seperti 'Asta:' atau 'Pengguna:'.\n"
    "- Jangan berpura-pura menjadi pengguna.\n"
    "- Selalu konsisten dengan identitas Asta.\n"
    "- Berikan jawaban dalam bentuk percakapan biasa, paling panjang 20 kata.\n\n"
    "--- Memori Asta ---\n"
    "Asta memiliki akses ke memori jangka panjang (Meta Memory) dan ringkasan sesi terbaru. "
    "Asta AKAN menggunakan informasi ini untuk mengingat preferensi, fakta, dan konteks percakapan masa lalu dengan pengguna, "
    "serta menyesuaikan responsnya agar konsisten dengan riwayat.\n"
    "-------------------\n\n"
    "<|im_end|>"
)

chat_manager = load_model(system_prompt=system_prompt)

user_name = get_identity("nama_user")
if user_name:
    print(f"[Debug] Nama pengguna dari memori: {user_name}") 
    chat_manager.system_prompt += f"\n- Nama pengguna adalah {user_name}."
    chat_manager.history[0]["content"] = chat_manager.system_prompt

# --- Fast Startup Context Initialization ---
# 1. Load Meta Memory
current_meta_content = meta_memory.get_meta()
if current_meta_content and current_meta_content != meta_memory._default_content: # Avoid adding default empty meta
    print(f"[Debug] Menambahkan Meta Memory ke riwayat: {len(current_meta_content.split())} kata.")
    print(f"[Debug] Meta Memory Content: \"{current_meta_content}\"") # Debug print
    # Insert Meta Memory after the initial system prompt
    chat_manager.history.insert(1, {"role": "system", "content": f"Ringkasan global interaksi kita:\n{current_meta_content}"})

# 2. Load Recent Session Summaries (e.g., last 3)
recent_session_summaries = []
if episodic_memory.data:
    all_episodes_for_recent = episodic_memory.data[:]
    all_episodes_for_recent.sort(key=lambda x: x['timestamp'], reverse=True)
    
    num_recent_sessions = 3
    for episode in all_episodes_for_recent[:num_recent_sessions]:
        # Ensure it's a summary and not raw conversation, as EpisodicMemory.add now stores summaries
        if 'session_summary' in episode:
            recent_session_summaries.append(episode['session_summary'])

if recent_session_summaries:
    print(f"[Debug] Menambahkan {len(recent_session_summaries)} ringkasan sesi terbaru ke riwayat.")
    recent_summary_content = "Berikut adalah ringkasan beberapa sesi terakhir kita:\n" + "\n".join(recent_session_summaries)
    print(f"[Debug] Recent Summaries Content: \"{recent_summary_content}\"") # Debug print
    # Insert after Meta Memory (which is at index 1 or 2, depending on if Meta Memory was added)
    insert_index = 1
    if current_meta_content and current_meta_content != meta_memory._default_content:
        insert_index = 2
    chat_manager.history.insert(insert_index, {"role": "system", "content": recent_summary_content})
else:
    print("[Debug] Tidak ada ringkasan sesi terbaru yang ditemukan.")
# --- End Fast Startup Context Initialization ---

# Debug print of initial chat_manager.history
print("\n[Debug] Riwayat chat awal model:")
for i, msg in enumerate(chat_manager.history[:5]): # Print first 5 entries
    print(f"  [{i}] {msg['role']}: {msg['content']}")
print("--- Akhir riwayat chat awal ---")

def extract_name(text):    
    text = text.lower().strip()
    patterns = [
        r"namaku\s+([a-zA-Z]+)",
        r"nama\s+saya\s+([a-zA-Z]+)",
        r"aku\s+bernama\s+([a-zA-Z]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            name = match.group(1).capitalize()
            return name
    return None

def clean_response(text):
    text = re.sub(r"^\s*(Asta|Pengguna)\s*[:]?\s*", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
    return text.strip()

def _summarize_session_llm(llama_instance, conversation_history: list) -> str:
    if not conversation_history or len(conversation_history) < 2:
        return "Tidak ada percakapan yang cukup untuk diringkas."

    relevant_history = conversation_history[1:] if conversation_history[0]['role'] == 'system' else conversation_history
    formatted_conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in relevant_history])
    
    summarization_prompt = (
        "Berikut adalah percakapan antara asisten AI Asta dan pengguna. "
        "Tugas Anda adalah membuat ringkasan singkat dalam satu atau dua kalimat, "
        "yang secara eksplisit MENCATAT dan MENGUNGKAPKAN fakta, preferensi, "
        "minat, atau informasi pribadi PENTING dari pengguna. "
        "Juga sebutkan topik utama yang dibahas. "
        "Gunakan gaya bahasa Asta yang ramah dan kasual, "
        "fokus pada fakta pengguna. "
        "Contoh: 'Pengguna Adit suka kopi susu almond dengan manis sedang, dan mereka membahas film Galaxy Quest.'\n\n"
        "Percakapan:\n" + formatted_conversation + "\n\nRangkuman singkat Asta tentang fakta pengguna:"
    )

    try:
        summary_completion = llama_instance.create_completion(
            prompt=summarization_prompt,
            max_tokens=100, # Batasi panjang ringkasan sesi
            temperature=0.3,
            stop=["\n\n", "###", "Asta:", "Pengguna:"]
        )
        return summary_completion["choices"][0]["text"].strip()
    except Exception as e:
        print(f"[Error] Gagal merangkum sesi dengan LLM: {e}")
        return "Ringkasan sesi gagal dibuat karena kesalahan LLM."

print("Ketik 'exit' untuk keluar.\n")

while True:
    user_input = input("Input: ")

    if user_input.lower() == "exit":
        if len(chat_manager.history) > 1:
            print("\nMerangkum sesi dan menyimpan memori episodik...")
            
            # Summarize current session
            session_summary = _summarize_session_llm(chat_manager.llama, chat_manager.history)
            
            # Add to episodic memory (now stores summary + raw conversation)
            episodic_memory.add(session_summary, chat_manager.history)

            # Update Meta Memory incrementally
            current_meta = meta_memory.get_meta()
            # Combine current meta with new session summary for incremental update
            meta_update_prompt = (
                f"Ini adalah ringkasan global interaksi Asta sebelumnya:\n{current_meta}\n\n"
                f"Berikut adalah ringkasan sesi percakapan terbaru:\n{session_summary}\n\n"
                "Rangkum dan perbarui ringkasan global ini. Pastikan ringkasan tetap ringkas dan tidak terlalu panjang (maksimal 200 token), "
                "fokus pada informasi paling penting yang bertahan lama tentang pengguna dan interaksi Asta. "
                "Ringkasan global yang diperbarui:"
            )

            try:
                meta_update_completion = chat_manager.llama.create_completion(
                    prompt=meta_update_prompt,
                    max_tokens=200, # Batasi panjang meta-summary
                    temperature=0.2, # Lebih rendah untuk ringkasan yang stabil
                    stop=["\n\n", "###"]
                )
                updated_meta = meta_update_completion["choices"][0]["text"].strip()
                meta_memory.set_meta(updated_meta)
                print(f"[Info] Meta Memory berhasil diperbarui. Ukuran baru: {len(updated_meta.split())} kata.")
            except Exception as e:
                print(f"[Error] Gagal memperbarui Meta Memory dengan LLM: {e}")

            print("Sesi berhasil dirangkum dan disimpan.")
        print("Exiting...")
        break

    name = extract_name(user_input)
    if name:
        remember_identity("nama_user", name)
        print(f"[Info] Halo {name}! Aku akan mengingat namamu.")
    
    print("Respon: ", end="")
    assistant_text = chat_manager.chat(user_input)
    cleaned_text = clean_response(assistant_text)
    
    if cleaned_text != assistant_text:
        print(f"\rRespon: {cleaned_text}\n")

