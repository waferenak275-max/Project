from pathlib import Path
from .memory_system import SemanticMemory, EpisodicMemory, MetaMemory # Menambahkan MetaMemory

MEMORY_DIR = Path("memory")
MEMORY_DIR.mkdir(exist_ok=True)

semantic_memory = SemanticMemory(MEMORY_DIR)
episodic_memory = EpisodicMemory(MEMORY_DIR)
meta_memory = MetaMemory(MEMORY_DIR) # Inisialisasi MetaMemory

def remember_identity(key: str, value: any):
    semantic_memory.add_fact(key, value)

def get_identity(key: str):
    return semantic_memory.get_fact(key)

def get_all_identities():
    return semantic_memory.get_all_facts()

def add_episodic(conversation: list):
    episodic_memory.add(conversation)

def search_episodic(query: str, top_k: int = 3):
    return episodic_memory.search(query, top_k)
