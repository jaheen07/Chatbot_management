import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import re
import requests
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class Message:
    role: str
    content: str


class ConversationMemory:
    def __init__(self, max_history: int = 10):
        self.messages: deque = deque(maxlen=max_history)
        self.entities: Dict[str, str] = {}

    def add(self, role: str, content: str):
        self.messages.append(Message(role, content))
        if role == "user":
            self._extract_entities(content)

    def _extract_entities(self, text: str):
        names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for name in names:
            self.entities[name.lower()] = name

    def resolve_reference(self, text: str) -> str:
        pronouns = {
            'he': 'male', 'she': 'female', 'him': 'male',
            'her': 'female', 'his': 'male', 'they': 'neutral'
        }

        for pronoun, gender in pronouns.items():
            if re.search(rf'\b{pronoun}\b', text, re.IGNORECASE):
                recent_entities = list(self.entities.values())
                if recent_entities:
                    text = re.sub(
                        rf'\b{pronoun}\b',
                        recent_entities[-1],
                        text,
                        flags=re.IGNORECASE
                    )
        return text

    def get_context(self) -> str:
        if not self.messages:
            return ""

        context = []
        for msg in list(self.messages)[-6:]:
            prefix = "User" if msg.role == "user" else "Assistant"
            context.append(f"{prefix}: {msg.content}")
        return "\n".join(context)


class DocumentProcessor:
    def __init__(self, pdf_path: str, csv_path: str):
        self.chunks = []
        self.metadata = []
        self.df = None

        self._process_pdf(pdf_path)
        self._process_csv(csv_path)

    def _process_pdf(self, path: str):
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join(page.extract_text() for page in reader.pages)

        sections = re.split(r'\n(\d+\.?\d*\s+[A-Z\s]+)\n', text)

        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                header = sections[i].strip()
                content = sections[i + 1].strip()

                sentences = re.split(r'(?<=[.!?])\s+', content)
                chunk = ""

                for sentence in sentences:
                    if len(chunk) + len(sentence) < 600:
                        chunk += sentence + " "
                    else:
                        if chunk:
                            self.chunks.append(f"[{header}]\n{chunk.strip()}")
                            self.metadata.append({'type': 'policy', 'section': header})
                        chunk = sentence + " "

                if chunk:
                    self.chunks.append(f"[{header}]\n{chunk.strip()}")
                    self.metadata.append({'type': 'policy', 'section': header})

    def _process_csv(self, path: str):
        self.df = pd.read_csv(path)

        for _, row in self.df.iterrows():
            record = " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
            self.chunks.append(record)
            self.metadata.append({
                'type': 'employee',
                'name': row.get('Employee Name', 'Unknown')
            })


class Calculator:
    @staticmethod
    def extract_numbers(text: str) -> List[float]:
        return [float(n) for n in re.findall(r'\b\d+\.?\d*\b', text)]

    @staticmethod
    def calculate(query: str, context: str, df: pd.DataFrame = None) -> str:
        query_lower = query.lower()

        # Count queries
        if any(word in query_lower for word in ['how many', 'count', 'total']):
            if df is not None and 'employee' in query_lower:
                if 'operations' in query_lower:
                    count = len(df[df['Department'].str.contains('Operations', na=False)])
                    return f"Count: {count}"
                return f"Total employees: {len(df)}"

        # Arithmetic operations
        numbers = Calculator.extract_numbers(query + " " + context)

        if 'average' in query_lower or 'mean' in query_lower:
            if numbers:
                avg = np.ceil(sum(numbers) / len(numbers))
                return f"Average: {avg}"

        if 'sum' in query_lower or 'total' in query_lower:
            if numbers:
                return f"Sum: {np.ceil(sum(numbers))}"

        # Salary deduction calculation
        if 'deduction' in query_lower or 'unpaid' in query_lower:
            if len(numbers) >= 3:
                days = numbers[0]
                total_days = numbers[1]
                salary = numbers[2]
                per_day = salary / total_days
                deduction = np.ceil(per_day * days)
                return f"Daily rate: {np.ceil(per_day)} | Deduction for {int(days)} days: {deduction}"

        return ""


class RAGChatbot:
    def __init__(self, pdf_path: str, csv_path: str, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

        print("Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        print("Processing documents...")
        self.docs = DocumentProcessor(pdf_path, csv_path)

        print("Building search index...")
        embeddings = self.embedder.encode(self.docs.chunks, show_progress_bar=False)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))

        self.memory = ConversationMemory()
        self.calculator = Calculator()

        print(f"Ready! Indexed {len(self.docs.chunks)} chunks\n")

    def _retrieve(self, query: str, k: int = 8) -> List[Tuple[str, Dict]]:
        query_emb = self.embedder.encode([query])
        distances, indices = self.index.search(query_emb.astype('float32'), k)

        return [(self.docs.chunks[i], self.docs.metadata[i])
                for i in indices[0] if distances[0][list(indices[0]).index(i)] < 1.5]

    def _build_prompt(self, query: str, context: str, calc_result: str, history: str) -> str:
        base = f"""Answer based on the provided information.

CONTEXT:
{context}"""

        if calc_result:
            base += f"\n\nCALCULATION:\n{calc_result}"

        if history:
            base += f"\n\nRECENT CONVERSATION:\n{history}"

        base += f"\n\nQUESTION: {query}\n\nProvide a direct, specific answer. If calculating, show your work."

        return base

    def _call_llm(self, prompt: str) -> str:
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400,
            "temperature": 0.1
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def ask(self, question: str) -> str:
        resolved_query = self.memory.resolve_reference(question)

        retrieved = self._retrieve(resolved_query, k=8)
        context = "\n\n".join(chunk for chunk, _ in retrieved)

        calc_result = self.calculator.calculate(
            resolved_query,
            context,
            self.docs.df
        )

        history = self.memory.get_context()

        prompt = self._build_prompt(resolved_query, context, calc_result, history)

        answer = self._call_llm(prompt)

        self.memory.add("user", question)
        self.memory.add("assistant", answer)

        return answer


if __name__ == "__main__":
    PDF_PATH = "./data/Policies.pdf"
    CSV_PATH = "./data/employee_table.csv"
    API_KEY = ""

    bot = RAGChatbot(PDF_PATH, CSV_PATH, API_KEY)

    print("Commands: 'exit' to quit, 'clear' to reset memory\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ['exit', 'quit']:
            break

        if user_input.lower() == 'clear':
            bot.memory = ConversationMemory()
            print("Memory cleared.\n")
            continue

        try:
            answer = bot.ask(user_input)
            print(f"\nBot: {answer}\n")
        except Exception as e:
            print(f"Error: {e}\n")
