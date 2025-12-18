# Optimized RAG Chatbot with Enhanced PDF Parsing and Mathematical Capabilities
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
import json
import re
import pandas as pd
import os
import pickle
from datetime import datetime
from collections import deque
import requests


class RAGChatbot:
    def __init__(self, pdf_path: str, csv_path: str, groq_api_key: str):
        self.pdf_path = pdf_path
        self.csv_path = csv_path
        self.groq_api_key = groq_api_key
        self.chunks = []
        self.chunk_metadata = []
        self.index = None
        self.embeddings_model = None

        # Groq API configuration
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        self.model_name = "llama-3.3-70b-versatile"

        # Complete chat history storage
        self.full_chat_history = []
        self.output_dir = "./"
        self.history_file = os.path.join(self.output_dir, "full_chat_history.pkl")

        # FAISS index for chat history
        self.chat_embeddings = []
        self.chat_index = None
        self.chat_embedding_file = os.path.join(self.output_dir, "chat_embeddings.pkl")

        # Conversation context tracking
        self.conversation_context = {
            'current_entities': deque(maxlen=10),
            'entity_attributes': {},
            'numerical_context': {}
        }

        # Employee data for quick lookup
        self.employee_data = None

        os.makedirs(self.output_dir, exist_ok=True)
        self._load_chat_history()
        self._setup()
        self._build_chat_history_index()

    def _load_chat_history(self):
        """Load complete chat history from file"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'rb') as f:
                    self.full_chat_history = pickle.load(f)
                print(f"Loaded {len(self.full_chat_history)} previous conversations")
            except Exception as e:
                print(f"Could not load chat history: {e}")
                self.full_chat_history = []
        else:
            self.full_chat_history = []

    def _save_chat_history(self):
        """Save complete chat history to file"""
        try:
            with open(self.history_file, 'wb') as f:
                pickle.dump(self.full_chat_history, f)
        except Exception as e:
            print(f"Could not save chat history: {e}")

    def _build_chat_history_index(self):
        """Build FAISS index from ALL chat history"""
        if len(self.full_chat_history) == 0:
            print("No chat history to index")
            return

        print(f"Building semantic index for {len(self.full_chat_history)} past conversations...")

        chat_texts = []
        for entry in self.full_chat_history:
            combined_text = f"Q: {entry['question']}\nA: {entry['answer']}"
            chat_texts.append(combined_text)

        self.chat_embeddings = self.embeddings_model.encode(chat_texts, show_progress_bar=True)

        dimension = self.chat_embeddings.shape[1]
        self.chat_index = faiss.IndexFlatL2(dimension)
        self.chat_index.add(np.array(self.chat_embeddings).astype('float32'))

        try:
            with open(self.chat_embedding_file, 'wb') as f:
                pickle.dump(self.chat_embeddings, f)
        except Exception as e:
            print(f"Could not save chat embeddings: {e}")

        print(f"Chat history index built successfully")

    def _search_chat_history(self, query: str, k: int = 10) -> List[Dict]:
        """Search through ALL past conversations"""
        if self.chat_index is None or len(self.full_chat_history) == 0:
            return []

        query_embedding = self.embeddings_model.encode([query])

        distances, indices = self.chat_index.search(
            np.array(query_embedding).astype('float32'),
            min(k, len(self.full_chat_history))
        )

        relevant_chats = []
        for idx, distance in zip(indices[0], distances[0]):
            if distance < 2.0:
                relevant_chats.append({
                    'chat': self.full_chat_history[idx],
                    'similarity_score': float(distance)
                })

        return relevant_chats

    def _extract_entities_from_text(self, text: str) -> List[str]:
        """Extract names and entities from text"""
        potential_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        return potential_names + emails

    def _detect_pronouns(self, query: str) -> bool:
        """Check if query contains pronouns"""
        pronouns = r'\b(he|she|him|her|his|they|them|their|theirs|it|its)\b'
        return bool(re.search(pronouns, query, re.IGNORECASE))

    def _resolve_pronouns(self, query: str) -> str:
        """Advanced pronoun resolution using context"""
        if not self._detect_pronouns(query):
            return query

        resolved_query = query

        if self.conversation_context['current_entities']:
            recent_entity = self.conversation_context['current_entities'][-1]

            pronoun_map = {
                r'\b(he|she)\b': recent_entity,
                r'\b(him|her)\b': recent_entity,
                r'\b(his|her|their)\b': f"{recent_entity}'s",
                r'\bthey\b': recent_entity,
                r'\bthem\b': recent_entity
            }

            for pattern, replacement in pronoun_map.items():
                resolved_query = re.sub(pattern, replacement, resolved_query, flags=re.IGNORECASE)

            print(f"[Pronoun Resolution] Resolved using: {recent_entity}")

        return resolved_query

    def _update_conversation_context(self, question: str, answer: str):
        """Track entities and context throughout conversation"""
        entities = self._extract_entities_from_text(question + " " + answer)

        for entity in entities:
            if entity not in self.conversation_context['current_entities']:
                self.conversation_context['current_entities'].append(entity)

            if entity in answer:
                self.conversation_context['entity_attributes'][entity] = {
                    'last_mentioned': datetime.now().isoformat(),
                    'context_snippet': answer[:200]
                }

        numbers = re.findall(r'\b\d+\.?\d*\b', answer)
        if numbers:
            self.conversation_context['numerical_context'] = {
                'last_calculation': answer,
                'numbers': numbers,
                'timestamp': datetime.now().isoformat()
            }

    def _setup(self):
        """Initialize embedding model and process documents"""
        print("Loading embedding model: all-MiniLM-L6-v2...")
        self.embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        print("Processing PDF with header-based chunking...")
        self._extract_text_with_headers()

        print("Processing CSV row-by-row...")
        self._extract_csv_data()

        self._build_index()

    def _extract_text_with_headers(self):
        """Extract text from PDF preserving section structure"""
        with open(self.pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"

        # Pattern to identify headers (numbered sections)
        header_pattern = r'^(\d+\.?\d*)\s+([A-Z\s]+)$'

        sections = []
        current_section = {'header': 'Introduction', 'content': ''}

        lines = full_text.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            header_match = re.match(header_pattern, line)

            if header_match:
                if current_section['content']:
                    sections.append(current_section)

                section_num = header_match.group(1)
                section_title = header_match.group(2).strip()
                current_section = {
                    'header': f"{section_num} {section_title}",
                    'content': ''
                }
            else:
                current_section['content'] += line + ' '

        if current_section['content']:
            sections.append(current_section)

        # Process Q&A patterns
        qa_pattern = r'Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)'

        for section in sections:
            content = section['content']
            header = section['header']

            qa_matches = re.findall(qa_pattern, content, re.DOTALL)

            for q, a in qa_matches:
                qa_chunk = f"[Section: {header}]\nQ: {q.strip()}\nA: {a.strip()}"
                self.chunks.append(qa_chunk)
                self.chunk_metadata.append({
                    'type': 'qa',
                    'section': header,
                    'source': 'pdf'
                })

            content = re.sub(qa_pattern, '', content, flags=re.DOTALL)

            if content.strip():
                sentences = re.split(r'(?<=[.!?])\s+', content)

                chunk_text = ""
                for sentence in sentences:
                    if len(chunk_text) + len(sentence) < 800:
                        chunk_text += sentence + " "
                    else:
                        if chunk_text.strip():
                            final_chunk = f"[Section: {header}]\n{chunk_text.strip()}"
                            self.chunks.append(final_chunk)
                            self.chunk_metadata.append({
                                'type': 'text',
                                'section': header,
                                'source': 'pdf'
                            })
                        chunk_text = sentence + " "

                if chunk_text.strip():
                    final_chunk = f"[Section: {header}]\n{chunk_text.strip()}"
                    self.chunks.append(final_chunk)
                    self.chunk_metadata.append({
                        'type': 'text',
                        'section': header,
                        'source': 'pdf'
                    })

        print(f"Extracted {len(self.chunks)} chunks from PDF with section headers")

    def _extract_csv_data(self):
        """Parse CSV row-by-row for precise querying"""
        try:
            df = pd.read_csv(self.csv_path)
            self.employee_data = df

            # Full table overview
            table_overview = "EMPLOYEE TABLE OVERVIEW:\n" + df.to_string(index=False)
            self.chunks.append(table_overview)
            self.chunk_metadata.append({
                'type': 'table_full',
                'source': 'csv',
                'row_count': len(df)
            })

            # Individual employee records
            for idx, row in df.iterrows():
                row_data = []
                for col in df.columns:
                    value = row[col]
                    if pd.notna(value):
                        row_data.append(f"{col}: {value}")

                row_chunk = " | ".join(row_data)
                self.chunks.append(row_chunk)
                self.chunk_metadata.append({
                    'type': 'employee_record',
                    'source': 'csv',
                    'row_index': idx,
                    'employee_name': row.get('Employee Name', 'Unknown')
                })

            print(f"Loaded {len(df)} employee records from CSV")

        except Exception as e:
            print(f"Error loading CSV: {e}")

    def _build_index(self):
        """Build FAISS index for document chunks"""
        print("Building FAISS index for documents...")
        embeddings = self.embeddings_model.encode(self.chunks, show_progress_bar=True)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        print(f"Index built with {self.index.ntotal} vectors")

    def _retrieve(self, query: str, k: int = 15) -> List[Tuple[str, Dict, float]]:
        """Retrieve relevant chunks from documents"""
        query_embedding = self.embeddings_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append((self.chunks[idx], self.chunk_metadata[idx], float(distance)))

        return results

    def _perform_calculations(self, query: str, context: str) -> str:
        """Extract and perform mathematical operations"""
        query_lower = query.lower()

        # Count queries
        if 'how many' in query_lower or 'count' in query_lower:
            if 'employee' in query_lower and self.employee_data is not None:
                # Department-specific count
                if 'operations' in query_lower:
                    count = len(
                        self.employee_data[self.employee_data['Department'].str.contains('Operations', na=False)])
                    return f"There are {count} employees in Operations."

                # Total count
                total = len(self.employee_data)
                return f"Total number of employees: {total}"

        # Average/sum calculations
        if 'average' in query_lower or 'mean' in query_lower:
            numbers = re.findall(r'\b\d+\.?\d*\b', context)
            if numbers:
                nums = [float(n) for n in numbers]
                avg = sum(nums) / len(nums)
                return f"Average: {avg:.2f}"

        return ""

    def _build_prompt(self, question: str, retrieved_data: List, relevant_past_chats: List[Dict],
                      calculation_result: str = "") -> str:
        """Build few-shot prompt with examples"""

        # Context from documents
        context_parts = []
        for chunk, metadata, score in retrieved_data[:10]:
            section_info = metadata.get('section', metadata.get('type', ''))
            context_parts.append(f"[{section_info}]\n{chunk}")

        context = "\n\n".join(context_parts)

        # Past conversation context
        past_context = ""
        if relevant_past_chats:
            past_context = "\n\nRELEVANT PAST CONVERSATIONS:\n"
            for chat_data in relevant_past_chats[:5]:
                chat = chat_data['chat']
                past_context += f"Q: {chat['question']}\nA: {chat['answer']}\n\n"

        # Calculation results
        calc_context = ""
        if calculation_result:
            calc_context = f"\n\nCALCULATION RESULT:\n{calculation_result}\n"

        # Few-shot examples
        few_shot_examples = """
EXAMPLES OF HOW TO ANSWER:

Example 1 - Counting:
Q: How many employees work in the Operations Team?
A: There are 34 employees in the Operations Team.

Example 2 - Specific Information:
Q: What is Zubaer's email?
A: Md. Zubaer Hossain's email addresses are zubaer.acmeai@gmail.com and project@acmeai.tech.

Example 3 - Policy Question:
Q: How many days of earned leave do employees get?
A: According to the HR Policy, employees are entitled to 16 annual leave days after one year of service, distributed evenly across four quarters.

Example 4 - Various answer scope in one user input:
Q: I want to know about leave policy.
if the company has multiple policies or categories, then answer would be: 
A: "There are multiple policies are present in this company. such as, 
{list of all policies fetched from document}
Which policy do you want to know about?"

Example 4 - Calculation:
Q: If someone takes 5 days of unpaid leave in a month with 30 days and earns 30,000 taka, what is the deduction?
A: Let me calculate:
- One day salary = 30,000 / 30 = 1,000 taka
- Total deduction for 5 days = 1,000 × 5 = 5,000 taka

Example 5 - Following up with pronouns:
Q: Who is the Operations Manager?
A: Md. Zubaer Hossain is the Operations Manager.
Q: What is his email?
A: His email addresses are zubaer.acmeai@gmail.com and project@acmeai.tech.
"""

        prompt = f"""You are an intelligent assistant helping employees at Acme AI Ltd. You have access to company policies and employee data. You excel at mathematical calculations, counting, and understanding context from previous conversations.

{few_shot_examples}

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. For mathematical questions, show your calculation steps clearly
3. For counting questions, provide exact numbers
4. When users refer to someone mentioned earlier (using "he", "she", "they", "him", "her"), understand they're referring to the person from the previous conversation
5. If information is not in the context, clearly state that
6. Be precise and factual
7. For policy questions, cite the relevant section

CONTEXT FROM DOCUMENTS:
{context}
{past_context}
{calc_context}

QUESTION: {question}

ANSWER:"""

        return prompt

    def ask(self, question: str) -> str:
        """Main query method"""
        if question.lower() in ['reset', 'clear history', 'reset chat']:
            self.full_chat_history = []
            self.chat_embeddings = []
            self.chat_index = None
            self.conversation_context = {
                'current_entities': deque(maxlen=10),
                'entity_attributes': {},
                'numerical_context': {}
            }
            self._save_chat_history()
            return "Complete chat history has been reset."

        # Resolve pronouns
        resolved_question = self._resolve_pronouns(question)

        # Search past conversations
        relevant_past_chats = self._search_chat_history(resolved_question, k=10)

        # Retrieve document chunks
        retrieved_data = self._retrieve(resolved_question, k=15)

        # Perform calculations
        context_text = "\n".join([chunk for chunk, _, _ in retrieved_data])
        calculation_result = self._perform_calculations(resolved_question, context_text)

        # Build prompt
        prompt = self._build_prompt(resolved_question, retrieved_data, relevant_past_chats, calculation_result)

        # Call Groq API
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 500,
            "temperature": 0.2,
            "top_p": 0.9
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=90)
            response.raise_for_status()
            result = response.json()
            answer = result["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            print(f"Error calling Groq API: {e}")
            answer = "I apologize, but I'm having trouble generating a response right now. Please try again."

        # Update context
        self._update_conversation_context(question, answer)

        # Store in history
        chat_entry = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'resolved_question': resolved_question,
            'answer': answer,
            'used_past_context': len(relevant_past_chats) > 0,
            'entities_mentioned': list(self.conversation_context['current_entities'])
        }

        self.full_chat_history.append(chat_entry)

        # Update chat history index
        new_text = f"Q: {question}\nA: {answer}"
        new_embedding = self.embeddings_model.encode([new_text])

        if self.chat_index is None:
            dimension = new_embedding.shape[1]
            self.chat_index = faiss.IndexFlatL2(dimension)
            self.chat_embeddings = new_embedding
        else:
            self.chat_embeddings = np.vstack([self.chat_embeddings, new_embedding])

        self.chat_index.add(np.array(new_embedding).astype('float32'))

        # Save history
        self._save_chat_history()

        return answer

    def get_stats(self) -> Dict:
        """Get chatbot statistics"""
        qa_chunks = sum(1 for c in self.chunk_metadata if c['type'] == 'qa')
        text_chunks = sum(1 for c in self.chunk_metadata if c['type'] == 'text')
        table_full = sum(1 for c in self.chunk_metadata if c['type'] == 'table_full')
        employee_records = sum(1 for c in self.chunk_metadata if c['type'] == 'employee_record')

        conversations_with_context = sum(
            1 for c in self.full_chat_history if c.get('used_past_context', False)
        )

        return {
            'total_chunks': len(self.chunks),
            'qa_chunks': qa_chunks,
            'text_chunks': text_chunks,
            'table_full': table_full,
            'employee_records': employee_records,
            'total_conversations': len(self.full_chat_history),
            'conversations_with_past_context': conversations_with_context,
            'tracked_entities': len(self.conversation_context['current_entities']),
            'recent_entities': list(self.conversation_context['current_entities'])
        }

    def get_full_history(self) -> List[Dict]:
        """Get complete chat history"""
        return self.full_chat_history

    def display_stats(self):
        """Display statistics"""
        stats = self.get_stats()

        print(f"\n{'=' * 80}")
        print("RAG CHATBOT STATISTICS")
        print(f"{'=' * 80}")
        print(f"Document Processing:")
        print(f"  - Total chunks: {stats['total_chunks']}")
        print(f"  - Q&A chunks: {stats['qa_chunks']}")
        print(f"  - Text chunks (with headers): {stats['text_chunks']}")
        print(f"  - Full table view: {stats['table_full']}")
        print(f"  - Employee records: {stats['employee_records']}")

        print(f"\nModel Configuration:")
        print(f"  - Embedding Model: sentence-transformers/all-MiniLM-L6-v2")
        print(f"  - LLM: Groq {self.model_name}")
        print(f"  - Vector Search: FAISS-CPU")

        print(f"\nConversation History:")
        print(f"  - Total conversations: {stats['total_conversations']}")
        print(f"  - Using past context: {stats['conversations_with_past_context']}")
        print(f"  - Entities tracked: {stats['tracked_entities']}")
        if stats['recent_entities']:
            print(f"  - Recent entities: {', '.join(stats['recent_entities'][-5:])}")

        print(f"\nFiles:")
        print(f"  - PDF: {os.path.basename(self.pdf_path)}")
        print(f"  - CSV: {os.path.basename(self.csv_path)}")
        print(f"  - History: {os.path.basename(self.history_file)}")
        print(f"{'=' * 80}\n")


# Main execution
if __name__ == "__main__":
    PDF_PATH = "./data/Policies.pdf"
    CSV_PATH = "./data/employee_table.csv"
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set")

    print("\n" + "=" * 80)
    print("Enhanced RAG Chatbot")
    print("Header-based PDF | Row-level CSV | Complete History | Pronoun Resolution")
    print("=" * 80 + "\n")

    bot = RAGChatbot(PDF_PATH, CSV_PATH, GROQ_API_KEY)
    bot.display_stats()

    print("Chatbot ready!")
    print("Commands: 'exit', 'stats', 'history', 'reset'")
    print("=" * 80 + "\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break

        if user_input.lower() == 'stats':
            stats = bot.get_stats()
            print("\nStatistics:")
            print(json.dumps(stats, indent=2))
            continue

        if user_input.lower() == 'history':
            history = bot.get_full_history()
            print(f"\nShowing last 5 conversations:")
            for entry in history[-5:]:
                print(f"\nQ: {entry['question']}")
                print(f"A: {entry['answer'][:150]}...")
            continue

        if not user_input:
            continue

        try:
            answer = bot.ask(user_input)
            print(f"\nBot: {answer}\n")
        except Exception as e:
            print(f"Error: {e}\n")