"""Translation module using Google Gemini API."""

import os
import json
from typing import Optional, List, Dict
import google.generativeai as genai
from config import GEMINI_MODEL


class Translator:
    """Handles translation using Google Gemini API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize translator with Gemini API.

        Args:
            api_key: Google Gemini API key. If None, uses GEMINI_API_KEY environment variable.
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. "
                "Please set it or pass api_key parameter."
            )

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)

    def translate_batch(self, texts: List[str], target_language: str = "English", batch_size: int = 80, context: str = None) -> Dict[str, str]:
        """
        Translate multiple Korean texts to English and return as pairs.
        Splits large batches into smaller chunks to avoid API issues.

        Args:
            texts: List of Korean texts to translate
            target_language: Target language (default: English)
            batch_size: Maximum texts per API call (default: 50)
            context: Optional context information to improve translations

        Returns:
            Dictionary mapping original Korean text -> translated text
        """
        if not texts or all(not t.strip() for t in texts):
            return {}
        
        # Split into chunks and translate each chunk
        all_translations = {}
        unique_texts = [t.strip() for t in texts if t.strip()]
        unique_texts = list(dict.fromkeys(unique_texts))  # Remove duplicates while preserving order
        
        for i in range(0, len(unique_texts), batch_size):
            chunk = unique_texts[i:i+batch_size]
            chunk_num = i // batch_size + 1
            print(f"[Translation] Processing chunk {chunk_num}: {len(chunk)} texts")
            # Pass previously translated texts as context (skip first chunk)
            prev_translations = all_translations if i > 0 else {}
            chunk_result = self._translate_chunk(chunk, target_language, prev_translations, context)
            all_translations.update(chunk_result)
        
        return all_translations
    
    def _translate_chunk(self, texts: List[str], target_language: str = "English", prev_translations: Dict[str, str] = None, user_context: str = None) -> Dict[str, str]:
        """
        Translate a smaller chunk of texts (internal method).
        
        Args:
            texts: List of Korean texts to translate
            target_language: Target language (default: English)
            prev_translations: Previously translated texts for maintaining consistency
            user_context: Optional user-provided context to improve translations

        Returns:
            Dictionary mapping original Korean text -> translated text
        """
        if not texts or all(not t.strip() for t in texts):
            return {}
        
        if prev_translations is None:
            prev_translations = {}

        try:
            # Create simple list format for the prompt
            texts_list = [t.strip() for t in texts if t.strip()]
            texts_for_prompt = "\n".join(texts_list)
            
            # Build context section if we have previous translations
            context_section = ""
            if prev_translations:
                context_section = "\nReference translations (use for consistency with tone and style):\n"
                for korean, english in list(prev_translations.items())[:10]:  # Show last 10 for context
                    context_section += f'  "{korean}" -> "{english}"\n'
            
            # Add user context if provided
            user_context_section = ""
            if user_context:
                user_context_section = f"\nAdditional context for better translations:\n{user_context}\n"
            
            prompt = f"""You are translating dialogue and text from a Korean manhwa (webtoon). Provide natural, conversational translations that fit the comic narrative and maintain the original tone and emotion.{context_section}{user_context_section}

Translate these Korean texts to {target_language}.
Return ONLY a valid JSON object with the original Korean text as keys and English translations as values.
Format: {{"korean text": "english translation", ...}}

Korean texts to translate:
{texts_for_prompt}"""

            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            #JSON parsing
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"[Translation] JSON parse failed: {e.msg}")
                # Try to extract JSON object between first { and last }
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    response_text = response_text[start_idx:end_idx+1]
                    try:
                        result = json.loads(response_text)
                    except:
                        print(f"[Translation] Could not parse JSON in chunk")
                        return {}
                else:
                    return {}
            
            # Ensure result is a dict
            if not isinstance(result, dict):
                return {}
            
            return result
        except Exception as e:
            print(f"Translation Error: {str(e)}")
            return {}


