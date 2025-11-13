from typing import Dict, Any, Optional
import json

def _call_ollama_understand(text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Use Ollama to understand and analyze text content."""
    try:
        # Format the prompt with context if provided
        if context:
            prompt = (f"Please analyze this {context.get('content_type', 'document')}. "
                     f"Consider its context: {json.dumps(context)}\n\n"
                     f"Content: {text}\n\n"
                     "Please provide a structured analysis including:\n"
                     "1. A brief summary\n"
                     "2. Key topics/themes\n"
                     "3. Important points\n"
                     "4. Document structure/organization\n"
                     "5. Any notable patterns or insights\n"
                     "Respond in JSON format with these sections.")
        else:
            prompt = (f"Please analyze this text:\n\n{text}\n\n"
                     "Provide a structured analysis including:\n"
                     "1. A brief summary\n"
                     "2. Key topics/themes\n"
                     "3. Important points\n"
                     "4. Document structure/organization\n"
                     "5. Any notable patterns or insights\n"
                     "Respond in JSON format with these sections.")

        # Use the existing Ollama integration
        from server import _call_ollama_generate
        
        response = _call_ollama_generate({
            "prompt": prompt,
            "model": "mistral:latest",  # or any other suitable model
            "stream": False,
        })
        
        # Parse the response
        try:
            if isinstance(response, dict):
                analysis = response.get('response', '')
            else:
                analysis = str(response)
                
            # Try to parse as JSON first
            try:
                analysis_json = json.loads(analysis)
                return {
                    "type": "structured",
                    "analysis": analysis_json
                }
            except json.JSONDecodeError:
                # If not JSON, return as unstructured text
                return {
                    "type": "unstructured",
                    "analysis": analysis
                }
        except Exception as e:
            return {
                "type": "error",
                "error": f"Failed to parse analysis: {str(e)}",
                "raw_response": str(response)
            }
            
    except Exception as e:
        return {
            "type": "error",
            "error": f"AI analysis failed: {str(e)}"
        }