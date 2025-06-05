import os
import time
import io
import base64
import json
import soundfile as sf
import librosa
import requests
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator

from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# --- Environment Setup ---
load_dotenv()

# Use environment variables for credentials for better security
# Ensure these are set in your .env file or environment
AZURE_PHI4_API_KEY = os.getenv("PHI4_API_KEY", "Cfjqwi4QrkAqdf5dwKOabIxAtEGh0Rgxi5LEM0EuwtfuNSj6MNtNJQQJ99BEACHYHv6XJ3w3AAAAACOGQbLu") # Fallback to your hardcoded key if not in .env
AZURE_PHI4_ENDPOINT = os.getenv("PHI4_ENDPOINT", "https://dm-ai-agents-openai-service.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview") # Fallback
# The model name deployed at your Azure endpoint for Phi-4 multimodal.
# This might be part of the endpoint URL or a parameter in the API call.
# For Azure OpenAI compatible endpoints, usually it's specified in the payload or the LLM client.
# Here, your payload has "model": "Phi-4-multimodal-instruct", so we'll use that.
AZURE_PHI4_MODEL_NAME = "Phi-4-multimodal-instruct"


# --- Define Pydantic Models for Structured Output ---
class TranscriptEntry(BaseModel):
    speaker: str = Field(description="Label for the speaker (e.g., 'Doctor', 'Patient', 'Agent', 'Customer')")
    text: str = Field(description="The spoken text by the speaker.")
    emotion: Optional[str] = Field(None, description="The emotion related to the spoken text, if identifiable.")
    # timestamp_start: Optional[float] = Field(None, description="Start time of the segment in seconds.") # Add if your model can provide this
    # timestamp_end: Optional[float] = Field(None, description="End time of the segment in seconds.")   # Add if your model can provide this

class Transcription(BaseModel):
    transcript: List[TranscriptEntry] = Field(description="A list of transcript entries.")

class CallMetadata(BaseModel):
    caller_name: Optional[str] = Field(None, description="The identified name of the person who initiated the call.")
    receiver_name: Optional[str] = Field(None, description="The identified name of the person who received the call.")
    call_subject: Optional[str] = Field(None, description="A brief subject or main topic of the call.")
    key_entities: Optional[List[str]] = Field(default_factory=list, description="List of key people, organizations, or locations mentioned.")
    overall_sentiment: Optional[str] = Field(None, description="Overall sentiment of the call (e.g., Positive, Negative, Neutral).")
    action_items: Optional[List[str]] = Field(default_factory=list, description="List of action items discussed during the call.")


# --- LangGraph State Definition ---
class GraphState(TypedDict):
    audio_path: str
    base64_audio: Optional[str]
    transcription_prompt: str
    transcript_json_str: Optional[str]
    parsed_transcript: Optional[Transcription]
    metadata_extraction_prompt: Optional[str]
    extracted_metadata: Optional[CallMetadata]
    final_result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    current_task_description: Optional[str]


# --- Utility Functions & LLM Initialization ---
def get_azure_phi4_llm(max_tokens=1024):
    """Initializes the ChatOpenAI client for Azure Phi-4 endpoint."""
    return ChatOpenAI(
        openai_api_key=AZURE_PHI4_API_KEY,
        openai_api_base=AZURE_PHI4_ENDPOINT.rsplit("/chat/completions", 1)[0], # Base URL
        model_name=AZURE_PHI4_MODEL_NAME, # This tells the Azure endpoint which model to use
        temperature=0.1, # Low temperature for factual tasks
        max_tokens=max_tokens,
        default_headers={ # Important for some Azure OpenAI setups
            "api-key": AZURE_PHI4_API_KEY
        }
    )

# --- LangGraph Nodes ---

def load_and_encode_audio(state: GraphState) -> GraphState:
    """Loads an audio file, processes it, and returns its base64 encoded representation."""
    print("\n--- Node: Load and Encode Audio ---")
    state["current_task_description"] = f"Loading and encoding audio from: {state['audio_path']}"
    try:
        audio, sr = librosa.load(state["audio_path"], sr=16000, mono=True)
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio, 16000, format='WAV', subtype='PCM_16')
        wav_buffer.seek(0)
        audio_bytes = wav_buffer.read()
        if not audio_bytes:
            return {**state, "error_message": "Audio data is empty after processing."}
        
        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
        print(f"Audio encoded to Base64 (first 100 chars): {base64_audio[:100]}...")
        return {**state, "base64_audio": base64_audio, "error_message": None}
    except Exception as e:
        error_msg = f"Error during audio preprocessing: {str(e)}"
        print(error_msg)
        return {**state, "error_message": error_msg}

def transcribe_audio_phi4(state: GraphState) -> GraphState:
    """Transcribes audio using the Azure Phi-4 multimodal endpoint."""
    print("\n--- Node: Transcribe Audio ---")
    state["current_task_description"] = "Transcribing audio using Phi-4 on Azure."
    if state.get("error_message") or not state.get("base64_audio"):
        return state # Skip if previous error or no audio

    llm = get_azure_phi4_llm(max_tokens=1500) # More tokens for transcription

    messages = [
        SystemMessage(content=(
            "You are an expert transcriptionist. Your ONLY task is to transcribe the provided phone call audio with high accuracy. "
            "Output ONLY valid JSON in the specified format, no explanations, no markdown, no extra text, no preamble, no postamble. "
            "The JSON must conform to the Transcription schema. "
            "If you cannot transcribe, return an empty transcript array in JSON."
        )),
        HumanMessage(content=[
            {"type": "text", "text": state["transcription_prompt"]},
            {
                "type": "image_url", # Phi-4 multimodal uses image_url for various data types like audio
                "image_url": {
                    "url": f"data:audio/wav;base64,{state['base64_audio']}"
                }
            }
        ])
    ]
    
    try:
        print("Sending transcription request to Phi-4 endpoint...")
        start_time = time.time()
        response = llm.invoke(messages)
        print(f"Phi-4 transcription call completed in {time.time() - start_time:.2f} seconds.")
        
        content = response.content
        print(f"Raw transcription response content: {content}")
        
        # Attempt to extract JSON robustly
        json_str = None
        try:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                # Validate if it's actual JSON before setting
                json.loads(json_str) # This will raise an error if not valid JSON
            else:
                raise ValueError("Could not find valid JSON delimiters '{' and '}' in the response.")
        except Exception as extraction_e:
            error_msg = f"Could not extract or validate JSON from transcription response: {extraction_e}. Raw content was: {content}"
            print(error_msg)
            return {**state, "error_message": error_msg, "transcript_json_str": content} # Store raw content if extraction fails

        print(f"Extracted JSON string for transcript: {json_str}")
        return {**state, "transcript_json_str": json_str, "error_message": None}

    except Exception as e:
        error_msg = f"Error during Phi-4 transcription API call: {str(e)}"
        print(error_msg)
        return {**state, "error_message": error_msg}

def parse_transcript_json(state: GraphState) -> GraphState:
    """Parses the transcript JSON string into a Pydantic model."""
    print("\n--- Node: Parse Transcript JSON ---")
    state["current_task_description"] = "Parsing transcript JSON."
    if state.get("error_message") or not state.get("transcript_json_str"):
        return state

    try:
        # Re-validate and parse, as the string might still be problematic
        raw_json_str = state["transcript_json_str"]
        json_start = raw_json_str.find('{')
        json_end = raw_json_str.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            cleaned_json_str = raw_json_str[json_start:json_end]
            data = json.loads(cleaned_json_str)
            parsed_transcript = Transcription(**data)
            print(f"Successfully parsed transcript: {parsed_transcript.dict()}")
            return {**state, "parsed_transcript": parsed_transcript, "error_message": None}
        else:
            raise ValueError("JSON delimiters not found in transcript_json_str for parsing.")
            
    except Exception as e:
        error_msg = f"Error parsing transcript JSON string '{state['transcript_json_str']}': {str(e)}"
        print(error_msg)
        return {**state, "error_message": error_msg, "parsed_transcript": None}


def extract_call_metadata(state: GraphState) -> GraphState:
    """Extracts call metadata from the transcript using Phi-4."""
    print("\n--- Node: Extract Call Metadata ---")
    state["current_task_description"] = "Extracting metadata from transcript using Phi-4."
    if state.get("error_message") or not state.get("parsed_transcript") or not state.get("parsed_transcript").transcript:
        print("Skipping metadata extraction due to previous error or empty transcript.")
        # If no transcript, we can't extract metadata. Set metadata to empty or default.
        return {**state, "extracted_metadata": CallMetadata()} # Return empty metadata

    llm_metadata = get_azure_phi4_llm(max_tokens=500) # Fewer tokens for metadata
    
    transcript_text_for_prompt = "\n".join([f"{entry.speaker}: {entry.text}" for entry in state["parsed_transcript"].transcript])

    metadata_prompt_instructions = (
        "You are an expert AI assistant. Your task is to analyze the provided call transcript and extract specific metadata. "
        "The transcript is a conversation. Identify the caller, receiver, subject, key entities, overall sentiment, and any action items. "
        "Respond ONLY with a valid JSON object that conforms to the CallMetadata schema. "
        "Do not include any explanations, markdown, or extra text. "
        "If a piece of information cannot be found, omit the field or set it to null where appropriate (as per schema).\n\n"
        "Call Transcript:\n"
        f"{transcript_text_for_prompt}"
    )
    
    messages = [
        SystemMessage(content=(
            "You are an AI assistant that extracts structured information from text and outputs it in JSON format according to the provided schema. "
            "Ensure your output is ONLY the JSON object."
        )),
        HumanMessage(content=metadata_prompt_instructions)
    ]

    try:
        print("Sending metadata extraction request to Phi-4 endpoint...")
        start_time = time.time()
        # Using .with_structured_output for Pydantic model enforcement
        structured_llm = llm_metadata.with_structured_output(CallMetadata)
        extracted_metadata: CallMetadata = structured_llm.invoke(messages)
        print(f"Phi-4 metadata extraction call completed in {time.time() - start_time:.2f} seconds.")
        print(f"Extracted metadata: {extracted_metadata.dict()}")
        return {**state, "extracted_metadata": extracted_metadata, "error_message": None}

    except Exception as e:
        error_msg = f"Error during Phi-4 metadata extraction: {str(e)}"
        # Attempt a non-structured call as fallback if structured output fails (might happen with complex models/prompts)
        try:
            print(f"Structured output failed for metadata, trying raw call: {e}")
            response = llm_metadata.invoke(messages)
            content = response.content
            print(f"Raw metadata response content: {content}")
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                data = json.loads(json_str)
                extracted_metadata = CallMetadata(**data) # Try to parse into Pydantic
                print(f"Fallback extracted metadata: {extracted_metadata.dict()}")
                return {**state, "extracted_metadata": extracted_metadata, "error_message": f"Fallback extraction used: {e}"}
            else:
                raise ValueError("Could not find JSON in fallback metadata response.")
        except Exception as fallback_e:
            final_error_msg = f"Error during Phi-4 metadata extraction (including fallback): {fallback_e}. Original error: {e}"
            print(final_error_msg)
            return {**state, "error_message": final_error_msg, "extracted_metadata": CallMetadata()} # Return empty metadata on failure

def compile_final_output(state: GraphState) -> GraphState:
    """Compiles the final structured output."""
    print("\n--- Node: Compile Final Output ---")
    state["current_task_description"] = "Compiling final results."
    if state.get("error_message") and not (state.get("parsed_transcript") or state.get("extracted_metadata")):
        # If there was a critical error before any meaningful output was generated
        final_result = {
            "status": "Error",
            "audio_file": state["audio_path"],
            "error": state["error_message"],
            "current_task_when_failed": state.get("current_task_description", "Unknown")
        }
    else:
        final_result = {
            "status": "Success" if not state.get("error_message") else "Partial Success with Errors",
            "audio_file": state["audio_path"],
            "transcript": state.get("parsed_transcript").dict() if state.get("parsed_transcript") else "Transcription failed or not available.",
            "metadata": state.get("extracted_metadata").dict() if state.get("extracted_metadata") else "Metadata extraction failed or not available.",
            "processing_error_details": state.get("error_message") if state.get("error_message") else None
        }
    print(f"Final compiled result: {json.dumps(final_result, indent=2)}")
    return {**state, "final_result": final_result}

# --- Conditional Edges (Decision Logic) ---
def should_proceed_or_handle_error(state: GraphState) -> str:
    """Determines the next step based on whether an error occurred."""
    if state.get("error_message"):
        print(f"Error detected: '{state['error_message']}'. Proceeding to compile output with error info.")
        return "compile_output" # Always compile output, even if it's just an error message
    print("No error detected, proceeding to next step.")
    return "continue"

# --- Build the Graph ---
workflow = StateGraph(GraphState)

workflow.add_node("load_audio", load_and_encode_audio)
workflow.add_node("transcribe", transcribe_audio_phi4)
workflow.add_node("parse_transcript_node", parse_transcript_json)
workflow.add_node("extract_metadata", extract_call_metadata)
workflow.add_node("compile_output", compile_final_output)

# Define edges
workflow.set_entry_point("load_audio")
workflow.add_edge("load_audio", "transcribe")
workflow.add_edge("transcribe", "parse_transcript_node")
workflow.add_edge("parse_transcript_node", "extract_metadata")
workflow.add_edge("extract_metadata", "compile_output")
workflow.add_edge("compile_output", END)


# Compile the graph
app = workflow.compile()

# --- Main Execution ---
if __name__ == "__main__":
    audio_file_path = r"C:\doc_audio_project\audio_files\Doctor Patient audio.mp3"  # Ensure this path is correct
    # audio_file_path = r"C:\doc_audio_project\audio_files\short_example.mp3" # Use a short audio for faster testing

    if not os.path.exists(audio_file_path):
        print(f"ERROR: Audio file not found at {audio_file_path}")
        exit()

    # Define the initial prompt for transcription
    # Note: The prompt for transcription in your original code asks for speaker, text, and emotion.
    # The Transcription Pydantic model reflects this.
    transcription_prompt_instructions = (
        "You are an expert transcriptionist. Your ONLY task is to transcribe the provided phone call audio with high accuracy.\n"
        "- Each segment must include:\n"
        "  - The speaker label (e.g., 'Doctor', 'Patient','agent','customer').\n"
        "  - The spoken text.\n"
        "  - The emotion related to the spoken text (e.g., 'neutral', 'happy', 'angry', 'sad'). If not clear, omit or use 'unknown'.\n"
        "- Do not summarize, do not explain, do not add any comments or extra text.\n"
        "- Output ONLY valid JSON that strictly conforms to the Transcription schema, no explanations, no markdown, no extra text, no preamble, no postamble.\n"
        "- The JSON must be in this format (ensure all fields are present as per Transcription schema):\n"
        "{\n"
        "  \"transcript\": [\n"
        "    { \"speaker\": \"Person1\", \"text\": \"Hello there.\", \"emotion\": \"neutral\" },\n"
        "    { \"speaker\": \"Person2\", \"text\": \"Hi, how are you?\", \"emotion\": \"curious\" }\n"
        "  ]\n"
        "}\n"
    )

    initial_state = GraphState(
        audio_path=audio_file_path,
        base64_audio=None,
        transcription_prompt=transcription_prompt_instructions,
        transcript_json_str=None,
        parsed_transcript=None,
        metadata_extraction_prompt=None, # Will be generated inside the node if needed
        extracted_metadata=None,
        final_result=None,
        error_message=None,
        current_task_description=None
    )

    print("Starting LangGraph audio processing workflow...")
    final_state = None
    for event in app.stream(initial_state, {"recursion_limit": 10}): # Limit recursion for safety
        for key, value in event.items():
            print(f"--- Event from Node: {key} ---")
            # print(f"Updated State: {json.dumps(value, indent=2, default=str)}") # Can be very verbose
            final_state = value # Keep track of the last state

    print("\n\n--- Workflow Complete ---")
    if final_state and final_state.get("final_result"):
        print("\n--- Final Compiled Result ---")
        print(json.dumps(final_state["final_result"], indent=2, default=str))
    elif final_state and final_state.get("error_message"):
        print(f"\n--- Workflow ended with an error ---")
        print(f"Error: {final_state['error_message']}")
        print(f"Last task description: {final_state.get('current_task_description')}")
    else:
        print("\n--- Workflow completed, but no final result or error message was explicitly set in the last state. ---")
        # print(f"Final state: {json.dumps(final_state, indent=2, default=str)}")