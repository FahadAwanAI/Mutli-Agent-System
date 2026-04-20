"""
Voice Utility - Live Text-to-Speech for AI Prediction Battle
Uses edge-tts for high-quality neural voices (free, no API key needed)
Speaks live without saving files.
"""

import os
import io
import asyncio
import edge_tts
import pygame
import tempfile


# Voice assignments for each agent
AGENT_VOICES = {
    "ChatGPT": "en-US-GuyNeural",        # Professional male
    "Grok": "en-US-ChristopherNeural",   # Casual male  
    "Gemini": "en-US-EricNeural",        # Analytical male
    "Moderator": "en-GB-SoniaNeural",    # British female
}

DEFAULT_VOICE = "en-US-AriaNeural"


def get_voice_for_agent(agent_name: str) -> str:
    """Get the voice ID for a given agent."""
    return AGENT_VOICES.get(agent_name, DEFAULT_VOICE)


async def _speak_async(text: str, voice: str):
    """Generate and play speech asynchronously."""
    # Create temp file for audio
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
        temp_path = f.name
    
    try:
        # Generate speech to temp file
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(temp_path)
        
        # Play audio
        pygame.mixer.init()
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
        
        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        pygame.mixer.quit()
    except Exception as e:
        print(f"[Voice Error: {e}]")
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass


def speak(text: str, agent_name: str = "Moderator"):
    """
    Speak text using agent's voice (live, no file saving).
    """
    voice = get_voice_for_agent(agent_name)
    
    # Fix Windows asyncio issue
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_speak_async(text, voice))
    finally:
        loop.close()


def test_voice():
    """Test all agent voices."""
    from src.utils.console import console
    
    console.print("\n🎙️ Testing live voice output...")
    
    test_texts = {
        "Moderator": "Welcome to the AI Prediction Battle.",
        "ChatGPT": "I predict YES with 75% confidence.",
        "Grok": "Early signals suggest this will happen.",
        "Gemini": "Historical constraints make this unlikely.",
    }
    
    for agent, text in test_texts.items():
        console.print(f"   🔊 {agent} speaking...")
        speak(text, agent)
        console.print(f"   ✓ Done")
    
    console.print("\n✅ Voice test complete!")
    return True
