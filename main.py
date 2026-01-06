# Standard library
import os
import sys
import logging
import time

# Third-party
import pyttsx3
import speech_recognition as sr
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaLLM
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# Local tools
from tools.time import get_time
from tools.OCR import read_text_from_latest_image
from tools.arp_scan import arp_scan_terminal
from tools.duckduckgo import duckduckgo_search_tool
from tools.matrix import matrix_mode
from tools.screenshot import take_screenshot

load_dotenv()

# Config constants
MIC_INDEX = None
TRIGGER_WORD = "jarvis"
CONVERSATION_TIMEOUT = 30  # seconds of inactivity before exiting conversation mode

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# Microphone setup
recognizer = sr.Recognizer()
try:
    mic = sr.Microphone(device_index=MIC_INDEX)
except OSError as e:
    logging.critical(f"❌ Microphone not found: {e}")
    sys.exit(1)

# Initialize LLM
llm = ChatOllama(model="qwen3:1.7b", reasoning=False)

# Tool list
tools = [get_time, arp_scan_terminal, read_text_from_latest_image,
         duckduckgo_search_tool, matrix_mode, take_screenshot]

# Tool-calling prompt
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are Jarvis, an intelligent, conversational AI assistant. "
     "Your goal is to be helpful, friendly, and informative. You can respond "
     "in natural, human-like language and use tools when needed to answer "
     "questions more accurately. Always explain your reasoning simply when "
     "appropriate, and keep your responses conversational and concise."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Agent + executor
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# TTS setup
def speak_text(text: str):
    """Speak text aloud using pyttsx3."""
    try:
        engine = pyttsx3.init()
        for voice in engine.getProperty("voices"):
            if "jamie" in voice.name.lower():
                engine.setProperty("voice", voice.id)
                break
        engine.setProperty("rate", 180)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
        time.sleep(0.3)
    except Exception as e:
        logging.error(f"❌ TTS failed: {e}")


# Helper functions
def listen_for_wake_word(source):
    """Listen for audio and return transcript for wake word detection."""
    audio = recognizer.listen(source, timeout=10)
    return recognizer.recognize_google(audio)


def listen_for_command(source):
    """Listen for audio and return transcript for commands."""
    audio = recognizer.listen(source, timeout=10)
    return recognizer.recognize_google(audio)


def handle_command(command: str):
    """Send command to agent and speak response."""
    logging.info("🤖 Sending command to agent...")
    response = executor.invoke({"input": command})
    content = response.get("output", "[No response]")
    logging.info(f"✅ Agent responded: {content}")
    logging.info(f"Jarvis: {content}")
    speak_text(content)
    return content


# Main interaction loop
def write():
    """Main interaction loop for wake word and conversation mode."""
    conversation_mode = False
    last_interaction_time = None

    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            while True:
                # Timeout check at start of loop
                if conversation_mode and last_interaction_time:
                    if time.time() - last_interaction_time > CONVERSATION_TIMEOUT:
                        logging.info("⌛ Timeout: Returning to wake word mode.")
                        conversation_mode = False

                try:
                    if not conversation_mode:
                        logging.info("🎤 Listening for wake word...")
                        transcript = listen_for_wake_word(source)
                        logging.info(f"🗣 Heard: {transcript}")

                        if TRIGGER_WORD.lower() in transcript.lower():
                            logging.info(f"🗣 Triggered by: {transcript}")
                            speak_text("Yes sir?")
                            conversation_mode = True
                            last_interaction_time = time.time()
                        else:
                            logging.debug("Wake word not detected, continuing...")
                    else:
                        logging.info("🎤 Listening for next command...")
                        command = listen_for_command(source)
                        logging.info(f"📥 Command: {command}")

                        handle_command(command)
                        last_interaction_time = time.time()

                except sr.WaitTimeoutError:
                    logging.warning("⚠️ Timeout waiting for audio.")
                except sr.UnknownValueError:
                    logging.warning("⚠️ Could not understand audio.")
                except Exception as e:
                    logging.error(f"❌ Error during recognition or tool call: {e}")
                    time.sleep(1)

    except Exception as e:
        logging.critical(f"❌ Critical error in main loop: {e}")


if __name__ == "__main__":
    write()