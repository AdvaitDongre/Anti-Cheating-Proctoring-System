import speech_recognition as sr
import subprocess
import time
import threading
import sys
import webbrowser
import os
from datetime import datetime
import random
import google.generativeai as genai
from dotenv import load_dotenv

class VoiceAgent:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize Gemini
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            print("Warning: GEMINI_API_KEY not found in .env file. Some features will be limited.")
            self.gemini_available = False
        else:
            try:
                genai.configure(api_key=self.gemini_api_key)
                # Using gemini-1.5-flash model
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                self.gemini_chat = self.gemini_model.start_chat(history=[])
                self.gemini_available = True
                print("Gemini 1.5 Flash initialized successfully")
                
                # Set up initial context for Gemini
                self._initialize_gemini_context()
            except Exception as e:
                print(f"Error initializing Gemini: {e}")
                self.gemini_available = False
        
        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.listening = True
        self.agent_name = "Ava"  # Give the agent a name
        self.user_name = "User"
        self.language = "en"  # Default language (can be changed dynamically)
        
        # Direct command mapping (for quick execution without Gemini)
        self.direct_commands = {
            "exit": self.exit_program,
            "quit": self.exit_program,
            "stop": self.exit_program,
            "goodbye": self.exit_program,
            "clear chat": self.clear_chat_history,
            "what time is it": self.current_time,
            "tell me a joke": self.tell_joke,
            "flip a coin": self.flip_coin,
            "change language": self.change_language
        }
        
        # Adjust for ambient noise at startup
        with self.microphone as source:
            print("Calibrating microphone for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        
        # Greet the user
        self.greet_user()

    def _initialize_gemini_context(self):
        """Set up the initial context for Gemini to understand its role"""
        context = """
        You are Ava, a friendly multilingual voice assistant with these capabilities:
        
        CORE FUNCTIONS:
        1. Browser Control:
           - Open/close Google Chrome
           - Open specific websites (ACTION: open_website)
           - Perform web searches (ACTION: web_search)
           - Search on YouTube (ACTION: youtube_search)
        
        2. System Control:
           - Exit/quit the program (ACTION: exit)
           - Clear conversation history (ACTION: clear_chat)
           - Change language (ACTION: change_language)
        
        3. Information:
           - Tell current time (ACTION: current_time)
           - Answer general knowledge questions
        
        4. Entertainment:
           - Tell jokes (ACTION: tell_joke)
           - Flip a coin (ACTION: flip_coin)
           - Play music (ACTION: play_music)
        
        5. Utilities:
           - Make calculations
           - Set reminders (ACTION: set_reminder)
           - Create to-do items (ACTION: add_todo)
        
        RESPONSE FORMAT:
        - For actionable requests, respond ONLY with:
          ACTION: <action_name>
          PARAMS: <parameters>
          LANG: <language_code>
        
        - For conversational responses, reply naturally in the user's preferred language.
        
        MULTILINGUAL SUPPORT:
        - You understand and respond in multiple languages.
        - Default to the user's current language preference.
        - If language isn't specified, use context clues to determine.
        
        USER PREFERENCES:
        - Current language: {self.language}
        - Be polite, concise, and helpful.
        - For media searches (YouTube), include relevant keywords.
        """
        self.gemini_chat.send_message(context)

    def greet_user(self):
        """Greet the user based on time of day and language"""
        hour = datetime.now().hour
        greetings = {
            'en': {
                'morning': "Good morning",
                'afternoon': "Good afternoon",
                'evening': "Good evening",
                'night': "Hello"
            },
            'es': {
                'morning': "Buenos días",
                'afternoon': "Buenas tardes",
                'evening': "Buenas noches",
                'night': "Hola"
            },
            'fr': {
                'morning': "Bonjour",
                'afternoon': "Bon après-midi",
                'evening': "Bonsoir",
                'night': "Salut"
            },
            'hi': {
                'morning': "शुभ प्रभात",
                'afternoon': "नमस्कार",
                'evening': "शुभ संध्या",
                'night': "नमस्ते"
            }
        }
        
        # Determine time of day
        if 5 <= hour < 12:
            time_key = 'morning'
        elif 12 <= hour < 17:
            time_key = 'afternoon'
        elif 17 <= hour < 22:
            time_key = 'evening'
        else:
            time_key = 'night'
        
        # Get appropriate greeting
        lang = self.language if self.language in greetings else 'en'
        greeting = greetings[lang].get(time_key, "Hello")
        
        if self.gemini_available:
            try:
                response = self.gemini_model.generate_content(
                    f"Generate a very short (1 sentence max) friendly greeting in {lang} "
                    f"for a voice assistant to say {greeting} to the user."
                )
                print(f"{self.agent_name}: {response.text}")
            except:
                print(f"{self.agent_name}: {greeting}! How can I help you today?")
        else:
            print(f"{self.agent_name}: {greeting}! How can I help you today?")

    def process_with_gemini(self, user_input):
        """Use Gemini to process natural language input and determine actions"""
        try:
            # Get Gemini's response
            response = self.gemini_chat.send_message(
                f"User said: {user_input}\n\n"
                f"Current language: {self.language}\n"
                "Determine if this requires an action or just a conversational response."
            )
            
            response_text = response.text
            
            # Check if this is an action command
            if "ACTION:" in response_text:
                # Extract action details
                action_line = response_text.split("ACTION:")[1].split("\n")[0].strip()
                params_line = ""
                if "PARAMS:" in response_text:
                    params_line = response_text.split("PARAMS:")[1].split("\n")[0].strip()
                lang_line = self.language
                if "LANG:" in response_text:
                    lang_line = response_text.split("LANG:")[1].split("\n")[0].strip()
                
                # Update language if changed
                if lang_line != self.language:
                    self.language = lang_line
                    print(f"{self.agent_name}: Language changed to {lang_line}")
                
                # Execute the action
                self.execute_action(action_line, params_line)
            else:
                # Just a conversational response - print it
                print(f"{self.agent_name}: {response_text}")
                
        except Exception as e:
            print(f"{self.agent_name}: I encountered an error processing that. Could you try again?")

    def execute_action(self, action, params):
        """Execute the specified action with parameters"""
        action = action.lower().strip()
        
        if action == "open_chrome":
            if params:
                self.open_chrome(params)
            else:
                self.open_chrome()
        elif action == "close_chrome":
            self.close_chrome()
        elif action == "web_search":
            if params:
                self.web_search(params)
            else:
                print(f"{self.agent_name}: What would you like me to search for?")
        elif action == "youtube_search":
            if params:
                self.youtube_search(params)
            else:
                print(f"{self.agent_name}: What would you like to search on YouTube?")
        elif action == "open_website":
            if params:
                self.open_website(params)
            else:
                print(f"{self.agent_name}: Which website would you like me to open?")
        elif action == "tell_joke":
            self.tell_joke()
        elif action == "flip_coin":
            self.flip_coin()
        elif action == "current_time":
            self.current_time()
        elif action == "play_music":
            if params:
                self.play_music(params)
            else:
                print(f"{self.agent_name}: What music would you like to play?")
        elif action == "set_reminder":
            if params:
                self.set_reminder(params)
            else:
                print(f"{self.agent_name}: What should I remind you about?")
        elif action == "add_todo":
            if params:
                self.add_todo(params)
            else:
                print(f"{self.agent_name}: What should I add to your to-do list?")
        elif action == "change_language":
            if params:
                self.change_language(params)
        elif action == "exit":
            self.exit_program()
        elif action == "clear_chat":
            self.clear_chat_history()
        else:
            print(f"{self.agent_name}: I'm not sure how to perform that action.")

    # Action implementations
    def open_chrome(self, url=None):
        """Open Google Chrome browser, optionally with a specific URL"""
        try:
            chrome_paths = [
                'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
                '/usr/bin/google-chrome',
                '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
                'start chrome'
            ]
            
            for path in chrome_paths:
                try:
                    if url:
                        subprocess.Popen([path, url], shell=True)
                        print(f"{self.agent_name}: Opening Chrome with {url}")
                    else:
                        subprocess.Popen(path, shell=True)
                        print(f"{self.agent_name}: Opening Google Chrome")
                    return True
                except Exception:
                    continue
            
            try:
                chrome = webbrowser.get('chrome')
                if url:
                    chrome.open(url)
                else:
                    chrome.open_new()
                print(f"{self.agent_name}: Opening Chrome")
                return True
            except:
                print(f"{self.agent_name}: I couldn't find Chrome on your system")
                return False
        except Exception as e:
            print(f"{self.agent_name}: Error opening Chrome: {e}")
            return False

    def close_chrome(self):
        """Close Google Chrome browser"""
        try:
            if sys.platform == 'win32':
                os.system('taskkill /f /im chrome.exe')
            else:
                os.system('pkill -f "Google Chrome"')
            print(f"{self.agent_name}: Closed Google Chrome")
            return True
        except Exception as e:
            print(f"{self.agent_name}: Error closing Chrome: {e}")
            return False

    def web_search(self, query):
        """Perform a web search"""
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        self.open_chrome(search_url)

    def youtube_search(self, query):
        """Perform a YouTube search"""
        search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
        self.open_chrome(search_url)

    def open_website(self, site):
        """Open a specific website"""
        if not site.startswith(('http://', 'https://')):
            site = f'https://{site}'
        self.open_chrome(site)

    def play_music(self, query):
        """Play music on YouTube"""
        self.youtube_search(f"{query} music")

    def set_reminder(self, text):
        """Set a reminder (simplified implementation)"""
        print(f"{self.agent_name}: I'll remind you to: {text}")
        # In a real implementation, you'd store this with a timestamp

    def add_todo(self, item):
        """Add an item to the to-do list (simplified implementation)"""
        print(f"{self.agent_name}: Added to your to-do list: {item}")

    def change_language(self, lang):
        """Change the assistant's language"""
        supported_languages = {
            'english': 'en',
            'spanish': 'es',
            'french': 'fr',
            'hindi': 'hi',
            'en': 'en',
            'es': 'es',
            'fr': 'fr',
            'hi': 'hi'
        }
        
        lang = lang.lower().strip()
        if lang in supported_languages:
            self.language = supported_languages[lang]
            print(f"{self.agent_name}: Language changed to {lang}")
            # Update Gemini context
            if self.gemini_available:
                self.gemini_chat.send_message(f"Note: The user has changed the language preference to {lang}")
        else:
            print(f"{self.agent_name}: Sorry, I don't support {lang}. Supported languages are: English, Spanish, French, Hindi")

    def tell_joke(self):
        """Tell a random joke in the current language"""
        if self.gemini_available:
            try:
                response = self.gemini_model.generate_content(
                    f"Tell me a very short, clean joke in {self.language}"
                )
                print(f"{self.agent_name}: {response.text}")
            except:
                self._tell_default_joke()
        else:
            self._tell_default_joke()

    def _tell_default_joke(self):
        """Fallback jokes when Gemini isn't available"""
        jokes = {
            'en': [
                "Why don't scientists trust atoms? Because they make up everything!",
                "Why did the scarecrow win an award? Because he was outstanding in his field!"
            ],
            'es': [
                "¿Qué le dice un semáforo a otro? No me mires, me estoy cambiando.",
                "¿Cómo se llama el campeón de buceo japonés? Tokofondo."
            ],
            'fr': [
                "Pourquoi les plongeurs plongent-ils toujours en arrière et pas en avant ? Parce que sinon ils tombent dans le bateau.",
                "Quel est le comble pour un électricien ? De ne pas avoir de courant."
            ],
            'hi': [
                "एक डॉक्टर दूसरे डॉक्टर से: आपके मरीज ने आपकी फीस चुका दी है? दूसरा डॉक्टर: नहीं, वह तो मेरे इलाज से ही ठीक हो गया!",
                "टीचर: बताओ, 2 और 2 कितने होते हैं? छात्र: 4 टीचर: बहुत अच्छा! छात्र: बहुत अच्छा नहीं, बिल्कुल सही!"
            ]
        }
        lang = self.language if self.language in jokes else 'en'
        print(f"{self.agent_name}: {random.choice(jokes[lang])}")

    def flip_coin(self):
        """Flip a virtual coin"""
        results = {
            'en': ["Heads", "Tails"],
            'es': ["Cara", "Cruz"],
            'fr': ["Pile", "Face"],
            'hi': ["सिर", "पुच्छ"]
        }
        lang = self.language if self.language in results else 'en'
        result = random.choice(results[lang])
        print(f"{self.agent_name}: {result}!")

    def current_time(self):
        """Tell the current time"""
        now = datetime.now()
        time_formats = {
            'en': f"The current time is {now.strftime('%I:%M %p')}",
            'es': f"La hora actual es {now.strftime('%I:%M %p')}",
            'fr': f"Il est actuellement {now.strftime('%H:%M')}",
            'hi': f"वर्तमान समय है {now.strftime('%I:%M %p')}"
        }
        lang = self.language if self.language in time_formats else 'en'
        print(f"{self.agent_name}: {time_formats[lang]}")

    def exit_program(self):
        """Exit the voice agent"""
        goodbyes = {
            'en': "Goodbye! Have a nice day!",
            'es': "¡Adiós! ¡Que tengas un buen día!",
            'fr': "Au revoir! Passe une bonne journée!",
            'hi': "अलविदा! आपका दिन शुभ हो!"
        }
        lang = self.language if self.language in goodbyes else 'en'
        print(f"{self.agent_name}: {goodbyes[lang]}")
        self.listening = False
        sys.exit(0)

    def clear_chat_history(self):
        """Clear the chat history"""
        if self.gemini_available:
            self.gemini_chat = self.gemini_model.start_chat(history=[])
            self._initialize_gemini_context()
            print(f"{self.agent_name}: Conversation history cleared.")
        else:
            print(f"{self.agent_name}: I don't have chat history to clear.")

    def process_command(self, command):
        """Process the voice command"""
        command = command.lower()
        
        # First check direct commands for quick execution
        for cmd in self.direct_commands:
            if cmd in command:
                self.direct_commands[cmd]()
                return
        
        # Otherwise process with Gemini for natural language understanding
        if self.gemini_available:
            self.process_with_gemini(command)
        else:
            # Fallback for when Gemini isn't available
            if "chrome" in command and ("open" in command or "start" in command):
                self.open_chrome()
            elif "chrome" in command and "close" in command:
                self.close_chrome()
            elif "search" in command and "youtube" in command:
                query = command.replace("search", "").replace("youtube", "").strip()
                if query:
                    self.youtube_search(query)
                else:
                    print(f"{self.agent_name}: What would you like to search on YouTube?")
            elif "search" in command:
                query = command.replace("search", "").strip()
                if query:
                    self.web_search(query)
                else:
                    print(f"{self.agent_name}: What would you like me to search for?")
            elif "open" in command and "website" in command:
                site = command.replace("open", "").replace("website", "").strip()
                if site:
                    self.open_website(site)
                else:
                    print(f"{self.agent_name}: Which website would you like me to open?")
            else:
                print(f"{self.agent_name}: I didn't understand that. You can ask me to open Chrome, search the web, or open websites.")

    def listen_continuously(self):
        """Continuously listen for voice commands"""
        print(f"{self.agent_name}: I'm listening. How can I help you?")
        
        while self.listening:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=5)
                    
                    try:
                        command = self.recognizer.recognize_google(audio)
                        print(f"{self.user_name}: {command}")
                        self.process_command(command)
                    except sr.UnknownValueError:
                        # No speech detected
                        continue
                    except sr.RequestError as e:
                        print(f"{self.agent_name}: Sorry, I'm having trouble with the speech service. Please try again.")
                        time.sleep(2)
                        
            except Exception as e:
                print(f"{self.agent_name}: Error in listening: {e}")
                time.sleep(1)

    def run(self):
        """Run the agent"""
        # Start listening in a separate thread
        listener_thread = threading.Thread(target=self.listen_continuously, daemon=True)
        listener_thread.start()
        
        # Keep main thread alive
        try:
            while self.listening:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.exit_program()

if __name__ == "__main__":
    agent = VoiceAgent()
    print(f"=== {agent.agent_name} Voice Assistant ===")
    print("Multilingual assistant with YouTube search powered by Gemini 1.5 Flash")
    agent.run()