from telethon import TelegramClient, events
from telethon.tl.types import User
import asyncio
import logging
import os
import random
import aiohttp
import json
from telethon import errors
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
API_ID = os.getenv('TELEGRAM_APP_API_ID_NIM')
API_HASH = os.getenv('TELEGRAM_APP_API_HASH_NIM')
FASTAPI_URL = os.getenv('FASTAPI_URL', 'http://localhost:8000')

# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
print(SUPABASE_URL, SUPABASE_KEY)

class RAGBot:
    def __init__(self, api_id: str, api_hash: str):
        self.client = TelegramClient('rag_session', api_id, api_hash)
        self.active_users = {}
        
        # TESTING MODIFICATION: Track the real Telegram user ID alongside test user ID
        self.test_mode = False
        self.test_user_id = None
        self.real_telegram_id = None
        
        # Initialize Supabase client
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized")

    async def store_message(self, user_id: str, message: str, sender: str):
        """Store a message in the chat history using Supabase"""
        current_time = datetime.utcnow().isoformat()
        
        try:
            # Insert message into Supabase chat_history_education table
            data = {
                'candidate_id': user_id,  # Using candidate_id field for user_id to maintain compatibility
                'message': message,
                'sender': sender,
                'timestamp': current_time
            }
            
            response = self.supabase.table('chat_history_education').insert(data).execute()
            if hasattr(response, 'error') and response.error:
                logger.error(f"Error storing message in Supabase: {response.error}")
            
        except Exception as e:
            logger.error(f"Error storing message: {e}")

    async def get_chat_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve chat history for a user from Supabase"""
        try:
            response = self.supabase.table('chat_history_education') \
                .select('message,sender,timestamp') \
                .eq('candidate_id', user_id) \
                .order('timestamp', desc=False) \
                .execute()
            
            if hasattr(response, 'error') and response.error:
                logger.error(f"Error fetching chat history from Supabase: {response.error}")
                return []
                
            # Format the response data
            history = [{"message": item["message"], 
                        "sender": item["sender"], 
                        "timestamp": item["timestamp"]} 
                       for item in response.data]
            
            return history
            
        except Exception as e:
            logger.error(f"Error retrieving chat history: {e}")
            return []

    async def get_rag_response(self, user_id: str, query: str, chat_history: List[Dict[str, Any]]) -> str:
        """Get response from RAG system via FastAPI"""
        try:
            async with aiohttp.ClientSession() as session:
                if chat_history is None:
                    chat_history = []

                # Ensure proper format for JSON serialization
                formatted_history = []
                for msg in chat_history:
                    formatted_history.append({
                        "message": str(msg.get("message", "")),
                        "sender": str(msg.get("sender", "")),
                        "timestamp": str(msg.get("timestamp", ""))
                    })

                # Create payload with properly formatted data
                payload = {
                    "query": query,
                    "chat_history": formatted_history
                }
                
                # Log request for debugging
                logger.info(f"Sending request to RAG API: {FASTAPI_URL}/query")
                logger.debug(f"Payload: {json.dumps(payload)}")
                
                # Make API request, ensuring proper JSON serialization
                try:
                    async with session.post(f"{FASTAPI_URL}/query", json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result.get("answer", "I'm sorry, I couldn't find an answer to your question.")
                        else:
                            error_text = await response.text()
                            logger.error(f"Error from FastAPI: {response.status} - {error_text}")
                            return "I'm experiencing technical difficulties. Please try again later."
                except aiohttp.ClientError as e:
                    logger.error(f"API request error: {e}")
                    return "I'm experiencing technical difficulties connecting to our knowledge base. Please try again later."
                
        except Exception as e:
            logger.error(f"Error connecting to FastAPI: {e}")
            return "Sorry, I'm having trouble connecting to our knowledge base right now. Please try again in a few moments."

    async def handle_user_message(self, test_user_id: str, real_user_id: str, message: str):
        """Process user message and respond with RAG answer"""
        # Store user's message using the test user ID
        await self.store_message(test_user_id, message, "user")
        
        # Show typing indicator using the REAL Telegram ID for UI interactions
        async with self.client.action(int(real_user_id), 'typing'):
            # Add a realistic delay
            await asyncio.sleep(random.uniform(1, 3))
            
            # Get chat history using the test user ID
            chat_history = await self.get_chat_history(test_user_id)
            
            # Get RAG response
            rag_response = await self.get_rag_response(test_user_id, message, chat_history)
            
            # Send response using the REAL Telegram ID
            await self.client.send_message(int(real_user_id), rag_response)
            
            # Store the response using the test user ID
            await self.store_message(test_user_id, rag_response, "bot")

    async def register_user(self, user_id: str, phone_number: str = ""):
        """Register a new user in Supabase"""
        try:
            # Add user to the candidates table in Supabase (matching original structure)
            new_user = {
                'candidate_id': user_id,
                'phone_number': phone_number,
                'created_at': datetime.utcnow().isoformat(),
                'status': 'active',
                # 'interview_complete': False  # Keeping this field for compatibility
            }
            
            response = self.supabase.table('candidates_education').insert(new_user).execute()
            if hasattr(response, 'error') and response.error:
                logger.error(f"Error registering user in Supabase: {response.error}")
                return False
            
            logger.info(f"User {user_id} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error registering user: {e}")
            return False

    async def welcome_user(self, test_user_id: str, real_user_id: str):
        """Welcome a new user and register them in Supabase"""
        # Register user in Supabase with the test ID
        success = await self.register_user(test_user_id)
        
        # Add user to active users
        self.active_users[test_user_id] = {"active": True, "real_id": real_user_id}
        
        # Send welcome message (keeping the same from original)
        welcome_message = (
            "👋 Welcome to the AICBT Assistant!\n"
            "I'm here to help you with anything related to the AICBT.\n"
            "Feel free to ask questions or request assistance anytime!"
        )
        
        # Send message using REAL Telegram ID but store with test ID
        await self.client.send_message(int(real_user_id), welcome_message)
        await self.store_message(test_user_id, welcome_message, "bot")

    async def enable_test_mode(self, test_user_id: str):
        """Enable test mode with a specific test user ID"""
        self.test_mode = True
        self.test_user_id = test_user_id
        logger.info(f"TEST MODE ENABLED: All messages will be stored with user ID: {test_user_id}")
        
    async def disable_test_mode(self):
        """Disable test mode"""
        self.test_mode = False
        self.test_user_id = None
        self.real_telegram_id = None
        logger.info("TEST MODE DISABLED: Using actual Telegram user IDs")

    async def add_candidate(self, phone_number: str, test_user_id: str = None):
        """Add a new candidate to the system and initiate conversation"""
        try:
            if not self.client.is_connected():
                await self.connect()
            
            # Get the actual Telegram user ID from the phone number
            try:
                input_entity = await self.client.get_input_entity(phone_number)
                contact = await self.client.get_entity(input_entity)
                real_user_id = str(contact.id)
                logger.info(f"Found Telegram user with ID: {real_user_id} for phone: {phone_number}")
            except errors.FloodWaitError as e:
                logger.warning(f"Need to wait {e.seconds} seconds before retrying")
                await asyncio.sleep(e.seconds)
                input_entity = await self.client.get_input_entity(phone_number)
                contact = await self.client.get_entity(input_entity)
                real_user_id = str(contact.id)
            
            # TESTING MODIFICATION: Use test_user_id if provided
            user_id_for_storage = test_user_id if test_user_id else real_user_id
            
            if test_user_id:
                logger.info(f"TESTING MODE: Using manual user_id: {test_user_id} instead of actual Telegram ID")
                # Store the mapping between test ID and real ID
                self.test_user_id = test_user_id
                self.real_telegram_id = real_user_id
                await self.enable_test_mode(test_user_id)
            
            # Check if user already exists in Supabase
            response = self.supabase.table('candidates_education') \
                .select('*') \
                .eq('candidate_id', user_id_for_storage) \
                .execute()
            
            user_exists = len(response.data) > 0 if hasattr(response, 'data') else False
            
            if not user_exists:
                # Register in Supabase with the appropriate ID
                await self.register_user(user_id_for_storage, phone_number)
                
                # Start conversation with welcome message using both IDs
                await self.welcome_user(user_id_for_storage, real_user_id)
            else:
                # User already exists, just add to active users without welcome message
                logger.info(f"User {user_id_for_storage} already exists, adding to active users")
                self.active_users[user_id_for_storage] = {"active": True, "real_id": real_user_id}
            
            logger.info(f"Added candidate with phone number: {phone_number}, storage user_id: {user_id_for_storage}")
            return True
        except Exception as e:
            logger.error(f"Error adding candidate: {e}")
            return False

    async def process_message(self, event, sender: User):
        """Handle incoming messages from users"""
        real_user_id = str(sender.id)
        message = event.message.text
        
        # TESTING MODIFICATION: Use the test user ID for storage if in test mode
        if self.test_mode and self.test_user_id and real_user_id == self.real_telegram_id:
            user_id_for_storage = self.test_user_id
            logger.info(f"Using test user ID {user_id_for_storage} for message from {real_user_id}")
        else:
            user_id_for_storage = real_user_id
        
        # Check if user exists in Supabase
        try:
            response = self.supabase.table('candidates_education') \
                .select('*') \
                .eq('candidate_id', user_id_for_storage) \
                .execute()
            
            user_exists = len(response.data) > 0 if hasattr(response, 'data') else False
            
            if not user_exists:
                # New user - welcome and register them
                await self.welcome_user(user_id_for_storage, real_user_id)
            else:
                # Existing user - just add to active users without sending welcome message
                self.active_users[user_id_for_storage] = {"active": True, "real_id": real_user_id}
            
            # Process their message using both IDs
            await self.handle_user_message(user_id_for_storage, real_user_id, message)
                
        except Exception as e:
            logger.error(f"Error checking user status: {e}")
            # Fallback - assume new user and welcome them
            await self.welcome_user(user_id_for_storage, real_user_id)
            await self.handle_user_message(user_id_for_storage, real_user_id, message)

    async def connect(self):
        """Connect to Telegram"""
        await self.client.connect()
        if not await self.client.is_user_authorized():
            logger.info("First time setup - you'll need to authenticate")
            await self.client.start()
        logger.info("Client connected successfully")

    async def start(self):
        """Start the client and register event handlers"""
        await self.connect()
        
        @self.client.on(events.NewMessage())
        async def handle_message(event):
            if event.is_private:  # Only respond to private messages
                sender = await event.get_sender()
                if isinstance(sender, User):
                    await self.process_message(event, sender)

        logger.info("Message handlers registered")
        await self.client.run_until_disconnected()

async def main():
    # Check for Supabase credentials
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("Missing Supabase credentials. Please add SUPABASE_URL and SUPABASE_KEY to your .env file")
        return
        
    bot = RAGBot(API_ID, API_HASH)
    await bot.connect()
    
    # TESTING MODIFICATION: Use test user ID
    # ------------------------------------------------
    # ORIGINAL CODE: 
    # test_phone = "+94712623943"  # Replace with actual phone number
    # success = await bot.add_candidate(test_phone)
    
    # TESTING CODE: Specify a test user ID
    test_phone = "+94712623943"  # Your actual test phone number
    test_user_id = "567"  # Your desired test user ID
    
    # Add test user with manual ID (will use actual Telegram ID for messaging)
    success = await bot.add_candidate(test_phone, test_user_id)
    
    # TO REVERT TO PRODUCTION:
    # 1. Remove the test_user_id line
    # 2. Change the add_candidate call to: success = await bot.add_candidate(test_phone)
    # ------------------------------------------------
    
    if success:
        logger.info(f"Test candidate added successfully with ID: {test_user_id}")
    
    # Start the bot to handle all conversations
    await bot.start()

if __name__ == "__main__":
    asyncio.run(main())