from telethon import TelegramClient, events
from telethon.tl.types import User
import asyncio
import logging
import os
from datetime import datetime
import random
import aiohttp
from telethon import errors
from typing import List
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
API_ID = os.getenv('TELEGRAM_APP_API_ID_PANDUKA')
API_HASH = os.getenv('TELEGRAM_APP_API_HASH_PANDUKA')
FASTAPI_URL = os.getenv('FASTAPI_URL', 'http://localhost:8000')

# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

async def read_questions_from_file(filename: str = "followup_questions.txt") -> List[str]:
    """Read questions from a file"""
    try:
        with open(filename, "r") as file:
            questions = file.readlines()
            return questions
    except Exception as e:
        logger.error(f"Error reading questions from file: {str(e)}")
        return []


class InterviewAgent:
    def __init__(self, api_id: str, api_hash: str):
        self.client = TelegramClient('interview_session', api_id, api_hash)
        self.active_sessions = {}
        
        # Initialize Supabase client
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized")

    async def store_message(self, candidate_id: str, message: str, sender: str):
        """Store a message in the chat history using Supabase"""
        current_time = datetime.utcnow().isoformat()
        
        try:
            # Insert message into Supabase chat_history_education table
            data = {
                'candidate_id': candidate_id,
                'message': message,
                'sender': sender,
                'timestamp': current_time
            }
            
            response = self.supabase.table('chat_history_education').insert(data).execute()
            if hasattr(response, 'error') and response.error:
                logger.error(f"Error storing message in Supabase: {response.error}")
            
        except Exception as e:
            logger.error(f"Error storing message: {e}")

    async def get_chat_history_education(self, candidate_id: str) -> List[dict]:
        """Retrieve chat history for a candidate from Supabase"""
        try:
            response = self.supabase.table('chat_history_education_education') \
                .select('message,sender,timestamp') \
                .eq('candidate_id', candidate_id) \
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

    async def get_next_question(self, candidate_id: str, chat_history_education: List[dict]) -> str:
        """Get the next question from FastAPI based on chat history"""
        try:
            async with aiohttp.ClientSession() as session:
                with open('jd.txt', 'r') as f:
                    job_description = f.read()
                payload = {
                    "chat_history_education": chat_history_education,
                    "job_description": job_description
                }
                async with session.post(f"{FASTAPI_URL}/conduct-interview", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(result)
                        return result.get("next_message", "Thank you for participating in the interview coaching session."), result.get("status", "unknown")
                    else:
                        logger.error(f"Error from FastAPI: {response.status}")
                        return "I'm having trouble connecting to our coaching system. Let me try again with a general question: Tell me about your experience with problem-solving in your most recent role.", "continue"
        except Exception as e:
            logger.error(f"Error connecting to FastAPI: {e}")
            return "I'm experiencing some technical difficulties. Could you tell me about a challenging project you've worked on recently?", "continue"

    async def handle_candidate_response(self, candidate_id: str, message: str):
        """Process candidate response and determine next steps"""
        # Store candidate's message
        await self.store_message(candidate_id, message, "candidate")
        
        # Show typing indicator
        async with self.client.action(int(candidate_id), 'typing'):
            # Add a realistic delay
            await asyncio.sleep(random.uniform(2, 4))
            
            # Get chat history
            chat_history_education = await self.get_chat_history_education(candidate_id)
            
            # Get next question from API
            next_question, status = await self.get_next_question(candidate_id, chat_history_education)
            
            # Check if interview is complete
            if status == 'END':
                try:
                    # Update candidate status in Supabase
                    self.supabase.table('candidates') \
                        .update({'status': 'completed', 'interview_complete': True}) \
                        .eq('candidate_id', candidate_id) \
                        .execute()
                    
                    # Send final message
                    await self.client.send_message(int(candidate_id), next_question)
                    await self.store_message(candidate_id, next_question, "agent")
                    
                    # Remove from active sessions
                    if candidate_id in self.active_sessions:
                        del self.active_sessions[candidate_id]
                    
                    logger.info(f"Interview completed for candidate {candidate_id}")
                    os._exit(0)
                    
                except Exception as e:
                    logger.error(f"Error updating candidate status: {e}")
            
            # Send next question
            await self.client.send_message(int(candidate_id), next_question)
            await self.store_message(candidate_id, next_question, "agent")

    async def start_interview(self, candidate_id: str):
        """Begin the interview process"""
        # Add candidate to active sessions
        self.active_sessions[candidate_id] = {"active": True}
        
        # Send welcome message
        welcome_message = (
            "👋 Welcome to our job follow-up session!\n"
            "We'll focus on reviewing your recent job applications and discussing any updates or next steps. This is a great opportunity to reflect on your progress, address any concerns, and prepare for what's ahead."
            "Take your time to share your thoughts and experiences—I'm here to support you every step of the way."
        )
        
        await self.client.send_message(int(candidate_id), welcome_message)
        await self.store_message(candidate_id, welcome_message, "agent")
        
        # Show typing indicator and add delay
        async with self.client.action(int(candidate_id), 'typing'):
            await asyncio.sleep(3)
            
            # Get first question from file instead of API initially
            questions = await read_questions_from_file()
            first_question = questions[0] if questions else "Tell me about your technical background and skills."
            
            # Send first question
            await self.client.send_message(int(candidate_id), first_question)
            await self.store_message(candidate_id, first_question, "agent")

    async def process_message(self, event, sender: User):
        """Handle incoming messages from candidates"""
        candidate_id = str(sender.id)
        message = event.message.text
        
        # Check if candidate exists in Supabase
        try:
            response = self.supabase.table('candidates') \
                .select('status,interview_complete') \
                .eq('candidate_id', candidate_id) \
                .execute()
            
            candidate_exists = len(response.data) > 0 if hasattr(response, 'data') else False
            
            if not candidate_exists:
                # New candidate - register them in Supabase
                new_candidate = {
                    'candidate_id': candidate_id,
                    'phone_number': "",  # Phone number isn't needed for this flow
                    'created_at': datetime.utcnow().isoformat(),
                    'status': 'active',
                    'interview_complete': False
                }
                
                self.supabase.table('candidates').insert(new_candidate).execute()
                
                # Start interview for new candidate
                await self.start_interview(candidate_id)
            else:
                # Existing candidate - check status
                status_data = response.data[0]
                
                if status_data.get('status') == 'completed' or status_data.get('interview_complete'):
                    # Interview already completed
                    await self.client.send_message(
                        int(candidate_id),
                        "Your interview coaching session has already been completed. Thank you for participating!"
                    )
                else:
                    # Process the message in ongoing interview
                    await self.handle_candidate_response(candidate_id, message)
                    
        except Exception as e:
            logger.error(f"Error checking candidate status: {e}")
            # Fallback - assume new candidate and start interview
            await self.start_interview(candidate_id)

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
            if event.is_private:
                sender = await event.get_sender()
                if isinstance(sender, User):
                    await self.process_message(event, sender)

        logger.info("Message handlers registered")
        await self.client.run_until_disconnected()

    async def add_candidate(self, phone_number: str):
        """Add a new candidate to the system (for initialization)"""
        try:
            if not self.client.is_connected():
                await self.connect()
                
            try:
                input_entity = await self.client.get_input_entity(phone_number)
                contact = await self.client.get_entity(input_entity)
            except errors.FloodWaitError as e:
                logger.warning(f"Need to wait {e.seconds} seconds before retrying")
                await asyncio.sleep(e.seconds)
                input_entity = await self.client.get_input_entity(phone_number)
                contact = await self.client.get_entity(input_entity)
                
            candidate_id = str(contact.id)
            
            # Store in Supabase
            new_candidate = {
                'candidate_id': candidate_id,
                'phone_number': phone_number,
                'created_at': datetime.utcnow().isoformat(),
                'status': 'active',
                'interview_complete': False
            }
            
            self.supabase.table('candidates').upsert(new_candidate).execute()
            
            # Start interview immediately
            await self.start_interview(candidate_id)
            
            logger.info(f"Added and started interview for candidate: {phone_number}")
            return True
        except Exception as e:
            logger.error(f"Error adding candidate: {e}")
            return False

async def main():
    # Check for Supabase credentials
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("Missing Supabase credentials. Please add SUPABASE_URL and SUPABASE_KEY to your .env file")
        return
        
    agent = InterviewAgent(API_ID, API_HASH)
    print(API_ID, API_HASH)
    await agent.connect()
    
    # Optional: Add test candidate to trigger initial conversation
    test_phone = "+94712623943"  # Replace with actual phone number
    success = await agent.add_candidate(test_phone)
    
    if success:
        logger.info("Test candidate added successfully")

    # Start the agent to handle all conversations
    await agent.start()

    
if __name__ == "__main__":
    asyncio.run(main())