"""
PMO Expert Consultant - AI-Powered Poe Server Bot
Deployed on Render.com
"""

from __future__ import annotations
from typing import AsyncIterable
import fastapi_poe as fp
import json
import httpx
import os
from openai import AsyncOpenAI

# Webhook endpoint
WEBHOOK_URL = "https://natsha.pythonanywhere.com/webhook/ai-program-create"

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Conversation state storage
conversation_states = {}


class PMOConsultantBot(fp.PoeBot):
    """
    AI-powered PMO consultant that intelligently collects program information
    and generates PowerPoint reports
    """

    async def get_response(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        """Main response handler with AI intelligence"""
        
        user_id = request.user_id
        user_message = request.query[-1].content
        
        # Initialize state if new user
        if user_id not in conversation_states:
            conversation_states[user_id] = {
                "phase": "intro",
                "data": {},
                "conversation_history": []
            }
        
        state = conversation_states[user_id]
        
        # Add user message to history
        state["conversation_history"].append({
            "role": "user",
            "content": user_message
        })
        
        # Process based on phase
        if state["phase"] == "intro":
            response = await self._handle_intro(state, user_message)
        elif state["phase"] == "collecting":
            response = await self._handle_collection(state, user_message)
        elif state["phase"] == "complete":
            response = await self._handle_completion(state, user_message)
        else:
            response = "I'm not sure what to do. Let's start over. Type 'hello' to begin."
            state["phase"] = "intro"
        
        # Add bot response to history
        state["conversation_history"].append({
            "role": "assistant",
            "content": response
        })
        
        yield fp.PartialResponse(text=response)

    async def _handle_intro(self, state: dict, message: str) -> str:
        """Handle introduction phase"""
        state["phase"] = "collecting"
        return """👋 Hi! I'm your AI-powered PMO consultant. I'll help you scope your program and deliver 5 professional PowerPoint reports to your inbox.

**You can provide information in any way you like:**
- Answer my questions one by one, OR
- Give me all the details at once, OR
- Just chat naturally - I'll figure it out!

**I need to collect:**
1. Program name
2. Program manager name
3. Executive sponsor name
4. Program goals/objectives
5. Estimated budget
6. Start and end dates
7. Key milestones (3-5)
8. RAID items (Risks, Assumptions, Issues, Dependencies)
9. Your email address

**Let's get started! Tell me about your program.** You can share as much or as little as you want, and I'll ask follow-up questions for anything missing."""

    async def _handle_collection(self, state: dict, message: str) -> str:
        """Handle data collection phase with AI intelligence"""
        
        # Use GPT-4o to extract information and determine next steps
        extraction_result = await self._extract_data_with_ai(state, message)
        
        # Update state with extracted data
        for key, value in extraction_result["extracted_data"].items():
            if value and value != "unknown":
                state["data"][key] = value
        
        # Check if we have everything
        required_fields = [
            "program_name", "program_manager", "executive_sponsor",
            "goals", "budget", "start_date", "end_date",
            "milestones", "raid_items", "email"
        ]
        
        missing_fields = [f for f in required_fields if f not in state["data"] or not state["data"][f]]
        
        if not missing_fields:
            # All data collected - prepare for submission
            state["phase"] = "complete"
            return await self._prepare_submission(state)
        else:
            # Still need more information
            return extraction_result["next_question"]

    async def _extract_data_with_ai(self, state: dict, message: str) -> dict:
        """Use GPT-4o to intelligently extract data and generate next question"""
        
        current_data = state["data"]
        
        system_prompt = f"""You are a PMO consultant collecting program information. 

**Current data collected:**
{json.dumps(current_data, indent=2)}

**Your tasks:**
1. Extract any new information from the user's message
2. Determine what's still missing
3. Generate a natural follow-up question

**Required fields:**
- program_name: Name of the program
- program_manager: Full name of program manager
- executive_sponsor: Full name of executive sponsor
- goals: Program goals/objectives (string)
- budget: Estimated budget (integer, just the number)
- start_date: Start date (YYYY-MM-DD format)
- end_date: End date (YYYY-MM-DD format)
- milestones: List of 3-5 key milestones (array of strings)
- raid_items: List of risks, assumptions, issues, dependencies (array of strings, can be empty)
- email: User's email address

**Response format (JSON):**
{{
  "extracted_data": {{
    "field_name": "extracted_value or null if not found"
  }},
  "next_question": "Natural follow-up question asking for missing information. Be conversational and helpful. If suggesting milestones or RAID items, provide specific examples based on the program type."
}}

**Guidelines:**
- Extract as much as possible from the user's message
- For milestones: suggest 3-5 specific milestones based on program type if user asks
- For RAID items: suggest relevant risks/dependencies if user asks
- For budget: extract just the number (e.g., "500000" from "$500,000")
- For dates: convert to YYYY-MM-DD format
- Be encouraging and professional
- If user provides everything at once, acknowledge it and ask for email if missing"""

        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"AI extraction error: {e}")
            return {
                "extracted_data": {},
                "next_question": "I had trouble processing that. Could you please rephrase or provide the information in a different way?"
            }

    async def _handle_completion(self, state: dict, message: str) -> str:
        """Handle completion phase - user wants to start over"""
        # Reset state
        state["phase"] = "intro"
        state["data"] = {}
        state["conversation_history"] = []
        return await self._handle_intro(state, message)

    async def _prepare_submission(self, state: dict) -> str:
        """Prepare data for webhook submission"""
        
        data = state["data"]
        
        # Prepare webhook payload
        webhook_data = {
            "program_name": data["program_name"],
            "program_manager": data["program_manager"],
            "executive_sponsor": data["executive_sponsor"],
            "goals": data["goals"],
            "budget": int(data["budget"]) if isinstance(data["budget"], str) else data["budget"],
            "start_date": data["start_date"],
            "end_date": data["end_date"],
            "milestones": data["milestones"] if isinstance(data["milestones"], list) else [data["milestones"]],
            "raid_items": data["raid_items"] if isinstance(data["raid_items"], list) else ([data["raid_items"]] if data["raid_items"] else []),
            "email": data["email"]
        }
        
        # Submit to webhook
        success = await self._submit_to_webhook(webhook_data)
        
        if success:
            return f"""✅ **Perfect! I have everything I need.**

**Submitting your program details now...**

📊 **5 PowerPoint reports are being generated:**
1. Program Charter
2. Status Report
3. Milestone Tracker
4. RAID Log
5. Budget Overview

They'll be delivered to **{data["email"]}** within 2-3 minutes.

---

**Program Summary:**
- **Name**: {data["program_name"]}
- **Manager**: {data["program_manager"]}
- **Sponsor**: {data["executive_sponsor"]}
- **Budget**: ${webhook_data["budget"]:,}
- **Duration**: {data["start_date"]} to {data["end_date"]}
- **Milestones**: {len(webhook_data["milestones"])}
- **RAID Items**: {len(webhook_data["raid_items"])}

Check your inbox! 📧

---

Want to create another program? Just say "hello" to start over."""
        else:
            return f"""❌ **Oops! There was an error submitting to the webhook.**

Please try again or contact support. Here's your data:

```json
{json.dumps(webhook_data, indent=2)}
```"""

    async def _submit_to_webhook(self, data: dict) -> bool:
        """Submit program data to webhook"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(WEBHOOK_URL, json=data)
                return response.status_code == 200
        except Exception as e:
            print(f"Webhook submission error: {e}")
            return False

    async def get_settings(self, setting: fp.SettingsRequest) -> fp.SettingsResponse:
        """Configure bot settings"""
        return fp.SettingsResponse(
            server_bot_dependencies={"GPT-4o-mini": 1},
            introduction_message="👋 Hi! I'm your AI-powered PMO consultant. I'll help you scope your program and deliver professional PowerPoint reports to your inbox. Tell me about your program!"
        )


# Create the bot instance
bot = PMOConsultantBot()

# Create FastAPI app
app = fp.make_app(bot, allow_without_key=True)

