"""
PMO Expert Consultant - AI-Powered Poe Server Bot
Deployed on Render.com
Uses Poe's Claude-Sonnet-4.5 for AI intelligence
"""

from __future__ import annotations
from typing import AsyncIterable
import fastapi_poe as fp
import json
import httpx
import re

# Webhook endpoint
WEBHOOK_URL = "https://natsha.pythonanywhere.com/webhook/fortnightly-update-direct"

# Conversation state storage
conversation_states = {}


class PMOConsultantBot(fp.PoeBot):
    """
    AI-powered PMO consultant that intelligently collects program information
    and generates PowerPoint reports using Poe's Claude-Sonnet-4.5
    """

    async def get_response(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        """Main response handler with AI intelligence"""
        
        user_id = request.user_id
        user_message = request.query[-1].content.strip()
        
        # Initialize state if new user
        if user_id not in conversation_states:
            conversation_states[user_id] = {
                "phase": "intro",
                "data": {
                    "schedule_variance": 0  # Default value
                },
                "conversation_history": [],
                "current_question": None
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
            yield fp.PartialResponse(text=response)
        elif state["phase"] == "collecting":
            async for partial in self._handle_collection(state, user_message, request):
                yield partial
        elif state["phase"] == "complete":
            response = await self._handle_completion(state, user_message)
            yield fp.PartialResponse(text=response)
        else:
            response = "I'm not sure what to do. Let's start over. Type 'hello' to begin."
            state["phase"] = "intro"
            yield fp.PartialResponse(text=response)

    async def _handle_intro(self, state: dict, message: str) -> str:
        """Handle introduction phase"""
        state["phase"] = "collecting"
        return """👋 Hi! I'm your AI-powered PMO consultant. I'll help you create a comprehensive program plan and deliver 5 professional PowerPoint reports to your inbox.

**Let's get started! Tell me about your program.** You can share as much or as little as you want, and I'll ask follow-up questions for anything missing.

**I need to know:**
- Program name, manager, sponsor, and your email
- Budget information
- Status and progress
- RAID items (Risks, Assumptions, Issues, Dependencies)

Go ahead and tell me what you know!"""

    def _extract_simple_data(self, state: dict, message: str):
        """Extract data using simple pattern matching"""
        msg_lower = message.lower()
        data = state["data"]
        
        # Extract email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', message)
        if email_match and "program_manager_email" not in data:
            data["program_manager_email"] = email_match.group(0)
        
        # Extract numbers for RAID counts when explicitly asked
        current_q = state.get("current_question", "") or ""
        
        # Look for standalone numbers (like "0", "5", "10")
        number_match = re.search(r'\b(\d+)\b', message)
        
        if "risk" in current_q.lower() and number_match:
            data["open_risks"] = int(number_match.group(1))
        elif "assumption" in current_q.lower() and number_match:
            data["open_assumptions"] = int(number_match.group(1))
        elif "issue" in current_q.lower() and number_match:
            data["open_issues"] = int(number_match.group(1))
        elif "dependenc" in current_q.lower() and number_match:
            data["open_dependencies"] = int(number_match.group(1))
        
        # Check for "none", "zero", "no" responses
        if any(word in msg_lower for word in ["none", "zero", "no ", "0"]):
            if "risk" in current_q.lower() and "open_risks" not in data:
                data["open_risks"] = 0
            if "assumption" in current_q.lower() and "open_assumptions" not in data:
                data["open_assumptions"] = 0
            if "issue" in current_q.lower() and "open_issues" not in data:
                data["open_issues"] = 0
            if "dependenc" in current_q.lower() and "open_dependencies" not in data:
                data["open_dependencies"] = 0
        
        # Extract budget numbers
        if "budget" in current_q.lower() or "budget" in msg_lower:
            # Find all numbers in the message
            numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', message.replace(',', ''))
            if numbers:
                # Convert to integers
                nums = [int(float(n)) for n in numbers]
                if "total" in msg_lower or "total_budget" not in data:
                    if len(nums) >= 1 and "total_budget" not in data:
                        data["total_budget"] = nums[0]
                    if len(nums) >= 2 and "budget_spent" not in data:
                        data["budget_spent"] = nums[1]
        
        # Extract status
        if "on track" in msg_lower:
            data["overall_status"] = "On Track"
        elif "at risk" in msg_lower:
            data["overall_status"] = "At Risk"
        elif "off track" in msg_lower:
            data["overall_status"] = "Off Track"

    async def _handle_collection(self, state: dict, message: str, request: fp.QueryRequest) -> AsyncIterable[fp.PartialResponse]:
        """Handle data collection phase"""
        
        # First, try to extract data using simple patterns
        self._extract_simple_data(state, message)
        
        # Then use Claude to extract remaining data and ask next question
        extraction_result = await self._get_claude_help(state, message, request)
        
        # Update data from Claude's extraction
        if extraction_result and "extracted_data" in extraction_result:
            for key, value in extraction_result["extracted_data"].items():
                if value and str(value).lower() not in ["unknown", "null", "none", "", "n/a"]:
                    # Only update if not already set
                    if key not in state["data"] or not state["data"][key]:
                        state["data"][key] = value
        
        # Check completion
        required_fields = [
            "program_name", "program_manager", "program_manager_email",
            "sponsor_name", "update_date", "update_title",
            "key_accomplishments", "upcoming_milestones",
            "total_budget", "budget_spent",
            "overall_status", "status_commentary",
            "open_risks", "open_assumptions", "open_issues", "open_dependencies"
        ]
        
        missing_fields = [f for f in required_fields if f not in state["data"] or not state["data"][f]]
        
        print(f"Current data: {json.dumps(state['data'], indent=2)}")
        print(f"Missing fields: {missing_fields}")
        
        if not missing_fields:
            # All data collected
            state["phase"] = "complete"
            response = await self._prepare_submission(state)
            yield fp.PartialResponse(text=response)
        else:
            # Ask next question
            next_q = extraction_result.get("next_question", "Could you provide more details?")
            state["current_question"] = next_q
            yield fp.PartialResponse(text=next_q)

    async def _get_claude_help(self, state: dict, message: str, request: fp.QueryRequest) -> dict:
        """Get Claude's help with extraction and next question"""
        
        current_data = state["data"]
        
        prompt = f"""You are helping collect program information. 

**Data we have so far:**
```json
{json.dumps(current_data, indent=2)}
```

**User just said:** "{message}"

**Required fields we still need:**
- program_name, program_manager, program_manager_email, sponsor_name
- update_date (use 2024-10-23 if not specified), update_title
- key_accomplishments, upcoming_milestones
- total_budget, budget_spent (integers only)
- overall_status ("On Track", "At Risk", or "Off Track")
- status_commentary
- open_risks, open_assumptions, open_issues, open_dependencies (integers, 0 if none)

**Your task:**
1. Extract any NEW information from the user's message
2. Determine what's STILL missing
3. Generate ONE clear question for the next missing item

**CRITICAL: Respond ONLY with valid JSON in this exact format:**
```json
{{
  "extracted_data": {{
    "field_name": "value"
  }},
  "next_question": "Your question here"
}}
```

**Rules:**
- If user said a number, extract it
- If user said "none" or "zero", use 0
- Ask for ONE thing at a time (or 2-3 related items)
- Be conversational and encouraging

Respond with ONLY the JSON, nothing else:"""

        try:
            last_msg = request.query[-1]
            new_msg = last_msg.model_copy(update={"content": prompt})
            new_request = request.model_copy(update={"query": [new_msg]})
            
            response_text = ""
            async for msg in fp.stream_request(new_request, "Claude-Sonnet-4.5", request.access_key):
                if isinstance(msg, fp.PartialResponse):
                    response_text += msg.text
            
            # Clean response
            response_text = response_text.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                parts = response_text.split("```")
                if len(parts) >= 2:
                    response_text = parts[1].strip()
            
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            
            result = json.loads(response_text)
            return result
            
        except Exception as e:
            print(f"Claude help error: {e}")
            print(f"Response: {response_text if 'response_text' in locals() else 'none'}")
            
            # Fallback: determine next question manually
            if "program_name" not in current_data:
                return {"extracted_data": {}, "next_question": "What's the name of your program?"}
            elif "program_manager" not in current_data:
                return {"extracted_data": {}, "next_question": "What's your name (as the program manager)?"}
            elif "program_manager_email" not in current_data:
                return {"extracted_data": {}, "next_question": "What's your email address?"}
            elif "sponsor_name" not in current_data:
                return {"extracted_data": {}, "next_question": "Who is the executive sponsor for this program?"}
            elif "overall_status" not in current_data:
                return {"extracted_data": {}, "next_question": "What's the overall program status? (On Track / At Risk / Off Track)"}
            elif "status_commentary" not in current_data:
                return {"extracted_data": {}, "next_question": "Can you provide a brief commentary on the program status?"}
            elif "key_accomplishments" not in current_data:
                return {"extracted_data": {}, "next_question": "What are the key accomplishments so far?"}
            elif "upcoming_milestones" not in current_data:
                return {"extracted_data": {}, "next_question": "What are the upcoming milestones?"}
            elif "total_budget" not in current_data:
                return {"extracted_data": {}, "next_question": "What's the total program budget? (just the number)"}
            elif "budget_spent" not in current_data:
                return {"extracted_data": {}, "next_question": "How much of the budget has been spent so far?"}
            elif "open_risks" not in current_data:
                return {"extracted_data": {}, "next_question": "How many open risks are there? (enter a number, or 0 if none)"}
            elif "open_issues" not in current_data:
                return {"extracted_data": {}, "next_question": "How many open issues? (enter a number, or 0 if none)"}
            elif "open_assumptions" not in current_data:
                return {"extracted_data": {}, "next_question": "How many assumptions? (enter a number, or 0 if none)"}
            elif "open_dependencies" not in current_data:
                return {"extracted_data": {}, "next_question": "How many dependencies? (enter a number, or 0 if none)"}
            else:
                return {"extracted_data": {}, "next_question": "Let me review what we have..."}

    async def _handle_completion(self, state: dict, message: str) -> str:
        """Handle completion phase"""
        state["phase"] = "intro"
        state["data"] = {"schedule_variance": 0}
        state["conversation_history"] = []
        return await self._handle_intro(state, message)

    async def _prepare_submission(self, state: dict) -> str:
        """Prepare and submit data"""
        data = state["data"]
        
        # Set defaults
        if "update_date" not in data:
            data["update_date"] = "2024-10-23"
        if "update_title" not in data:
            data["update_title"] = "Program Update"
        
        webhook_data = {
            "program_name": data.get("program_name", "Unknown"),
            "program_manager": data.get("program_manager", "Unknown"),
            "program_manager_email": data.get("program_manager_email", ""),
            "sponsor_name": data.get("sponsor_name", "Unknown"),
            "update_date": data.get("update_date"),
            "update_title": data.get("update_title"),
            "key_accomplishments": data.get("key_accomplishments", "None"),
            "upcoming_milestones": data.get("upcoming_milestones", "None"),
            "total_budget": int(data.get("total_budget", 0)),
            "budget_spent": int(data.get("budget_spent", 0)),
            "schedule_variance": 0,
            "overall_status": data.get("overall_status", "On Track"),
            "status_commentary": data.get("status_commentary", "No commentary"),
            "raid_counts": {
                "risks": int(data.get("open_risks", 0)),
                "assumptions": int(data.get("open_assumptions", 0)),
                "issues": int(data.get("open_issues", 0)),
                "dependencies": int(data.get("open_dependencies", 0))
            },
            "open_risks": int(data.get("open_risks", 0)),
            "open_assumptions": int(data.get("open_assumptions", 0)),
            "open_issues": int(data.get("open_issues", 0)),
            "open_dependencies": int(data.get("open_dependencies", 0))
        }
        
        success = await self._submit_to_webhook(webhook_data)
        
        if success:
            return f"""✅ **Perfect! Submitting your program details now...**

📊 **5 PowerPoint reports are being generated and will be sent to {webhook_data["program_manager_email"]}**

**Program Summary:**
- **Name**: {webhook_data["program_name"]}
- **Manager**: {webhook_data["program_manager"]}
- **Budget**: ${webhook_data["total_budget"]:,} (${webhook_data["budget_spent"]:,} spent)
- **Status**: {webhook_data["overall_status"]}
- **RAID**: {webhook_data["open_risks"]} risks, {webhook_data["open_issues"]} issues, {webhook_data["open_assumptions"]} assumptions, {webhook_data["open_dependencies"]} dependencies

Check your inbox in 2-3 minutes! 📧

Type "hello" to create another program update."""
        else:
            return "❌ Error submitting to webhook. Please try again or contact support."

    async def _submit_to_webhook(self, data: dict) -> bool:
        """Submit to webhook"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(WEBHOOK_URL, json=data)
                print(f"Webhook: {response.status_code} - {response.text}")
                return response.status_code == 200
        except Exception as e:
            print(f"Webhook error: {e}")
            return False

    async def get_settings(self, setting: fp.SettingsRequest) -> fp.SettingsResponse:
        """Bot settings"""
        return fp.SettingsResponse(
            server_bot_dependencies={"Claude-Sonnet-4.5": 1},
            introduction_message="👋 Hi! I'm your AI PMO consultant. Say 'hello' to create program reports!"
        )


bot = PMOConsultantBot()
app = fp.make_app(bot, allow_without_key=True)

