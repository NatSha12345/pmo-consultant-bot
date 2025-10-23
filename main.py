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

**You can provide information in any way you like:**
- Answer my questions one by one, OR
- Give me all the details at once, OR
- Just chat naturally - I'll figure it out!

**I need to collect information about your program:**
1. **Program basics**: Name, manager, sponsor
2. **Timeline**: Start and end dates  
3. **Budget**: Total budget and amount spent so far
4. **Status**: Overall status (On Track/At Risk/Off Track) and commentary
5. **Progress**: Key accomplishments and upcoming milestones
6. **RAID items**: Risks, Assumptions, Issues, Dependencies
7. **Your email**: Where to send the reports

**Let's get started! Tell me about your program.** You can share as much or as little as you want, and I'll ask follow-up questions for anything missing."""

    async def _handle_collection(self, state: dict, message: str, request: fp.QueryRequest) -> AsyncIterable[fp.PartialResponse]:
        """Handle data collection phase with AI intelligence using Poe's Claude"""
        
        # Use Poe's Claude to extract information and determine next steps
        extraction_result = await self._extract_data_with_poe_ai(state, message, request)
        
        # Update state with extracted data
        for key, value in extraction_result["extracted_data"].items():
            if value and value != "unknown" and value != "null" and value != None:
                state["data"][key] = value
        
        # Check if we have everything
        required_fields = [
            "program_name", "program_manager", "program_manager_email",
            "sponsor_name", "update_date", "update_title",
            "key_accomplishments", "upcoming_milestones",
            "total_budget", "budget_spent", "schedule_variance",
            "overall_status", "status_commentary",
            "open_risks", "open_assumptions", "open_issues", "open_dependencies"
        ]
        
        missing_fields = [f for f in required_fields if f not in state["data"] or not state["data"][f]]
        
        if not missing_fields:
            # All data collected - prepare for submission
            state["phase"] = "complete"
            response = await self._prepare_submission(state)
            yield fp.PartialResponse(text=response)
        else:
            # Still need more information
            yield fp.PartialResponse(text=extraction_result["next_question"])

    async def _extract_data_with_poe_ai(self, state: dict, message: str, request: fp.QueryRequest) -> dict:
        """Use Poe's Claude to intelligently extract data and generate next question"""
        
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
- program_manager_email: Email address of program manager
- sponsor_name: Full name of executive sponsor
- update_date: Current date (YYYY-MM-DD format, use today's date: 2024-10-23 if not specified)
- update_title: Title for this update (e.g., "Q4 2024 Update")
- key_accomplishments: Recent achievements (string, can be bullet points)
- upcoming_milestones: Future milestones (string, can be bullet points)
- total_budget: Total program budget (integer, just the number)
- budget_spent: Amount spent so far (integer, just the number)
- schedule_variance: Days ahead/behind schedule (integer, positive or negative, use 0 if on track)
- overall_status: "On Track", "At Risk", or "Off Track"
- status_commentary: Explanation of current status (string)
- open_risks: Number of open risks (integer)
- open_assumptions: Number of open assumptions (integer)
- open_issues: Number of open issues (integer)
- open_dependencies: Number of open dependencies (integer)

**User's message:** {message}

**Response format (JSON ONLY, no other text):**
{{
  "extracted_data": {{
    "field_name": "extracted_value or null if not found"
  }},
  "next_question": "Natural follow-up question asking for missing information. Be conversational and helpful. Acknowledge what they've provided. If many fields are missing, ask for 2-3 related items at once."
}}

**Guidelines:**
- Extract as much as possible from the user's message
- For budget: extract just the number (e.g., "500000" from "$500,000")
- For dates: convert to YYYY-MM-DD format
- For RAID counts: if user says "no risks" or "none", set to 0
- If user provides general info, make reasonable assumptions (e.g., if they say "just started", budget_spent could be low)
- Be encouraging and professional
- Group related questions together (e.g., ask for all RAID counts at once)

Respond ONLY with valid JSON, no other text."""

        try:
            # Get the last user message
            last_msg = request.query[-1]
            
            # Create a new message with the system prompt
            new_msg = last_msg.model_copy(update={"content": system_prompt})
            
            # Create a new request with the modified message
            new_request = request.model_copy(update={"query": [new_msg]})
            
            # Use Poe's stream_request to call Claude
            extraction_response = ""
            async for msg in fp.stream_request(
                new_request, "Claude-Sonnet-4.5", request.access_key
            ):
                if isinstance(msg, fp.PartialResponse):
                    extraction_response += msg.text
            
            # Parse JSON from response
            # Sometimes Claude adds markdown formatting, so clean it
            extraction_response = extraction_response.strip()
            if extraction_response.startswith("```json"):
                extraction_response = extraction_response[7:]
            if extraction_response.startswith("```"):
                extraction_response = extraction_response[3:]
            if extraction_response.endswith("```"):
                extraction_response = extraction_response[:-3]
            extraction_response = extraction_response.strip()
            
            result = json.loads(extraction_response)
            return result
            
        except Exception as e:
            print(f"AI extraction error: {e}")
            print(f"Response was: {extraction_response if 'extraction_response' in locals() else 'No response'}")
            return {
                "extracted_data": {},
                "next_question": "I had trouble processing that. Could you please provide some basic information about your program: the program name, your name as program manager, and your email address?"
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
        
        # Prepare webhook payload with all required fields
        webhook_data = {
            "program_name": data.get("program_name", "Unknown Program"),
            "program_manager": data.get("program_manager", "Unknown Manager"),
            "program_manager_email": data.get("program_manager_email", ""),
            "sponsor_name": data.get("sponsor_name", "Unknown Sponsor"),
            "update_date": data.get("update_date", "2024-10-23"),
            "update_title": data.get("update_title", "Program Update"),
            "key_accomplishments": data.get("key_accomplishments", "None reported"),
            "upcoming_milestones": data.get("upcoming_milestones", "None reported"),
            "total_budget": int(data.get("total_budget", 0)) if str(data.get("total_budget", 0)).replace("-","").isdigit() else 0,
            "budget_spent": int(data.get("budget_spent", 0)) if str(data.get("budget_spent", 0)).replace("-","").isdigit() else 0,
            "schedule_variance": int(data.get("schedule_variance", 0)) if str(data.get("schedule_variance", 0)).replace("-","").isdigit() else 0,
            "overall_status": data.get("overall_status", "On Track"),
            "status_commentary": data.get("status_commentary", "No commentary provided"),
            "raid_counts": {
                "risks": int(data.get("open_risks", 0)) if str(data.get("open_risks", 0)).isdigit() else 0,
                "assumptions": int(data.get("open_assumptions", 0)) if str(data.get("open_assumptions", 0)).isdigit() else 0,
                "issues": int(data.get("open_issues", 0)) if str(data.get("open_issues", 0)).isdigit() else 0,
                "dependencies": int(data.get("open_dependencies", 0)) if str(data.get("open_dependencies", 0)).isdigit() else 0
            },
            "open_risks": int(data.get("open_risks", 0)) if str(data.get("open_risks", 0)).isdigit() else 0,
            "open_assumptions": int(data.get("open_assumptions", 0)) if str(data.get("open_assumptions", 0)).isdigit() else 0,
            "open_issues": int(data.get("open_issues", 0)) if str(data.get("open_issues", 0)).isdigit() else 0,
            "open_dependencies": int(data.get("open_dependencies", 0)) if str(data.get("open_dependencies", 0)).isdigit() else 0
        }
        
        # Submit to webhook
        success = await self._submit_to_webhook(webhook_data)
        
        if success:
            return f"""✅ **Perfect! I have everything I need.**

**Submitting your program details now...**

📊 **5 PowerPoint reports are being generated:**
1. Steering Committee Pack
2. Program Status Report
3. RAID Log
4. Budget Tracking Dashboard
5. Milestone Tracker

They'll be delivered to **{webhook_data["program_manager_email"]}** within 2-3 minutes.

---

**Program Summary:**
- **Name**: {webhook_data["program_name"]}
- **Manager**: {webhook_data["program_manager"]}
- **Sponsor**: {webhook_data["sponsor_name"]}
- **Budget**: ${webhook_data["total_budget"]:,} (${webhook_data["budget_spent"]:,} spent)
- **Status**: {webhook_data["overall_status"]}
- **RAID**: {webhook_data["open_risks"]} risks, {webhook_data["open_issues"]} issues, {webhook_data["open_assumptions"]} assumptions, {webhook_data["open_dependencies"]} dependencies

Check your inbox! 📧

---

Want to create another program? Just say "hello" to start over."""
        else:
            return f"""❌ **Oops! There was an error submitting to the webhook.**

Please try again or contact support. Here's your data for reference:

```json
{json.dumps(webhook_data, indent=2)}
```

Type "hello" to start over."""

    async def _submit_to_webhook(self, data: dict) -> bool:
        """Submit program data to webhook"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(WEBHOOK_URL, json=data)
                print(f"Webhook response: {response.status_code}")
                print(f"Webhook response body: {response.text}")
                return response.status_code == 200
        except Exception as e:
            print(f"Webhook submission error: {e}")
            return False

    async def get_settings(self, setting: fp.SettingsRequest) -> fp.SettingsResponse:
        """Configure bot settings"""
        return fp.SettingsResponse(
            server_bot_dependencies={"Claude-Sonnet-4.5": 1},
            introduction_message="👋 Hi! I'm your AI-powered PMO consultant. I'll help you create a comprehensive program plan and deliver 5 professional PowerPoint reports to your inbox. Just say 'hello' to get started!"
        )


# Create the bot instance
bot = PMOConsultantBot()

# Create FastAPI app
app = fp.make_app(bot, allow_without_key=True)

