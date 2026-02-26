# Multi-Agent Framework for Tau-Bench
# Architecture:
#   Orchestrator (LLM)
#     ├── Tool Navigator (LLM)
#     ├── Execution & State Tracker (LLM)
#     └── Response Synthesizer (LLM)
#           └── Deterministic Enforcement Layer (Pure Code)

import json
import re
from typing import List, Optional, Dict, Any, Tuple

from litellm import completion
from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME, RESPOND_ACTION_FIELD_NAME


# ==============================================================================
# DETERMINISTIC ENFORCEMENT LAYER (Pure Code — No LLM)
# ==============================================================================

class DeterministicEnforcementLayer:
    """
    Pure code enforcement of policy rules.
    No LLM involved. Hard gates that cannot be reasoned around.
    """

    # Keywords that indicate explicit user confirmation
    CONFIRM_KEYWORDS = [
        "yes", "yeah", "yep", "confirm", "confirmed", "go ahead", "proceed",
        "do it", "book it", "sure", "absolutely", "correct", "that's right",
        "sounds good", "ok", "okay", "please do", "yes please", "affirmative",
        "that works", "looks good", "approve", "approved"
    ]

    # Keywords that indicate negation / NOT a confirmation
    DENY_KEYWORDS = [
        "no", "nope", "don't", "do not", "cancel", "stop", "wait",
        "hold on", "not yet", "change", "different", "wrong", "incorrect"
    ]

    def __init__(self):
        self.authenticated = False
        self.certificates_used = 0
        self.MAX_CERTIFICATES = 1
        self.pending_confirmation = False
        self.state_ledger: Dict[str, Any] = {}

    def reset(self):
        """Reset state for a new task."""
        self.authenticated = False
        self.certificates_used = 0
        self.pending_confirmation = False
        self.state_ledger = {}

    # ------------------------------------------------------------------
    # Policy Rule Validators
    # ------------------------------------------------------------------

    def check_authentication(self) -> Tuple[bool, str]:
        """Rule: User must be authenticated before accessing orders/bookings."""
        if not self.authenticated:
            return False, "BLOCK: User not authenticated. Must call find_user_id or equivalent first."
        return True, "OK"

    def check_certificate_limit(self, tool_name: str, kwargs: Dict[str, Any]) -> Tuple[bool, str]:
        """Rule: At most 1 travel certificate per reservation."""
        if "certificate" in str(kwargs).lower() or "voucher" in str(kwargs).lower():
            if self.certificates_used >= self.MAX_CERTIFICATES:
                return False, f"BLOCK: Certificate limit reached. Max {self.MAX_CERTIFICATES} certificate per reservation."
        return True, "OK"

    def check_time_constraint(self, kwargs: Dict[str, Any], constraint_after_hour: Optional[int] = None) -> Tuple[bool, str]:
        """Rule: Enforce departure time constraints if present."""
        if constraint_after_hour is None:
            return True, "OK"
        time_fields = ["departure_time", "depart_time", "time", "flight_time"]
        for field in time_fields:
            if field in kwargs:
                val = str(kwargs[field])
                hour_match = re.search(r"(\d{1,2}):(\d{2})\s*(AM|PM)?", val, re.IGNORECASE)
                if hour_match:
                    hour = int(hour_match.group(1))
                    meridiem = hour_match.group(3)
                    if meridiem and meridiem.upper() == "PM" and hour != 12:
                        hour += 12
                    if hour < constraint_after_hour:
                        return False, f"BLOCK: Departure time {val} violates after-{constraint_after_hour}:00 constraint."
        return True, "OK"

    def check_payment_source(self, kwargs: Dict[str, Any]) -> Tuple[bool, str]:
        """Rule: Payment IDs must come from user profile, not invented."""
        if "payment_id" in kwargs:
            pid = str(kwargs["payment_id"])
            # Payment IDs from user profile follow real patterns, not placeholder text
            if pid.lower() in ["string", "none", "null", "payment_id", "id", "xxx", "unknown"]:
                return False, f"BLOCK: Invalid payment_id '{pid}'. Must use real payment ID from user profile."
        return True, "OK"

    def is_write_operation(self, tool_name: str) -> bool:
        """Identify tools that modify state and require explicit confirmation."""
        write_tools = [
            "book_reservation", "update_reservation", "cancel_reservation",
            "exchange_delivered_order_items", "return_delivered_order_items",
            "book_flight", "modify_flight", "cancel_flight",
            "update_reservation_baggages", "apply_certificate",
            "update_flight", "create_booking", "modify_booking",
        ]
        return any(w in tool_name.lower() for w in write_tools)

    def is_auth_tool(self, tool_name: str) -> bool:
        """Identify tools that perform authentication."""
        auth_tools = [
            "find_user_id", "get_user_details", "verify_user",
            "authenticate", "login", "find_user_id_by_name_zip",
            "find_user_id_by_email"
        ]
        return any(a in tool_name.lower() for a in auth_tools)

    # ------------------------------------------------------------------
    # Confirmation Classifier (Pure Code)
    # ------------------------------------------------------------------

    def classify_confirmation(self, user_message: str) -> bool:
        """
        Deterministic confirmation classifier.
        Returns True only if explicit confirmation detected.
        """
        msg = user_message.lower().strip()

        # Check for denial first (higher priority)
        for deny in self.DENY_KEYWORDS:
            if re.search(r'\b' + re.escape(deny) + r'\b', msg):
                return False

        # Check for confirmation
        for confirm in self.CONFIRM_KEYWORDS:
            if re.search(r'\b' + re.escape(confirm) + r'\b', msg):
                return True

        return False

    # ------------------------------------------------------------------
    # Schema Validator
    # ------------------------------------------------------------------

    def validate_tool_schema(self, tool_name: str, kwargs: Dict[str, Any],
                              tools_info: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Validate tool arguments against the known tool schema.
        Blocks calls with missing required fields or hallucinated fields.
        """
        # Find the tool schema
        tool_schema = None
        for tool in tools_info:
            if tool.get("function", {}).get("name") == tool_name:
                tool_schema = tool["function"].get("parameters", {})
                break
            elif tool.get("name") == tool_name:
                tool_schema = tool.get("parameters", {})
                break

        if tool_schema is None:
            return False, f"BLOCK: Tool '{tool_name}' not found in available tools."

        required_fields = tool_schema.get("required", [])
        properties = tool_schema.get("properties", {})

        # Check required fields are present
        for field in required_fields:
            if field not in kwargs:
                return False, f"BLOCK: Required field '{field}' missing for tool '{tool_name}'."

        # Check for hallucinated fields not in schema
        if properties:
            for field in kwargs:
                if field not in properties:
                    return False, f"BLOCK: Field '{field}' does not exist in tool '{tool_name}' schema."

        return True, "OK"

    # ------------------------------------------------------------------
    # Master Gate — runs all checks before any tool call
    # ------------------------------------------------------------------

    def gate(self, tool_name: str, kwargs: Dict[str, Any],
             tools_info: List[Dict[str, Any]],
             user_confirmed: bool = False) -> Tuple[bool, str]:
        """
        Master enforcement gate. Called before every tool execution.
        Returns (proceed: bool, reason: str).
        """
        # 1. Schema validation
        schema_ok, schema_msg = self.validate_tool_schema(tool_name, kwargs, tools_info)
        if not schema_ok:
            return False, schema_msg

        # 2. Authentication check (skip for auth tools themselves)
        if not self.is_auth_tool(tool_name):
            # Only enforce auth for order/booking access tools
            order_tools = ["get_order", "book_", "update_", "cancel_", "exchange_", "return_"]
            needs_auth = any(tool_name.lower().startswith(t) or t in tool_name.lower()
                           for t in order_tools)
            if needs_auth:
                auth_ok, auth_msg = self.check_authentication()
                if not auth_ok:
                    return False, auth_msg

        # 3. Write operation confirmation check
        if self.is_write_operation(tool_name) and not user_confirmed:
            return False, "BLOCK: Write operation requires explicit user confirmation first."

        # 4. Certificate limit
        cert_ok, cert_msg = self.check_certificate_limit(tool_name, kwargs)
        if not cert_ok:
            return False, cert_msg

        # 5. Payment source validation
        pay_ok, pay_msg = self.check_payment_source(kwargs)
        if not pay_ok:
            return False, pay_msg

        return True, "OK"

    # ------------------------------------------------------------------
    # State Ledger
    # ------------------------------------------------------------------

    def record(self, key: str, value: Any):
        """Record a verified fact into the state ledger."""
        self.state_ledger[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a fact from the state ledger."""
        return self.state_ledger.get(key, default)

    def ledger_summary(self) -> str:
        """Return a human-readable summary of the current state ledger."""
        if not self.state_ledger:
            return "State ledger: empty"
        lines = ["=== STATE LEDGER (verified facts only) ==="]
        for k, v in self.state_ledger.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ==============================================================================
# LLM CALL HELPER
# ==============================================================================

def llm_call(model: str, provider: str, messages: List[Dict[str, Any]],
             temperature: float = 0.0,
             tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, Any]:
    """
    Unified LLM call wrapper.
    Returns (text_content, raw_message).
    """
    kwargs = dict(
        model=model,
        custom_llm_provider=provider,
        messages=messages,
        temperature=temperature,
    )
    if tools:
        kwargs["tools"] = tools

    res = completion(**kwargs)
    message = res.choices[0].message
    content = message.content or ""
    return content, message.model_dump()


# ==============================================================================
# AGENT 1: ORCHESTRATOR
# ==============================================================================

ORCHESTRATOR_SYSTEM = """You are the Orchestrator in a multi-agent customer service system.

Your job:
1. Understand the user's request
2. Decompose it into an ordered list of subtasks
3. Decide what needs to happen next based on conversation history and the state ledger
4. Output a JSON plan for the current step

You have access to a state ledger that contains only verified facts from tool calls.
You must NEVER invent facts not in the ledger or conversation.

Output format (always valid JSON):
{{
  "interpretation": "<one sentence: what the user wants>",
  "next_action": "tool_call" | "respond_to_user" | "request_confirmation",
  "reasoning": "<why this next action>",
  "task_description": "<if tool_call: describe what tool should be called and why>",
  "response_needed": "<if respond_to_user: what information to convey>"
}}

Rules:
- If authentication has not happened and user asks about orders/bookings, next_action must be tool_call to authenticate
- If a write operation is needed, next_action must be request_confirmation BEFORE tool_call
- Never claim a booking is done unless the state ledger confirms it
- If you are unsure of the user's intent, next_action = respond_to_user to ask clarification
- If the user has ALREADY confirmed (said yes/ok/proceed/confirmed) in the last message, next_action must be tool_call immediately - do NOT ask for confirmation again
- If read-only tools are needed (get_order_details, find_user_id, get_product_details, search flights), next_action = tool_call directly without confirmation
- Only request_confirmation for irreversible write operations (exchange, cancel, book, return) and only ONCE
"""

def run_orchestrator(model: str, provider: str, temperature: float,
                     wiki: str, conversation_history: List[Dict[str, Any]],
                     ledger_summary: str) -> Dict[str, Any]:
    """Run the Orchestrator agent."""
    messages = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM + f"\n\n{wiki}"},
        {"role": "user", "content": f"{ledger_summary}\n\nConversation so far:\n" +
         _format_history(conversation_history) +
         "\n\nWhat is the next action? Respond in JSON only."}
    ]
    content, _ = llm_call(model, provider, messages, temperature)
    try:
        # Strip markdown code fences if present
        clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        clean = re.sub(r"```(?:json)?|```", "", clean).strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        # Fallback: treat as respond_to_user
        return {
            "interpretation": "unclear",
            "next_action": "respond_to_user",
            "reasoning": "Could not parse orchestrator output",
            "response_needed": content
        }


# ==============================================================================
# AGENT 2: TOOL NAVIGATOR
# ==============================================================================

TOOL_NAVIGATOR_SYSTEM = """You are the Tool Navigator in a multi-agent customer service system.

Your ONLY job: Given a task description, select the correct tool and construct valid arguments.

Rules:
- You MUST select a tool from the provided tool list. Never invent tools.
- Arguments must come ONLY from the conversation history or state ledger. Never invent values.
- If required information is missing, output needs_info=true with what is missing.

Output format (always valid JSON):
{{
  "tool_name": "<exact tool name from the list>",
  "arguments": {{<argument key-value pairs>}},
  "needs_info": false,
  "missing_info": "<if needs_info=true: what is missing>"
}}
"""

def run_tool_navigator(model: str, provider: str, temperature: float,
                       task_description: str, tools_info: List[Dict[str, Any]],
                       conversation_history: List[Dict[str, Any]],
                       ledger_summary: str) -> Dict[str, Any]:
    """Run the Tool Navigator agent."""
    tools_manifest = json.dumps(tools_info, indent=2)
    messages = [
        {"role": "system", "content": TOOL_NAVIGATOR_SYSTEM},
        {"role": "user", "content": (
            f"Available tools:\n{tools_manifest}\n\n"
            f"{ledger_summary}\n\n"
            f"Conversation history:\n{_format_history(conversation_history)}\n\n"
            f"Task: {task_description}\n\n"
            f"Select the correct tool and construct arguments. Respond in JSON only."
        )}
    ]
    content, _ = llm_call(model, provider, messages, temperature)
    try:
        clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        clean = re.sub(r"```(?:json)?|```", "", clean).strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"tool_name": None, "arguments": {}, "needs_info": True,
                "missing_info": f"Tool Navigator failed to parse: {content}"}


# ==============================================================================
# AGENT 3: EXECUTION & STATE TRACKER
# ==============================================================================

EXECUTION_TRACKER_SYSTEM = """You are the Execution & State Tracker in a multi-agent customer service system.

Your job:
1. Receive the result of a tool call
2. Extract key facts from the result
3. Determine what important state should be recorded in the ledger
4. Summarize the result clearly

Output format (always valid JSON):
{{
  "summary": "<one or two sentence human-readable summary of what the tool returned>",
  "ledger_updates": {{
    "<key>": "<value>",
    ...
  }},
  "auth_completed": true | false,
  "booking_confirmed": true | false,
  "task_complete": true | false
}}

Rules:
- Only record facts that are explicitly present in the tool result
- Never invent or assume values
- If the tool returned an error, set task_complete=false and summarize the error
- auth_completed=true only if the tool result contains a valid user_id or equivalent
- booking_confirmed=true only if the tool result contains a reservation/booking ID
"""

def run_execution_tracker(model: str, provider: str, temperature: float,
                          tool_name: str, tool_result: str,
                          ledger_summary: str) -> Dict[str, Any]:
    """Run the Execution & State Tracker agent."""
    messages = [
        {"role": "system", "content": EXECUTION_TRACKER_SYSTEM},
        {"role": "user", "content": (
            f"Tool called: {tool_name}\n"
            f"Tool result: {tool_result}\n\n"
            f"Current ledger:\n{ledger_summary}\n\n"
            f"Extract facts and update the ledger. Respond in JSON only."
        )}
    ]
    content, _ = llm_call(model, provider, messages, temperature)
    try:
        clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        clean = re.sub(r"```(?:json)?|```", "", clean).strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return {
            "summary": tool_result,
            "ledger_updates": {},
            "auth_completed": False,
            "booking_confirmed": False,
            "task_complete": False
        }


# ==============================================================================
# AGENT 4: RESPONSE SYNTHESIZER
# ==============================================================================

RESPONSE_SYNTHESIZER_SYSTEM = """You are the Response Synthesizer in a multi-agent customer service system.

Your job: Generate the final user-facing response.

Rules:
- Only state facts that are in the state ledger or conversation history
- Never invent prices, IDs, times, or any other values
- Be clear, helpful, and concise
- If asking for confirmation before a write operation, clearly state what will happen and ask "shall I proceed?"
- If the task is complete, confirm what was done using only ledger-verified facts
"""

def run_response_synthesizer(model: str, provider: str, temperature: float,
                              response_goal: str,
                              conversation_history: List[Dict[str, Any]],
                              ledger_summary: str) -> str:
    """Run the Response Synthesizer agent."""
    messages = [
        {"role": "system", "content": RESPONSE_SYNTHESIZER_SYSTEM},
        {"role": "user", "content": (
            f"{ledger_summary}\n\n"
            f"Conversation history:\n{_format_history(conversation_history)}\n\n"
            f"Goal for this response: {response_goal}\n\n"
            f"Generate the response to the user now."
        )}
    ]
    content, _ = llm_call(model, provider, messages, temperature)
    return content.strip()


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _format_history(history: List[Dict[str, Any]]) -> str:
    """Format conversation history as readable text."""
    lines = []
    for msg in history[-10:]:  # Last 10 messages for context window management
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if content:
            lines.append(f"{role.upper()}: {content}")
        # Handle tool call messages
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                lines.append(f"TOOL_CALL: {fn.get('name')}({fn.get('arguments', '')})")
    return "\n".join(lines)


def _extract_user_message(conversation_history: List[Dict[str, Any]]) -> str:
    """Get the most recent user message."""
    for msg in reversed(conversation_history):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


# ==============================================================================
# MULTI-AGENT ORCHESTRATION LOOP
# ==============================================================================

class MultiAgentSystem(Agent):
    """
    Multi-agent architecture for Tau-Bench.

    Components:
    - Orchestrator: task decomposition and routing
    - Tool Navigator: tool selection and argument construction
    - Execution & State Tracker: tool execution and state management
    - Response Synthesizer: final user-facing response generation
    - Deterministic Enforcement Layer: hard policy enforcement (pure code)
    """

    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:

        # Initialize enforcement layer fresh for each task
        enforcer = DeterministicEnforcementLayer()

        # Reset environment
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        total_cost = 0.0

        # Conversation history (what tau-bench sees as the trajectory)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]

        # Internal working history for agents (richer context)
        agent_history: List[Dict[str, Any]] = [
            {"role": "user", "content": obs}
        ]

        # Track whether we are waiting for user confirmation
        awaiting_confirmation = False
        pending_tool_name = None
        pending_tool_kwargs = None
        max_retries = 3

        for step in range(max_num_steps):

            # ------------------------------------------------------------------
            # STEP A: Orchestrator decides next action
            # ------------------------------------------------------------------
            orchestrator_plan = run_orchestrator(
                model=self.model,
                provider=self.provider,
                temperature=self.temperature,
                wiki=self.wiki,
                conversation_history=agent_history,
                ledger_summary=enforcer.ledger_summary(),
            )

            next_action = orchestrator_plan.get("next_action", "respond_to_user")

            # ------------------------------------------------------------------
            # STEP B: Handle confirmation requests
            # ------------------------------------------------------------------
            if next_action == "request_confirmation" and not awaiting_confirmation:
                # Synthesize a confirmation request to the user
                confirm_response = run_response_synthesizer(
                    model=self.model,
                    provider=self.provider,
                    temperature=self.temperature,
                    response_goal=f"Ask user to confirm: {orchestrator_plan.get('task_description', 'this action')}",
                    conversation_history=agent_history,
                    ledger_summary=enforcer.ledger_summary(),
                )
                awaiting_confirmation = True

                # Send confirmation request to environment
                action = Action(
                    name=RESPOND_ACTION_NAME,
                    kwargs={RESPOND_ACTION_FIELD_NAME: confirm_response}
                )
                env_response = env.step(action)
                reward = env_response.reward
                info = {**info, **env_response.info.model_dump()}

                messages.extend([
                    {"role": "assistant", "content": confirm_response},
                    {"role": "user", "content": env_response.observation},
                ])
                agent_history.extend([
                    {"role": "assistant", "content": confirm_response},
                    {"role": "user", "content": env_response.observation},
                ])

                if env_response.done:
                    break
                continue

            # ------------------------------------------------------------------
            # STEP C: If awaiting confirmation, check user's response
            # ------------------------------------------------------------------
            if awaiting_confirmation:
                last_user_msg = _extract_user_message(agent_history)
                user_confirmed = enforcer.classify_confirmation(last_user_msg)

                if not user_confirmed:
                    # User did not confirm — ask orchestrator to re-plan
                    awaiting_confirmation = False
                    pending_tool_name = None
                    pending_tool_kwargs = None
                    next_action = "respond_to_user"
                else:
                    # Confirmed — proceed with the pending tool call
                    awaiting_confirmation = False
                    next_action = "tool_call"

            # ------------------------------------------------------------------
            # STEP D: Tool call path
            # ------------------------------------------------------------------
            if next_action == "tool_call":
                task_desc = orchestrator_plan.get("task_description", "")

                # Run Tool Navigator
                nav_result = run_tool_navigator(
                    model=self.model,
                    provider=self.provider,
                    temperature=self.temperature,
                    task_description=task_desc,
                    tools_info=self.tools_info,
                    conversation_history=agent_history,
                    ledger_summary=enforcer.ledger_summary(),
                )

                # If navigator needs more info, respond to user
                if nav_result.get("needs_info"):
                    missing = nav_result.get("missing_info", "some required information")
                    clarify_response = run_response_synthesizer(
                        model=self.model,
                        provider=self.provider,
                        temperature=self.temperature,
                        response_goal=f"Ask user for missing information: {missing}",
                        conversation_history=agent_history,
                        ledger_summary=enforcer.ledger_summary(),
                    )
                    action = Action(
                        name=RESPOND_ACTION_NAME,
                        kwargs={RESPOND_ACTION_FIELD_NAME: clarify_response}
                    )
                    env_response = env.step(action)
                    reward = env_response.reward
                    info = {**info, **env_response.info.model_dump()}
                    messages.extend([
                        {"role": "assistant", "content": clarify_response},
                        {"role": "user", "content": env_response.observation},
                    ])
                    agent_history.extend([
                        {"role": "assistant", "content": clarify_response},
                        {"role": "user", "content": env_response.observation},
                    ])
                    if env_response.done:
                        break
                    continue

                tool_name = nav_result.get("tool_name")
                tool_kwargs = nav_result.get("arguments", {})

                if not tool_name:
                    # Navigator failed to select a tool — respond to user
                    next_action = "respond_to_user"
                else:
                    # ----------------------------------------------------------
                    # DETERMINISTIC ENFORCEMENT GATE
                    # ----------------------------------------------------------
                    last_user_msg = _extract_user_message(agent_history)
                    user_confirmed = enforcer.classify_confirmation(last_user_msg)

                    proceed, block_reason = enforcer.gate(
                        tool_name=tool_name,
                        kwargs=tool_kwargs,
                        tools_info=self.tools_info,
                        user_confirmed=user_confirmed,
                    )

                    if not proceed:
                        # Enforcement blocked the call
                        # If it's a confirmation block, request confirmation
                        if "confirmation" in block_reason.lower():
                            awaiting_confirmation = True
                            pending_tool_name = tool_name
                            pending_tool_kwargs = tool_kwargs

                            confirm_response = run_response_synthesizer(
                                model=self.model,
                                provider=self.provider,
                                temperature=self.temperature,
                                response_goal=f"Ask user to confirm the action: {tool_name} with {tool_kwargs}",
                                conversation_history=agent_history,
                                ledger_summary=enforcer.ledger_summary(),
                            )
                            action = Action(
                                name=RESPOND_ACTION_NAME,
                                kwargs={RESPOND_ACTION_FIELD_NAME: confirm_response}
                            )
                            env_response = env.step(action)
                            reward = env_response.reward
                            info = {**info, **env_response.info.model_dump()}
                            messages.extend([
                                {"role": "assistant", "content": confirm_response},
                                {"role": "user", "content": env_response.observation},
                            ])
                            agent_history.extend([
                                {"role": "assistant", "content": confirm_response},
                                {"role": "user", "content": env_response.observation},
                            ])
                            if env_response.done:
                                break
                            continue
                        else:
                            # Other policy block — inform and re-plan
                            block_response = run_response_synthesizer(
                                model=self.model,
                                provider=self.provider,
                                temperature=self.temperature,
                                response_goal=f"Inform user of policy constraint: {block_reason}",
                                conversation_history=agent_history,
                                ledger_summary=enforcer.ledger_summary(),
                            )
                            action = Action(
                                name=RESPOND_ACTION_NAME,
                                kwargs={RESPOND_ACTION_FIELD_NAME: block_response}
                            )
                            env_response = env.step(action)
                            reward = env_response.reward
                            info = {**info, **env_response.info.model_dump()}
                            messages.extend([
                                {"role": "assistant", "content": block_response},
                                {"role": "user", "content": env_response.observation},
                            ])
                            agent_history.extend([
                                {"role": "assistant", "content": block_response},
                                {"role": "user", "content": env_response.observation},
                            ])
                            if env_response.done:
                                break
                            continue

                    # ----------------------------------------------------------
                    # EXECUTE TOOL CALL via tau-bench environment
                    # ----------------------------------------------------------
                    action = Action(name=tool_name, kwargs=tool_kwargs)
                    env_response = env.step(action)
                    reward = env_response.reward
                    info = {**info, **env_response.info.model_dump()}
                    tool_result = env_response.observation

                    # Update auth state if this was an auth tool
                    if enforcer.is_auth_tool(tool_name):
                        enforcer.authenticated = True

                    # Track certificate usage
                    if "certificate" in str(tool_kwargs).lower():
                        enforcer.certificates_used += 1

                    # ----------------------------------------------------------
                    # RUN EXECUTION & STATE TRACKER
                    # ----------------------------------------------------------
                    tracker_result = run_execution_tracker(
                        model=self.model,
                        provider=self.provider,
                        temperature=self.temperature,
                        tool_name=tool_name,
                        tool_result=tool_result,
                        ledger_summary=enforcer.ledger_summary(),
                    )

                    # Update ledger with verified facts
                    for k, v in tracker_result.get("ledger_updates", {}).items():
                        enforcer.record(k, v)

                    # Update auth status from tracker
                    if tracker_result.get("auth_completed"):
                        enforcer.authenticated = True

                    # Add tool call to messages (tau-bench trajectory format)
                    import uuid
                    tool_call_id = str(uuid.uuid4())[:8]
                    tool_message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_kwargs)
                            }
                        }]
                    }
                    messages.extend([
                        tool_message,
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": tool_result,
                        }
                    ])
                    agent_history.append({
                        "role": "assistant",
                        "content": f"[Called {tool_name}] Result: {tool_result[:300]}"
                    })

                    if env_response.done:
                        break

                    # After tool call, synthesize a brief update to user if needed
                    tracker_summary = tracker_result.get("summary", "")
                    if tracker_summary:
                        update_response = run_response_synthesizer(
                            model=self.model,
                            provider=self.provider,
                            temperature=self.temperature,
                            response_goal=f"Brief update to user after tool call: {tracker_summary}",
                            conversation_history=agent_history,
                            ledger_summary=enforcer.ledger_summary(),
                        )
                        action = Action(
                            name=RESPOND_ACTION_NAME,
                            kwargs={RESPOND_ACTION_FIELD_NAME: update_response}
                        )
                        env_response2 = env.step(action)
                        reward = env_response2.reward
                        info = {**info, **env_response2.info.model_dump()}
                        messages.extend([
                            {"role": "assistant", "content": update_response},
                            {"role": "user", "content": env_response2.observation},
                        ])
                        agent_history.extend([
                            {"role": "assistant", "content": update_response},
                            {"role": "user", "content": env_response2.observation},
                        ])
                        if env_response2.done:
                            break
                    continue

            # ------------------------------------------------------------------
            # STEP E: Respond to user path
            # ------------------------------------------------------------------
            response_goal = orchestrator_plan.get(
                "response_needed",
                "Help the user with their request based on the conversation and state ledger."
            )
            final_response = run_response_synthesizer(
                model=self.model,
                provider=self.provider,
                temperature=self.temperature,
                response_goal=response_goal,
                conversation_history=agent_history,
                ledger_summary=enforcer.ledger_summary(),
            )

            action = Action(
                name=RESPOND_ACTION_NAME,
                kwargs={RESPOND_ACTION_FIELD_NAME: final_response}
            )
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}

            messages.extend([
                {"role": "assistant", "content": final_response},
                {"role": "user", "content": env_response.observation},
            ])
            agent_history.extend([
                {"role": "assistant", "content": final_response},
                {"role": "user", "content": env_response.observation},
            ])

            if env_response.done:
                break

        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )